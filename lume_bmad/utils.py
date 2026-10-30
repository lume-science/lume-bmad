from lume_bmad.actions import CombStatVariable, StatVariable
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Any, TYPE_CHECKING
import logging
import warnings
import weakref

import h5py
import numpy as np
from beamphysics import ParticleGroup
from pytao import Tao
from lume_bmad.archive import (
    _archive_variable_entries,
    _write_variable_metadata,
    _write_variables,
)
from lume.variables import NDVariable

if TYPE_CHECKING:
    from lume_bmad.model import LUMEBmadModel

logger = logging.getLogger(__name__)


def archive_model(
    model: "LUMEBmadModel",
    archive_path: str | Path,
    *,
    include_variable_values: bool = True,
) -> Path:
    """Write the current model state to a single HDF5 file.

    The file contains UTF-8 datasets ``run_tao.cmd``, ``run_tao.lat.bmad``,
    and ``run_tao.sh``, a name-keyed ``variables`` group readable with
    ``lume_bmad.archive.read_archive_variables``, and optional
    ``initial_particles`` / ``final_particles`` openPMD groups.
    Each variable entry has a ``variable_type`` (qualified class name)
    attribute and native attributes for its LUME descriptor fields.
    Descriptor fields whose value is None use HDF5 null attributes.

    Parent directories are created as needed. An existing archive is
    replaced only after the new archive has been written successfully.
    No tracking mode changes are made to obtain unavailable particles.
    Failed variable or particle reads are logged and saved as NaN placeholders
    with an ``error`` attribute. Array variables retain their declared shape.
    Successful reads refresh the model cache; failed reads do not replace it.
    With ``include_variable_values=False``, skip variable reads and cache updates
    but retain descriptor metadata in groups marked ``kind="omitted"``.
    The initial/final particle distributions and Tao files are unaffected.

    Returns
    -------
    Path
        Absolute path to the completed archive.
    """
    archive_path = Path(archive_path).expanduser().resolve()
    values: dict[str, Any] = {}
    errors: dict[str, str] = {}
    for name, variable in model.supported_variables.items():
        if not include_variable_values:
            values[name] = None
            continue
        try:
            value = variable._get(model.tao)
            variable.validate_value(value)
        except Exception as exc:
            logger.exception("Unable to read archive variable %s", name)
            errors[name] = f"{type(exc).__name__}: {exc}"
            values[name] = (
                np.full(variable.shape, np.nan)
                if isinstance(variable, NDVariable)
                else np.nan
            )
        else:
            values[name] = value
            model._state[name] = value

    distributions: dict[str, ParticleGroup | None] = {}
    particle_errors: dict[str, str] = {}
    for name in ("initial_particles", "final_particles"):
        try:
            distributions[name] = getattr(model, name)
        except Exception as exc:
            logger.exception("Unable to read archive distribution %s", name)
            particle_errors[name] = f"{type(exc).__name__}: {exc}"

    with TemporaryDirectory(prefix="lume-bmad-") as directory:
        # Tao's lattice writer cannot handle whitespace in its output path.
        if any(character.isspace() for character in directory):
            raise ValueError(
                "Tao archive staging requires a temporary directory without "
                "whitespace. Set TMPDIR to a path without spaces."
            )
        sh_file, cmd_file = model.tao.archive(directory)
        scripts = {
            path.name: path.read_text()
            for path in (
                cmd_file,
                Path(directory) / "run_tao.lat.bmad",
                sh_file,
            )
        }

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        dir=archive_path.parent, prefix=".lume-bmad-", suffix=".h5", delete=False
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)
    try:
        with h5py.File(temporary_path, "w") as archive:
            archive.attrs["dataType"] = "lume-bmad"
            archive.attrs["archive_version"] = 2
            for name, content in scripts.items():
                archive.create_dataset(
                    name, data=content, dtype=h5py.string_dtype("utf-8")
                )
            variables = archive.create_group("variables", track_order=True)
            _write_variables(variables, values)
            for node in variables.values():
                name = node.attrs["name"]
                if not include_variable_values:
                    node.attrs["kind"] = "omitted"
                variable = model.supported_variables[name]
                variable_type = type(variable)
                node.attrs["variable_type"] = (
                    f"{variable_type.__module__}.{variable_type.__qualname__}"
                )
                if name in errors:
                    node.attrs["error"] = errors[name]
                _write_variable_metadata(node, variable.model_dump(mode="json"))
            for name, error in particle_errors.items():
                node = archive.create_dataset(name, data=np.nan)
                node.attrs["error"] = error
            for name, particles in distributions.items():
                if particles is not None:
                    particles.write(archive.create_group(name))
        temporary_path.replace(archive_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return archive_path


def tao_from_archive(archive_path: str | Path) -> Tao:
    """Restore a Tao session and its initial beam from a LUME-Bmad HDF5 archive.

    The archive must contain ``run_tao.lat.bmad``, ``run_tao.cmd``, and
    ``initial_particles``. Archived particle-file paths are ignored in favor
    of the embedded ParticleGroup. Tracking settings are restored only after
    installing that distribution, so stale external beam files are never read.
    Archived variable read errors emit warnings; these diagnostic values are
    not applied to Tao. If ``initial_particles`` is an error placeholder, warn
    and restore single-particle mode without loading a beam. Missing particle
    entries and unmarked corrupt distributions still raise errors.

    Temporary lattice and particle files remain available for the lifetime of
    the returned Tao object for subsequent tracking.
    The temporary directory must not contain whitespace. The input archive
    path may contain spaces.

    Only load trusted archives: the archived Tao commands are executed.
    Tao's in-process library supports one active session; creating this object
    replaces any existing in-process Tao simulation.

    Raises
    ------
    ValueError
        If the archive format is unsupported, required entries are missing, or
        the temporary directory contains whitespace.
    """
    with h5py.File(Path(archive_path).expanduser(), "r") as archive:
        if archive.attrs.get("dataType") != "lume-bmad" or archive.attrs.get(
            "archive_version"
        ) not in (1, 2):
            raise ValueError("Not a supported lume-bmad HDF5 archive")
        for name in ("run_tao.lat.bmad", "run_tao.cmd", "initial_particles"):
            if name not in archive:
                raise ValueError(f"Archive is missing required entry {name!r}")
        lattice_text = archive["run_tao.lat.bmad"].asstr()[()]
        commands = archive["run_tao.cmd"].asstr()[()].splitlines()
        if "variables" in archive:
            for name, node in _archive_variable_entries(archive):
                if "error" in node.attrs:
                    warnings.warn(
                        f"Archived variable {name!r} could not be read: "
                        f"{node.attrs['error']}. Its NaN placeholder is not applied to Tao.",
                        UserWarning,
                        stacklevel=2,
                    )
        for name in ("initial_particles", "final_particles"):
            if name in archive and "error" in archive[name].attrs:
                message = f"Archived {name} could not be read: {archive[name].attrs['error']}."
                if name == "initial_particles":
                    message += " Restoring Tao in single-particle mode without a beam."
                warnings.warn(message, UserWarning, stacklevel=2)
        particles = (
            None
            if "error" in archive["initial_particles"].attrs
            else ParticleGroup(h5=archive["initial_particles"])
        )

    directory = TemporaryDirectory(prefix="lume-bmad-restore-")
    try:
        if any(character.isspace() for character in directory.name):
            raise ValueError(
                "Tao restoration requires a temporary directory without whitespace. "
                "Set TMPDIR to a path without spaces."
            )
        lattice_path = Path(directory.name) / "run_tao.lat.bmad"
        lattice_path.write_text(lattice_text)
        tao = Tao(lattice_file=lattice_path, noinit=True, nostartup=True, noplot=True)
        tao.cmd("set global track_type = single")
        tao.cmd("set global lattice_calc_on = F")
        deferred = {}
        for command in commands:
            setting = command.partition("=")[0].strip().lower()
            if setting == "set beam_init position_file":
                continue
            if setting in ("set global track_type", "set global lattice_calc_on"):
                deferred[setting] = command
            elif command.strip():
                tao.cmd(command)
        if particles is not None:
            tao.set_initial_particles(
                particles, write_to=Path(directory.name) / "initial_particles.h5"
            )
            if "set global track_type" in deferred:
                tao.cmd(deferred["set global track_type"])
        else:
            tao.cmd("set beam_init position_file = ''")
        if "set global lattice_calc_on" in deferred:
            tao.cmd(deferred["set global lattice_calc_on"])
        weakref.finalize(tao, directory.cleanup)
        return tao
    except BaseException:
        directory.cleanup()
        raise


TAO_OUTPUT_UNITS = {
    "name": "",
    "s_ele": "m",
    "ix_ele": "",
    "ix_branch": "",
    "a.beta": "m",
    "a.alpha": "",
    "a.eta": "m",
    "a.etap": "",
    "a.gamma": "1/m",
    "a.phi": "",
    "b.beta": "m",
    "b.alpha": "",
    "b.eta": "m",
    "b.etap": "",
    "b.gamma": "1/m",
    "b.phi": "",
    "l": "m",
    "e_tot": "eV",
    "p0c": "eV/c",
    "mat6": "",
    "vec0": "m",
}

TAO_COMB_OUTPUT_UNITS = {
    "s": "m",
    "x": "m",
    "px": "rad",
    "y": "m",
    "py": "rad",
    "t": "s",
    "x.beta": "m",
    "x.alpha": "",
    "x.eta": "m",
    "x.etap": "",
    "x.emit": "m*rad",
    "x.norm_emit": "m*rad",
    "y.beta": "m",
    "y.alpha": "",
    "y.eta": "m",
    "y.etap": "",
    "y.emit": "m*rad",
    "y.norm_emit": "m*rad",
    "n_particle_live": "",
}

TAO_COMB_OUTPUT_NAMES = {f"comb:{key}" for key in TAO_COMB_OUTPUT_UNITS.keys()}

###############################################################
# Utility functions for importing control and output variables
################################################################


def get_tao_stat_output_variables(tao: Tao) -> list[StatVariable]:
    """
    Returns list of StatVariable instances for Tao lattice outputs.

    Parameters
    ----------
    tao: Tao
        Instance of the Tao class.

    Returns
    -------
    list[StatVariable]
        A list of StatVariable instances.

    """
    elements = tao.lat_list("*", "ele.name")
    element_count = len(elements)
    out_list = []

    for parameter_name in TAO_OUTPUT_UNITS.keys():
        if parameter_name in ["name"]:
            # Avoid fixed-width unicode dtypes (<U0, <U12, ...) so any name length is valid.
            data_type_ = np.dtype(object)
        elif parameter_name in ["ix_ele"]:
            data_type_ = np.dtype(np.int32)
        else:
            data_type_ = np.dtype(float)

        if parameter_name == "mat6":
            shape = (element_count, 6, 6)
        elif parameter_name == "vec0":
            shape = (element_count, 6)
        else:
            shape = (element_count,)

        out_list.append(
            StatVariable(
                name=parameter_name,
                statistic_name=parameter_name,
                shape=shape,
                unit=TAO_OUTPUT_UNITS[parameter_name],
                read_only=True,
                dtype=data_type_,
            )
        )

    return out_list


def get_tao_comb_output_variables(tao: Tao) -> list[CombStatVariable]:
    """
    Returns list of CombStatVariable instances for Tao comb outputs.

    Parameters
    ----------
    tao: Tao
        Instance of the Tao class.

    Returns
    -------
    list[CombStatVariable]
        A list of CombStatVariable instances.

    """
    out_list = []

    # handle comb outputs
    if tao.tao_global()["track_type"] == "beam":
        s = tao.bunch_comb("s")
        shape = s.shape
        for parameter_name in TAO_COMB_OUTPUT_UNITS.keys():
            out_list.append(
                CombStatVariable(
                    name=f"comb:{parameter_name}",
                    statistic_name=parameter_name,
                    shape=shape,
                    unit=TAO_COMB_OUTPUT_UNITS[parameter_name],
                    read_only=True,
                    dtype=np.dtype(float),
                )
            )

    return out_list


def get_tao_output_variables(tao: Tao) -> list[StatVariable | CombStatVariable]:
    """
    Returns all available Tao output variables, including lattice and comb outputs.

    Parameters
    ----------
    tao: Tao
        Instance of the Tao class.

    Returns
    -------
    list[StatVariable | CombStatVariable]
        A list of StatVariable and CombStatVariable instances.

    """
    return [
        *get_tao_stat_output_variables(tao),
        *get_tao_comb_output_variables(tao),
    ]


def rmat_get(tao, element_a, element_b, design=False):
    """
    Returns dictionary with Rmat from a to b

    Parameters
    ----------
    tao: Tao
        Instance of the Tao class.

    Returns
    -------
    array (6,6) containing Rmat

    """
    if design:
        element_a = element_a + "|design"
    return tao.matrix(element_a, element_b)["mat6"]
