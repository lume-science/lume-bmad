"""Native HDF5 storage for model variable snapshots."""

from pathlib import Path
from typing import Any
import logging

import h5py
import numpy as np
import pandas as pd
from beamphysics import ParticleGroup

logger = logging.getLogger(__name__)


def _archive_variable_entries(
    archive: h5py.File,
) -> list[tuple[str, h5py.Dataset | h5py.Group]]:
    group = archive["variables"]
    if archive.attrs["archive_version"] == 1:
        return [
            (name, group["values"][str(index)])
            for index, name in enumerate(group["names"].asstr()[()].tolist())
        ]
    return [(node.attrs["name"], node) for node in group.values()]


def _write_variables(group: h5py.Group, values: dict[str, Any]) -> None:
    # Attribute order tracking also enables large array-valued metadata.
    for name, value in values.items():
        if not name or "\x00" in name:
            raise ValueError(f"Variable name {name!r} cannot be stored in HDF5")
        # Escape HDF5 path separators without changing ordinary variable names.
        key = name.replace("%", "%25").replace("/", "%2F")
        if key == ".":
            key = "%2E"
        if isinstance(value, ParticleGroup):
            node = group.create_group(key, track_order=True)
            node.attrs["kind"] = "particle_group"
            value.write(node)
        elif value is None:
            node = group.create_group(key, track_order=True)
            node.attrs["kind"] = "none"
        elif isinstance(value, str):
            node = group.create_dataset(
                key, data=value, dtype=h5py.string_dtype("utf-8"), track_order=True
            )
            node.attrs["kind"] = "string"
        elif isinstance(value, np.ndarray):
            if value.dtype.kind in "UO":
                if not all(isinstance(item, str) for item in value.flat):
                    raise TypeError(
                        f"Variable {name!r} contains unsupported object-array values"
                    )
                node = group.create_dataset(
                    key,
                    data=value.astype(object),
                    dtype=h5py.string_dtype("utf-8"),
                    track_order=True,
                )
                node.attrs["kind"] = "string_array"
                node.attrs["numpy_dtype"] = value.dtype.str
            else:
                node = group.create_dataset(key, data=value, track_order=True)
                node.attrs["kind"] = "array"
        elif isinstance(value, (bool, int, float, complex, np.number, np.bool_)):
            node = group.create_dataset(key, data=value, track_order=True)
            node.attrs["kind"] = "scalar"
        else:
            raise TypeError(
                f"Variable {name!r} has unsupported archive type {type(value).__name__}"
            )
        node.attrs["name"] = name


def _write_variable_metadata(
    node: h5py.Dataset | h5py.Group, metadata: dict[str, Any]
) -> None:
    """Write descriptor fields as native attributes, representing None as null."""
    for key, value in metadata.items():
        if key in node.attrs and not (key == "name" and value == node.attrs[key]):
            raise ValueError(
                f"Metadata field {key!r} conflicts with an archive attribute on {node.name!r}"
            )
        if isinstance(value, dict):
            raise TypeError(
                f"Metadata field {key!r} on {node.name!r} cannot store a nested dict "
                "as an HDF5 attribute"
            )
        node.attrs[key] = h5py.Empty("f8") if value is None else value


def read_archive_variables(archive_path: str | Path) -> pd.DataFrame:
    """Read an archive's ``name``/``value`` dataframe without unpickling objects.

    Array-valued cells retain their shapes and dtypes. Particle-valued cells
    contain ParticleGroups loaded from their native openPMD HDF5 groups.
    Both named-entry archives and the original indexed layout are supported.
    Failed reads retain NaN placeholders and are logged. Their saved error
    messages are available in ``frame.attrs["errors"]``, keyed by variable name.
    Intentionally omitted values become ``pd.NA`` and their names are listed in
    ``frame.attrs["omitted_values"]``; they are not read errors.
    """
    with h5py.File(Path(archive_path).expanduser(), "r") as archive:
        if archive.attrs.get("dataType") != "lume-bmad" or archive.attrs.get(
            "archive_version"
        ) not in (1, 2):
            raise ValueError("Not a supported lume-bmad HDF5 archive")
        entries = _archive_variable_entries(archive)
        names = []
        values = []
        errors = {}
        omitted_values = []
        for name, node in entries:
            if "error" in node.attrs:
                errors[name] = node.attrs["error"]
                logger.warning(
                    "Archived variable %s read failed: %s", name, errors[name]
                )
            kind = node.attrs["kind"]
            if kind == "omitted":
                value = pd.NA
                omitted_values.append(name)
            elif kind == "particle_group":
                value = ParticleGroup(h5=node)
            elif kind == "none":
                value = None
            elif kind == "string":
                value = node.asstr()[()]
            elif kind == "string_array":
                value = np.asarray(node.asstr()[()], dtype=node.attrs["numpy_dtype"])
            elif kind in ("array", "scalar"):
                value = node[()]
                if kind == "array":
                    value = np.asarray(value)
            else:
                raise ValueError(f"Variable {name!r} has unknown archive kind {kind!r}")
            names.append(name)
            values.append(value)
    frame = pd.DataFrame({"name": names, "value": pd.Series(values, dtype=object)})
    frame.attrs["errors"] = errors
    frame.attrs["omitted_values"] = omitted_values
    return frame
