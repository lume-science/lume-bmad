import h5py
import pytest

from lume_bmad.utils import tao_from_archive


def test_tao_from_archive_rejects_unknown_format(tmp_path):
    archive_path = tmp_path / "unknown.h5"
    with h5py.File(archive_path, "w"):
        pass
    with pytest.raises(ValueError, match="Not a supported"):
        tao_from_archive(archive_path)


@pytest.mark.parametrize("missing", ["run_tao.lat.bmad", "run_tao.cmd"])
def test_tao_from_archive_requires_lattice_and_commands(tmp_path, missing):
    archive_path = tmp_path / "incomplete.h5"
    with h5py.File(archive_path, "w") as archive:
        archive.attrs["dataType"] = "lume-bmad"
        archive.attrs["archive_version"] = 2
        for name in ("run_tao.lat.bmad", "run_tao.cmd", "initial_particles"):
            if name != missing:
                archive.create_dataset(name, data="unused")
    with pytest.raises(ValueError, match="missing required entry"):
        tao_from_archive(archive_path)
