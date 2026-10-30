import h5py
import numpy as np
import pytest

from lume_bmad.archive import (
    _write_variable_metadata,
    _write_variables,
    read_archive_variables,
)


def test_variable_round_trip(tmp_path):
    values = {
        "float": 0.25,
        "int": 7,
        "bool": True,
        "complex": 1 + 2j,
        "numpy_scalar": np.float32(0.5),
        "string": "beam",
        "none": None,
        "matrix": np.arange(24, dtype=np.int32).reshape(2, 3, 4),
        "unicode": np.array(["beam", "\u03b2"], dtype="U4"),
        "object_strings": np.array(["qf", "qd"], dtype=object),
        "bytes": np.array([b"qf", b"qd"]),
        "empty": np.empty((0, 3)),
        "empty_strings": np.array([], dtype=object),
        "zero_dimensional": np.array(3.0),
        "zero_dimensional_string": np.array("qf"),
        "nan": np.nan,
        "infinity": np.inf,
        "name/with:separators": np.array([1.0, 2.0]),
        "name%2Fwith:separators": 3.0,
        "qf:B1_GRADIENT": 0.2,
        ".": 1.0,
        "%2E": 2.0,
        "/": 3.0,
        "names": 4.0,
        "values": 5.0,
    }
    archive_path = tmp_path / "variables.h5"
    with h5py.File(archive_path, "w") as archive:
        archive.attrs["dataType"] = "lume-bmad"
        archive.attrs["archive_version"] = 2
        _write_variables(archive.create_group("variables", track_order=True), values)

    with h5py.File(archive_path, "r") as archive:
        group = archive["variables"]
        assert len(group) == len(values)
        assert group["qf:B1_GRADIENT"][()] == 0.2
        assert group["names"][()] == 4.0
        assert group["values"][()] == 5.0
        assert group["name%2Fwith:separators"].attrs["name"] == "name/with:separators"
        assert group["name%252Fwith:separators"][()] == 3.0
        assert group["%2E"][()] == 1.0
        assert group["%252E"][()] == 2.0
        assert group["%2F"][()] == 3.0

    frame = read_archive_variables(archive_path)
    assert frame["name"].tolist() == list(values)
    for name, actual in frame.itertuples(index=False, name=None):
        expected = values[name]
        if expected is None:
            assert actual is None
        else:
            np.testing.assert_equal(actual, expected)
        if isinstance(expected, np.ndarray):
            assert isinstance(actual, np.ndarray)
            assert actual.dtype == expected.dtype
            assert actual.shape == expected.shape


@pytest.mark.parametrize(
    "value", [object(), {"a": 1}, np.array([1, "a"], dtype=object)]
)
def test_unsupported_variable_type(tmp_path, value):
    with h5py.File(tmp_path / "variables.h5", "w") as archive:
        with pytest.raises(TypeError, match="Variable 'unsupported'"):
            _write_variables(archive.create_group("variables"), {"unsupported": value})


def test_empty_variables(tmp_path):
    archive_path = tmp_path / "variables.h5"
    with h5py.File(archive_path, "w") as archive:
        archive.attrs["dataType"] = "lume-bmad"
        archive.attrs["archive_version"] = 2
        _write_variables(archive.create_group("variables"), {})
    frame = read_archive_variables(archive_path)
    assert frame.empty
    assert list(frame.columns) == ["name", "value"]


def test_reject_unknown_archive_format(tmp_path):
    archive_path = tmp_path / "unknown.h5"
    with h5py.File(archive_path, "w"):
        pass
    with pytest.raises(ValueError, match="Not a supported"):
        read_archive_variables(archive_path)


def test_reject_unknown_variable_kind(tmp_path):
    archive_path = tmp_path / "variables.h5"
    with h5py.File(archive_path, "w") as archive:
        archive.attrs["dataType"] = "lume-bmad"
        archive.attrs["archive_version"] = 2
        _write_variables(archive.create_group("variables"), {"value": 1.0})
        archive["variables/value"].attrs["kind"] = "unknown"
    with pytest.raises(ValueError, match="unknown archive kind"):
        read_archive_variables(archive_path)


@pytest.mark.parametrize("name", ["", "invalid\x00name"])
def test_reject_invalid_variable_names(tmp_path, name):
    with h5py.File(tmp_path / "variables.h5", "w") as archive:
        with pytest.raises(ValueError, match="cannot be stored in HDF5"):
            _write_variables(archive.create_group("variables"), {name: 1.0})


def test_read_legacy_indexed_archive(tmp_path):
    archive_path = tmp_path / "legacy.h5"
    with h5py.File(archive_path, "w") as archive:
        archive.attrs["dataType"] = "lume-bmad"
        archive.attrs["archive_version"] = 1
        group = archive.create_group("variables")
        group.create_dataset(
            "names",
            data=["qf:B1_GRADIENT", "track_type", "array"],
            dtype=h5py.string_dtype("utf-8"),
        )
        entries = group.create_group("values")
        entries.create_dataset("0", data=0.2).attrs["kind"] = "scalar"
        entries.create_dataset(
            "1", data="beam", dtype=h5py.string_dtype("utf-8")
        ).attrs["kind"] = "string"
        entries.create_dataset("2", data=np.arange(6).reshape(2, 3)).attrs["kind"] = (
            "array"
        )

    frame = read_archive_variables(archive_path)
    assert frame["name"].tolist() == ["qf:B1_GRADIENT", "track_type", "array"]
    assert frame.iloc[0]["value"] == 0.2
    assert frame.iloc[1]["value"] == "beam"
    np.testing.assert_array_equal(frame.iloc[2]["value"], np.arange(6).reshape(2, 3))


@pytest.mark.parametrize("value", [1.0, None])
def test_native_metadata_attributes(tmp_path, value):
    metadata = {
        "name": "variable",
        "description": "Beam \u03b2",
        "read_only": True,
        "unit": None,
        "shape": [128, 128],
        "dtype": "float64",
        "options": ["single", "beam"],
        "empty": [],
        "value_range": [-np.inf, np.inf],
        "default_value": np.zeros((128, 128)).tolist(),
    }
    archive_path = tmp_path / "metadata.h5"
    with h5py.File(archive_path, "w") as archive:
        group = archive.create_group("variables")
        _write_variables(group, {"variable": value})
        _write_variable_metadata(group["variable"], metadata)

    with h5py.File(archive_path, "r") as archive:
        node = archive["variables/variable"]
        assert "metadata" not in node.attrs
        assert node.attrs["kind"] == ("none" if value is None else "scalar")
        assert node.attrs["default_value"].nbytes > 64 * 1024
        for key, expected in metadata.items():
            if expected is None:
                assert isinstance(node.attrs[key], h5py.Empty)
            else:
                np.testing.assert_equal(node.attrs[key], expected)
        assert not isinstance(node.attrs["empty"], h5py.Empty)


@pytest.mark.parametrize("key", ["kind", "name", "variable_type"])
def test_metadata_cannot_overwrite_archive_attributes(tmp_path, key):
    with h5py.File(tmp_path / "metadata.h5", "w") as archive:
        group = archive.create_group("variables")
        _write_variables(group, {"variable": 1.0})
        node = group["variable"]
        node.attrs["variable_type"] = "lume.variables.ScalarVariable"
        original = node.attrs[key]
        with pytest.raises(ValueError, match="conflicts with an archive attribute"):
            _write_variable_metadata(node, {key: "conflicting value"})
        assert node.attrs[key] == original


def test_metadata_rejects_nested_dict(tmp_path):
    with h5py.File(tmp_path / "metadata.h5", "w") as archive:
        node = archive.create_dataset("variable", data=1.0)
        with pytest.raises(TypeError, match="cannot store a nested dict"):
            _write_variable_metadata(node, {"nested": {"field": 1}})
