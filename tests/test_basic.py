import pytest
import numpy as np
import h5py
import os
import gc
from pathlib import Path
from pytao import Tao, TaoCommandError

from lume_bmad.model import LUMEBmadModel
from lume_bmad.archive import read_archive_variables
from lume_bmad.actions import (
    EleScalarVariable,
    ScreenImageShapeVariable,
    ScreenImageVariable,
    ScreenResolutionVariable,
    ScreenSpec,
)
from lume_bmad.utils import TAO_COMB_OUTPUT_NAMES, archive_model, tao_from_archive
from beamphysics import ParticleGroup

TEST_BEAM_PATH = os.path.join(Path(__file__).parent, "test_beam.h5")


class TestModel:
    @pytest.fixture
    def model(self):
        control_variables = [
            EleScalarVariable(
                name="qf:B1_GRADIENT",
                unit="1/m^2",
                element_name="qf",
                property_name="B1_GRADIENT",
            ),
            EleScalarVariable(
                name="qd:B1_GRADIENT",
                unit="1/m^2",
                element_name="qd",
                property_name="B1_GRADIENT",
            ),
        ]
        tao = Tao(init_file="tests/fodo.init", noplot=True)

        model = LUMEBmadModel(tao, control_variables, dump_locations=["qf", "qd"])
        return model

    def test_model_initialization(self, model):

        # test that beam dump locations are set correctly
        assert model.start_element == "BEGINNING"
        assert model.end_element == "END"

        assert model.tao.beam(0)["saved_at"] == "qf,qd,BEGINNING,END"

        model.set({"qf:B1_GRADIENT": 0.2})
        assert model.get(["qf:B1_GRADIENT"])["qf:B1_GRADIENT"] == 0.2

        model.reset()

    def test_beam_tracking(self, model):
        # set track_type to "beam" to enable tracking
        model.set({"track_type": "beam"})

        output_beam = model.final_particles
        assert isinstance(output_beam, ParticleGroup)
        assert output_beam.n_particle == 1000

        # set the input beam element name and set the input beam
        model.input_beam_element_name = "qf"
        model.initial_particles = output_beam.copy()

        # read the input beam back and check that it is the same as the output beam
        input_beam = model.initial_particles
        assert input_beam == output_beam

        # test that beam is being dumped at the correct locations
        for ele in ["qf", "qd"]:
            beam = model.get([f"{ele}_beam"])[f"{ele}_beam"]
            assert isinstance(beam, ParticleGroup)
            assert beam.n_particle == 1000

        model.reset()

    def test_screen(self, model):
        # set track_type to "beam" to enable tracking
        extra_control_variables = [
            ScreenImageVariable(
                name="qf_screen",
                shape=(100, 100),
                pixel_size=0.002,
                element_name="qf",
            ),
        ]
        for var in extra_control_variables:
            model.register_action_variable(var)
        model.set({"track_type": "beam"})
        qf_screen = model.get(["qf_screen"])["qf_screen"]
        assert isinstance(qf_screen, np.ndarray)
        assert qf_screen.shape == (100, 100)

        model.reset()

    def test_screen_variables_from_shared_spec(self, model):
        screen_spec = ScreenSpec(
            element_name="qf",
            shape=(60, 80),
            pixel_size=0.0015,
        )

        extra_control_variables = [
            ScreenImageVariable.from_screen_spec(
                name="qf_screen_image",
                screen_spec=screen_spec,
            ),
            ScreenResolutionVariable.from_screen_spec(
                name="qf_screen_resolution",
                screen_spec=screen_spec,
            ),
            ScreenImageShapeVariable.from_screen_spec(
                name="qf_screen_shape_x",
                screen_spec=screen_spec,
                index=0,
            ),
            ScreenImageShapeVariable.from_screen_spec(
                name="qf_screen_shape_y",
                screen_spec=screen_spec,
                index=1,
            ),
        ]

        for var in extra_control_variables:
            model.register_action_variable(var)

        values = model.get(
            [
                "qf_screen_image",
                "qf_screen_resolution",
                "qf_screen_shape_x",
                "qf_screen_shape_y",
            ]
        )
        assert values["qf_screen_image"].shape == (60, 80)
        assert np.allclose(values["qf_screen_resolution"], 0.0015)
        assert values["qf_screen_shape_x"] == 60
        assert values["qf_screen_shape_y"] == 80

        model.reset()

    def test_mat6_output(self, model):
        # test that mat6 output variable is being read and has correct shape
        mat6 = model.get(["mat6"])["mat6"]
        assert isinstance(mat6, np.ndarray)
        assert mat6.shape == (len(model.tao.lat_list("*", "ele.name")), 6, 6)

    def test_vec0_output(self, model):
        # test that vec0 output variable is being read and has correct shape
        vec0 = model.get(["vec0"])["vec0"]
        assert isinstance(vec0, np.ndarray)
        assert vec0.shape == (len(model.tao.lat_list("*", "ele.name")), 6)

    def test_track_type_toggle_updates_beam_state(self, model):
        # start in beam tracking mode and verify beam outputs are populated
        model.set({"track_type": "beam"})
        tracked = model.get(["track_type", "qf_beam", "qd_beam"])
        assert tracked["track_type"] == "beam"
        assert isinstance(tracked["qf_beam"], ParticleGroup)
        assert isinstance(tracked["qd_beam"], ParticleGroup)

        # switching back to single-particle mode should clear beam dumps
        model.set({"track_type": "single"})
        assert model.get(["track_type"])["track_type"] == "single"

        # In single mode update_state removes the beam dumps from the list of supported variables
        for var in ["qf_beam", "qd_beam"]:
            with pytest.raises(ValueError):
                model.get([var])

        assert model.initial_particles is None
        assert model.final_particles is None

        model.reset()

    def test_reset_maintains_current_tracking_mode(self, model):
        # model starts in single-particle tracking mode by default
        assert model.get(["track_type"])["track_type"] == "single"

        # switch to beam tracking mode and reset the model
        model.set({"track_type": "beam"})
        model.reset()

        # reset should not revert track_type back to its initial value
        assert model.get(["track_type"])["track_type"] == "beam"

        # switch back to single-particle mode and reset again
        model.set({"track_type": "single"})
        model.reset()

        assert model.get(["track_type"])["track_type"] == "single"

    def test_supported_variables_contains_model_interfaces(self, model):
        supported = model.supported_variables
        expected = {
            "qf:B1_GRADIENT",
            "qd:B1_GRADIENT",
            "track_type",
            "name",
            "mat6",
            "vec0",
        }
        assert expected.issubset(set(supported.keys()))

    def test_name_output_uses_object_dtype(self, model):
        names = model.get(["name"])["name"]
        assert isinstance(names, np.ndarray)
        assert names.dtype == object
        assert len(names) == len(model.tao.lat_list("*", "ele.name"))

    def test_comb_ds_save_setting(self, model):
        def assert_comb_matches_tao():
            name = "comb:x.beta"
            actual = model.get([name])[name]
            expected = model.tao.bunch_comb("x.beta")

            assert isinstance(actual, np.ndarray)
            assert actual.ndim == 1
            assert actual.size > 0
            assert actual.shape == expected.shape
            np.testing.assert_allclose(actual, expected)
            assert tuple(model.supported_variables[name].shape) == actual.shape
            assert model.tao.beam(0)["comb_ds_save"] == pytest.approx(
                model.comb_ds_save
            )

        supported = model.supported_variables
        expected = {
            "qf:B1_GRADIENT",
            "qd:B1_GRADIENT",
            "track_type",
            "name",
            "mat6",
            "vec0",
        }
        assert expected.issubset(set(supported.keys()))

        # setting track_type to beam should add to the list of expected pvs
        model.set({"track_type": "beam"})
        supported = model.supported_variables
        expected.update(TAO_COMB_OUTPUT_NAMES)
        assert expected.issubset(set(supported.keys()))

        assert_comb_matches_tao()

        # setting track_type back to single should remove comb output variables
        model.set({"track_type": "single"})
        supported = model.supported_variables
        expected.difference_update(TAO_COMB_OUTPUT_NAMES)
        assert expected.issubset(set(supported.keys()))
        assert set(TAO_COMB_OUTPUT_NAMES).isdisjoint(supported)

        # try to get a comb output -- should raise error since they should no longer be supported
        with pytest.raises(ValueError):
            model.get(["comb:x.beta"])

        # Particle replacement must keep comb values and shapes synchronized with Tao.
        model.set({"track_type": "beam"})
        model.initial_particles = ParticleGroup(TEST_BEAM_PATH)

        assert_comb_matches_tao()

        model.reset()

    @pytest.mark.parametrize("sample_counts", [(3, 7), (7, 3)])
    def test_initial_particles_refresh_comb_shapes(
        self, model, monkeypatch, sample_counts
    ):
        samples = np.linspace(0.0, 2.5, sample_counts[0])
        monkeypatch.setattr(
            model.tao, "bunch_comb", lambda statistic_name: samples.copy()
        )
        model.set({"track_type": "beam"})
        for name in TAO_COMB_OUTPUT_NAMES:
            assert tuple(model.supported_variables[name].shape) == samples.shape
            np.testing.assert_array_equal(model.get([name])[name], samples)

        samples = np.linspace(0.0, 2.5, sample_counts[1])
        model.initial_particles = ParticleGroup(TEST_BEAM_PATH)

        for name in TAO_COMB_OUTPUT_NAMES:
            assert tuple(model.supported_variables[name].shape) == samples.shape
            np.testing.assert_array_equal(model.get([name])[name], samples)

    def test_getting_all_variables(self, model):
        variable_names = list(model.supported_variables.keys())

        model.get(variable_names)

        # run the model in beam tracking mode and then try to get all variables again, including beam variables
        model.set({"track_type": "beam"})
        model.get(list(model.supported_variables.keys()))
        model.reset()

    def test_setting_initial_particles_updates_state(self, model):
        model.set({"track_type": "beam"})
        particles = ParticleGroup(TEST_BEAM_PATH)

        # add a dummy value that should be updated when the state is updated after setting the initial particles
        model._state["mat6"] = np.zeros(
            (len(model.tao.lat_list("*", "ele.name")), 6, 6)
        )

        # after setting the initial particles, the model state should be updated to reflect the new beam
        model.initial_particles = particles
        assert model.initial_particles == particles
        assert not np.all(model._state["mat6"] == 0)

        model.reset()

    def test_setting_initial_particles_zeros_z_coordinates(self, model):
        model.set({"track_type": "beam"})
        particles = ParticleGroup(TEST_BEAM_PATH)

        # Ensure the input distribution has nonzero z values before setting it.
        particles.z[:] = np.linspace(-1e-3, 1e-3, particles.n_particle)
        assert np.any(particles.z != 0.0)

        model.initial_particles = particles
        input_beam = model.initial_particles

        assert np.all(input_beam.z == 0.0)

        model.reset()

    def test_setting_initial_particles_before_first_beam_track(
        self, model, tmp_path, monkeypatch
    ):
        workdir = tmp_path / "directory with spaces"
        workdir.mkdir()
        monkeypatch.chdir(workdir)
        model.tao.cmd("set global lattice_calc_on = F")
        model.tao.cmd("set beam_init position_file = /missing/old_beam.h5")
        model.tao.cmd("set global track_type = beam")
        p0c = model.tao.ele_gen_attribs("BEGINNING")["P0C"]
        particles = ParticleGroup(
            data=dict(
                x=[0.0],
                y=[0.0],
                z=[0.0],
                t=[0.0],
                px=[0.0],
                py=[0.0],
                pz=[p0c],
                weight=[5e-12],
                status=[1],
                species="electron",
            )
        )
        model.initial_particles = particles
        assert model.tao.tao_global()["track_type"] == "beam"
        assert model.tao.tao_global()["lattice_calc_on"]
        assert model.initial_particles.n_particle == 1
        assert model.final_particles.n_particle == 1
        assert model.get_value("qf_beam").n_particle == 1
        assert not (workdir / "input_beam.h5").exists()
        assert Path(model.tao.beam_init(0)["position_file"]).is_file()

    def test_recovery_from_bad_bmad(self, model):
        model.set({"qf:B1_GRADIENT": 0.2})
        cached_state = model.get(list(model.supported_variables.keys()))

        # a NaN value passes lume's own validation but is rejected by Tao,
        # triggering a TaoCommandError inside LUMEBmadModel._set
        with pytest.raises(TaoCommandError):
            model.set({"qf:B1_GRADIENT": float("nan")})

        # the model should have recovered by restoring the previously cached state
        restored_state = model.get(list(model.supported_variables.keys()))
        assert restored_state["qf:B1_GRADIENT"] == cached_state["qf:B1_GRADIENT"]
        for name, value in cached_state.items():
            if isinstance(value, np.ndarray):
                assert np.array_equal(restored_state[name], value, equal_nan=True)
            else:
                assert restored_state[name] == value

        # lattice calculations should be re-enabled after recovering
        assert model.tao.tao_global()["lattice_calc_on"] is True

        model.reset()

    def test_recovery_from_bad_bmad_is_atomic(self, model):
        model.set({"qf:B1_GRADIENT": 0.2, "qd:B1_GRADIENT": -0.2})

        # qf is set successfully before qd fails; the whole batch should
        # still roll back, including the already-applied qf change
        with pytest.raises(TaoCommandError):
            model.set({"qf:B1_GRADIENT": 0.5, "qd:B1_GRADIENT": float("nan")})

        restored = model.get(["qf:B1_GRADIENT", "qd:B1_GRADIENT"])
        assert restored["qf:B1_GRADIENT"] == 0.2
        assert restored["qd:B1_GRADIENT"] == -0.2

        # confirm the underlying Tao simulator was actually rolled back, not
        # just the model's cached state
        assert model.tao.ele_gen_attribs("qf")["B1_GRADIENT"] == 0.2
        assert model.tao.ele_gen_attribs("qd")["B1_GRADIENT"] == -0.2

        model.reset()

    def test_model_usable_after_recovery(self, model):
        model.set({"track_type": "beam"})

        # trigger a recovery while in beam-tracking mode
        with pytest.raises(TaoCommandError):
            model.set({"qf:B1_GRADIENT": float("nan")})

        # the model should still be in beam mode and accept further valid sets
        assert model.get(["track_type"])["track_type"] == "beam"
        model.set({"qf:B1_GRADIENT": 0.3})
        assert model.get(["qf:B1_GRADIENT"])["qf:B1_GRADIENT"] == 0.3

        # beam outputs should still be readable after recovering
        beam = model.get(["qf_beam"])["qf_beam"]
        assert isinstance(beam, ParticleGroup)

        model.reset()

    def test_archive(self, model, tmp_path):
        model.set({"track_type": "beam"})
        archive_path = tmp_path / "model.h5"
        assert model.archive(archive_path) == archive_path
        with h5py.File(archive_path, "r") as archive:
            assert "run_tao.lat.pals.yaml" not in archive
            assert (
                ParticleGroup(h5=archive["initial_particles"])
                == model.initial_particles
            )
            assert ParticleGroup(h5=archive["final_particles"]) == model.final_particles
        frame = read_archive_variables(archive_path)
        assert frame["name"].tolist() == list(model.supported_variables)

    def test_tao_from_archive(self, model, tmp_path):
        model.set({"track_type": "beam", "qf:B1_GRADIENT": 0.3})
        expected_initial = model.initial_particles
        expected_final = model.final_particles
        expected_beam = model.tao.beam(0)
        archive_path = model.archive(tmp_path / "archive with spaces" / "model.h5")
        with h5py.File(archive_path, "r+") as archive:
            commands = archive["run_tao.cmd"].asstr()[()]
            commands = "\n".join(
                "set beam_init position_file = /missing/old_particles.h5"
                if line.startswith("set beam_init position_file =")
                else line
                for line in commands.splitlines()
            )
            archive["run_tao.cmd"][()] = commands

        restored = tao_from_archive(archive_path)
        assert restored.tao_global()["track_type"] == "beam"
        assert restored.ele_gen_attribs("qf")["B1_GRADIENT"] == 0.3
        for key in ("track_start", "track_end", "saved_at", "comb_ds_save"):
            assert restored.beam(0)[key] == expected_beam[key]
        particle_file = Path(restored.beam_init(0)["position_file"])
        assert particle_file.exists()
        assert ParticleGroup(particle_file) == expected_initial
        archive_path.unlink()
        restored.cmd("set global track_type = single")
        restored.cmd("set global track_type = beam")
        for actual, expected in (
            (restored.get_initial_particles(), expected_initial),
            (restored.particles("END"), expected_final),
        ):
            assert actual.n_particle == expected.n_particle
            assert actual.species == expected.species
            for key in ("x", "px", "y", "py", "z", "pz", "t", "weight"):
                np.testing.assert_allclose(
                    actual[key], expected[key], rtol=1e-9, atol=1e-12
                )
        del restored
        gc.collect()
        assert not particle_file.parent.exists()

    def test_tao_from_archive_requires_initial_particles(self, model, tmp_path):
        model.set({"track_type": "single"})
        archive_path = model.archive(tmp_path / "single.h5")
        with pytest.raises(
            ValueError, match="missing required entry 'initial_particles'"
        ):
            tao_from_archive(archive_path)

    def test_tao_from_archive_cleans_up_on_command_error(
        self, model, tmp_path, monkeypatch
    ):
        model.set({"track_type": "beam"})
        archive_path = model.archive(tmp_path / "invalid_command.h5")
        with h5py.File(archive_path, "r+") as archive:
            archive["run_tao.cmd"][()] = "set global invalid_archive_setting = 1"
        monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
        with pytest.raises(TaoCommandError):
            tao_from_archive(archive_path)
        assert not list(tmp_path.glob("lume-bmad-restore-*"))

    @pytest.mark.parametrize("track_type", ["single", "beam"])
    @pytest.mark.parametrize("writer", [archive_model, LUMEBmadModel.archive])
    def test_archive_round_trip(self, model, tmp_path, monkeypatch, track_type, writer):
        gradient = model.supported_variables["qf:B1_GRADIENT"]
        gradient.description = "Quadrupole gradient"
        gradient.value_range = (-0.5, 0.5)
        gradient.default_value = 0.2
        gradient.default_validation_config = "warn"
        matrix = model.supported_variables["mat6"]
        matrix.default_value = np.zeros(matrix.shape, dtype=matrix.dtype)
        model.set({"track_type": track_type, "qf:B1_GRADIENT": 0.3})
        model.tao.cmd("set ele qf B1_GRADIENT = 0.4")
        expected = {
            name: variable._get(model.tao)
            for name, variable in model.supported_variables.items()
        }
        initial_particles = model.initial_particles
        final_particles = model.final_particles
        scripts = {}
        staging_paths = []
        tao_archive = model.tao.archive
        tao_cmd = model.tao.cmd

        def capture_archive(directory):
            staging_paths.append(Path(directory))
            result = tao_archive(directory)
            for name in ("run_tao.cmd", "run_tao.lat.bmad", "run_tao.sh"):
                scripts[name] = (Path(directory) / name).read_text()
            return result

        def capture_cmd(command, *args, **kwargs):
            assert not command.startswith("write pals")
            return tao_cmd(command, *args, **kwargs)

        monkeypatch.setattr(model.tao, "archive", capture_archive)
        monkeypatch.setattr(model.tao, "cmd", capture_cmd)
        archive_path = tmp_path / "archive with spaces" / "model.h5"
        assert writer(model, archive_path) == archive_path
        assert list(archive_path.parent.iterdir()) == [archive_path]
        assert all(not path.exists() for path in staging_paths)
        assert set(scripts) == {
            "run_tao.cmd",
            "run_tao.lat.bmad",
            "run_tao.sh",
        }

        with h5py.File(archive_path, "r") as archive:
            assert archive.attrs["archive_version"] == 2
            assert list(archive["variables"]) == list(expected)
            assert archive["variables/qf:B1_GRADIENT"][()] == 0.4
            assert archive["variables/track_type"].asstr()[()] == track_type
            for node in archive["variables"].values():
                variable = model.supported_variables[node.attrs["name"]]
                variable_type = type(variable)
                assert node.attrs["variable_type"] == (
                    f"{variable_type.__module__}.{variable_type.__qualname__}"
                )
                assert "metadata" not in node.attrs
                for field, value in variable.model_dump(mode="json").items():
                    if value is None:
                        assert isinstance(node.attrs[field], h5py.Empty)
                    else:
                        np.testing.assert_equal(node.attrs[field], value)
            gradient_metadata = archive["variables/qf:B1_GRADIENT"].attrs
            np.testing.assert_array_equal(gradient_metadata["value_range"], [-0.5, 0.5])
            assert gradient_metadata["default_value"] == 0.2
            assert gradient_metadata["default_validation_config"] == "warn"
            matrix_metadata = archive["variables/mat6"].attrs
            np.testing.assert_array_equal(matrix_metadata["shape"], matrix.shape)
            assert matrix_metadata["dtype"] == matrix.dtype.name
            np.testing.assert_array_equal(
                matrix_metadata["default_value"], matrix.default_value
            )
            expected_keys = set(scripts) | {"variables"}
            if track_type == "beam":
                assert (
                    ParticleGroup(h5=archive["variables/qf_beam"])
                    == expected["qf_beam"]
                )
                expected_keys |= {"initial_particles", "final_particles"}
                assert (
                    ParticleGroup(h5=archive["initial_particles"]) == initial_particles
                )
                assert ParticleGroup(h5=archive["final_particles"]) == final_particles
            assert set(archive) == expected_keys
            for name, content in scripts.items():
                assert archive[name].asstr()[()] == content
                assert content

        frame = read_archive_variables(archive_path)
        assert list(frame.columns) == ["name", "value"]
        assert frame["name"].tolist() == list(expected)
        for name, actual in frame.itertuples(index=False, name=None):
            value = expected[name]
            if isinstance(value, ParticleGroup):
                assert actual == value
            else:
                np.testing.assert_equal(actual, value)
                if isinstance(value, np.ndarray):
                    assert actual.dtype == value.dtype
        assert model.get_value("track_type") == track_type
        assert model.get_value("qf:B1_GRADIENT") == 0.4

    @pytest.mark.parametrize("writer", [archive_model, LUMEBmadModel.archive])
    @pytest.mark.parametrize("track_type", ["single", "beam"])
    def test_archive_without_variable_values(
        self, model, tmp_path, monkeypatch, writer, track_type
    ):
        model.set({"track_type": track_type, "qf:B1_GRADIENT": 0.3})
        initial = model.initial_particles
        final = model.final_particles
        archive_path = model.archive(tmp_path / "metadata_only.h5")
        cached = model._state.copy()

        def unexpected_read(*args, **kwargs):
            pytest.fail("Variable values must not be read or validated")

        for variable_type in {type(var) for var in model.supported_variables.values()}:
            monkeypatch.setattr(variable_type, "_get", unexpected_read)
            monkeypatch.setattr(variable_type, "validate_value", unexpected_read)

        assert (
            writer(model, archive_path, include_variable_values=False) == archive_path
        )
        assert model._state.keys() == cached.keys()
        assert all(model._state[name] is value for name, value in cached.items())
        with h5py.File(archive_path, "r") as archive:
            assert list(archive["variables"]) == list(model.supported_variables)
            for node in archive["variables"].values():
                assert isinstance(node, h5py.Group)
                assert len(node) == 0
                assert node.attrs["kind"] == "omitted"
                assert "error" not in node.attrs
                variable = model.supported_variables[node.attrs["name"]]
                assert node.attrs["variable_type"] == (
                    f"{type(variable).__module__}.{type(variable).__qualname__}"
                )
                for name, value in variable.model_dump(mode="json").items():
                    if value is None:
                        assert isinstance(node.attrs[name], h5py.Empty)
                    else:
                        np.testing.assert_equal(node.attrs[name], value)
            for name in ("run_tao.cmd", "run_tao.sh", "run_tao.lat.bmad"):
                assert archive[name].asstr()[()]
            if track_type == "beam":
                assert ParticleGroup(h5=archive["initial_particles"]) == initial
                assert ParticleGroup(h5=archive["final_particles"]) == final
            else:
                assert "initial_particles" not in archive
                assert "final_particles" not in archive
        frame = read_archive_variables(archive_path)
        assert frame["value"].isna().all()
        assert frame["name"].tolist() == list(model.supported_variables)
        assert frame.attrs["errors"] == {}
        assert frame.attrs["omitted_values"] == list(model.supported_variables)
        if track_type == "beam":
            restored = tao_from_archive(archive_path)
            assert restored.tao_global()["track_type"] == "beam"
            assert restored.ele_gen_attribs("qf")["B1_GRADIENT"] == 0.3
            assert restored.get_initial_particles().n_particle == initial.n_particle

    def test_archive_preserves_existing_file_on_failure(
        self, model, tmp_path, monkeypatch
    ):
        archive_path = tmp_path / "model.h5"
        archive_path.write_bytes(b"existing archive")

        def fail_write(group, values):
            raise OSError("Unable to write variable snapshot")

        monkeypatch.setattr("lume_bmad.utils._write_variables", fail_write)
        with pytest.raises(OSError, match="Unable to write variable snapshot"):
            model.archive(archive_path)
        assert archive_path.read_bytes() == b"existing archive"
        assert list(tmp_path.iterdir()) == [archive_path]

    @pytest.mark.parametrize(
        "failed_name",
        ["qf:B1_GRADIENT", "mat6", "ix_ele", "name", "track_type", "qf_beam"],
    )
    def test_archive_variable_read_failure(
        self, model, tmp_path, monkeypatch, caplog, failed_name
    ):
        model.set({"track_type": "beam", "qf:B1_GRADIENT": 0.3, "qd:B1_GRADIENT": -0.2})
        variable = model.supported_variables[failed_name]
        cached = model._state[failed_name]
        original_get = type(variable)._get

        def fail_get(self, simulator):
            if self.name == failed_name:
                raise RuntimeError("test variable read failure")
            return original_get(self, simulator)

        monkeypatch.setattr(type(variable), "_get", fail_get)
        archive_path = model.archive(tmp_path / "partial.h5")
        with h5py.File(archive_path, "r") as archive:
            node = archive["variables"][failed_name]
            assert np.isnan(node[()]).all()
            assert node.shape == getattr(variable, "shape", ())
            assert node.attrs["error"] == "RuntimeError: test variable read failure"
            assert node.attrs["variable_type"] == (
                f"{type(variable).__module__}.{type(variable).__qualname__}"
            )
            assert "error" not in archive["variables/qd:B1_GRADIENT"].attrs
            assert archive["variables/qd:B1_GRADIENT"][()] == -0.2
        assert model._state[failed_name] is cached
        assert f"Unable to read archive variable {failed_name}" in caplog.text
        frame = read_archive_variables(archive_path)
        assert frame.attrs["errors"] == {
            failed_name: "RuntimeError: test variable read failure"
        }
        assert np.isnan(frame.set_index("name").at[failed_name, "value"]).all()
        with pytest.warns(UserWarning, match="test variable read failure"):
            restored = tao_from_archive(archive_path)
        assert restored.tao_global()["track_type"] == "beam"
        assert restored.ele_gen_attribs("qf")["B1_GRADIENT"] == 0.3
        assert restored.get_initial_particles().n_particle == 1000

    def test_archive_variable_validation_failure(self, model, tmp_path, monkeypatch):
        original_get = EleScalarVariable._get

        def invalid_get(self, simulator):
            if self.name == "qf:B1_GRADIENT":
                return "not a number"
            return original_get(self, simulator)

        monkeypatch.setattr(EleScalarVariable, "_get", invalid_get)
        archive_path = model.archive(tmp_path / "invalid_value.h5")
        with h5py.File(archive_path, "r") as archive:
            node = archive["variables/qf:B1_GRADIENT"]
            assert np.isnan(node[()])
            assert node.attrs["error"].startswith("TypeError:")

    @pytest.mark.parametrize(
        "failed_names",
        [
            ("initial_particles",),
            ("final_particles",),
            ("initial_particles", "final_particles"),
        ],
    )
    def test_archive_particle_read_failure(
        self, model, tmp_path, monkeypatch, caplog, failed_names
    ):
        model.set({"track_type": "beam", "qf:B1_GRADIENT": 0.2})

        def fail_particles(self):
            raise RuntimeError("test particle read failure")

        for name in failed_names:
            monkeypatch.setattr(LUMEBmadModel, name, property(fail_particles))
        archive_path = model.archive(tmp_path / "partial_particles.h5")
        with h5py.File(archive_path, "r") as archive:
            for name in failed_names:
                assert np.isnan(archive[name][()])
                assert archive[name].attrs["error"] == (
                    "RuntimeError: test particle read failure"
                )
                assert f"Unable to read archive distribution {name}" in caplog.text
        with pytest.warns(UserWarning, match="test particle read failure"):
            restored = tao_from_archive(archive_path)
        if "initial_particles" in failed_names:
            assert restored.tao_global()["track_type"] == "single"
            assert restored.beam_init(0)["position_file"] == ""
        else:
            assert restored.tao_global()["track_type"] == "beam"
            assert restored.get_initial_particles().n_particle == 1000
        assert restored.ele_gen_attribs("qf")["B1_GRADIENT"] == 0.2

    def test_archive_preserves_existing_file_on_metadata_failure(
        self, model, tmp_path, monkeypatch
    ):
        archive_path = tmp_path / "model.h5"
        archive_path.write_bytes(b"existing archive")

        def fail_serialization(self, **kwargs):
            raise TypeError("Unable to serialize variable metadata")

        monkeypatch.setattr(EleScalarVariable, "model_dump", fail_serialization)
        with pytest.raises(TypeError, match="Unable to serialize variable metadata"):
            model.archive(archive_path)
        assert archive_path.read_bytes() == b"existing archive"
        assert list(tmp_path.iterdir()) == [archive_path]

    @pytest.mark.parametrize("missing", ["initial_particles", "final_particles"])
    def test_archive_with_one_distribution(self, model, tmp_path, monkeypatch, missing):
        model.set({"track_type": "beam"})
        monkeypatch.setattr(LUMEBmadModel, missing, property(lambda self: None))
        archive_path = model.archive(tmp_path / "model.h5")
        available = (
            "final_particles" if missing == "initial_particles" else "initial_particles"
        )
        with h5py.File(archive_path, "r") as archive:
            assert missing not in archive
            assert ParticleGroup(h5=archive[available]) == getattr(model, available)

    def test_archive_replaces_existing_snapshot(self, model, tmp_path):
        model.set({"track_type": "beam"})
        archive_path = model.archive(tmp_path / "model.h5")
        model.set({"track_type": "single"})
        assert model.archive(str(archive_path)) == archive_path
        with h5py.File(archive_path, "r") as archive:
            assert "initial_particles" not in archive
            assert "final_particles" not in archive
        frame = read_archive_variables(archive_path).set_index("name")
        assert frame.at["track_type", "value"] == "single"
        assert "qf_beam" not in frame.index
        assert list(tmp_path.iterdir()) == [archive_path]
