import pytest
import numpy as np
import os
from pathlib import Path
from pytao import Tao

from lume_bmad.model import LUMEBmadModel
from lume_bmad.utils import TAO_COMB_OUTPUT_UNITS
from lume.variables import NDVariable, ScalarVariable
from lume_bmad.transformer import BasicTransformer
from beamphysics import ParticleGroup

TEST_BEAM_PATH = os.path.join(Path(__file__).parent, "test_beam.h5")


class TestModel:
    @pytest.fixture
    def model(self):
        control_variables = {
            "qf:B1_GRADIENT": ScalarVariable(name="qf:B1_GRADIENT", units="1/m^2"),
            "qd:B1_GRADIENT": ScalarVariable(name="qd:B1_GRADIENT", units="1/m^2"),
        }
        transformer = BasicTransformer({})
        tao = Tao(init_file="tests/fodo.init", noplot=True)

        model = LUMEBmadModel(
            tao, control_variables, {}, transformer, dump_locations=["qf", "qd"]
        )
        return model

    def test_model_initialization(self, model):

        # test that beam dump locations are set correctly
        assert model.start_element == "BEGINNING"
        assert model.end_element == "END"

        assert model.tao.beam(0)["saved_at"] == "qf,qd,BEGINNING,END"

        model.set({"qf:B1_GRADIENT": 0.2})
        assert model.get(["qf:B1_GRADIENT"])["qf:B1_GRADIENT"] == 0.2

    def test_beam_tracking(self, model):
        # set track_type to 1 to enable tracking
        model.set({"track_type": 1})

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
            beam = model.get(f"{ele}_beam")
            assert isinstance(beam, ParticleGroup)
            assert beam.n_particle == 1000

    def test_screen(self, model):
        # set track_type to 1 to enable tracking
        control_variables = {
            "qf:B1_GRADIENT": ScalarVariable(name="qf:B1_GRADIENT", units="1/m^2"),
            "qd:B1_GRADIENT": ScalarVariable(name="qd:B1_GRADIENT", units="1/m^2"),
        }
        output_variables = {
            "qf_screen": NDVariable(name="qf_screen", read_only=True, shape=(100, 100)),
        }

        class ScreenTransformer(BasicTransformer):
            def get_tao_property(self, tao, control_name):
                if control_name == "qf_screen":
                    if tao.tao_global()["track_type"] != "beam":
                        return np.zeros((100, 100))
                    beam = tao.particles("qf")
                    # simple screen that counts number of particles
                    hist, _ = beam.histogramdd(
                        "x", "y", bins=(100, 100), range=((-0.1, 0.1), (-0.1, 0.1))
                    )
                    return hist
                else:
                    return super().get_tao_property(tao, control_name)

        model = LUMEBmadModel(
            Tao(init_file="tests/fodo.init", noplot=True),
            control_variables,
            output_variables,
            ScreenTransformer({}),
            dump_locations=["qf", "qd"],
        )
        model.set({"track_type": 1})
        qf_screen = model.get(["qf_screen"])["qf_screen"]
        assert isinstance(qf_screen, np.ndarray)
        assert qf_screen.shape == (100, 100)

    def test_mat6_output(self, model):
        # test that mat6 output variable is being read and has correct shape
        mat6 = model.get("mat6")
        assert isinstance(mat6, np.ndarray)
        assert mat6.shape == (len(model.tao.lat_list("*", "ele.name")), 6, 6)

    def test_vec0_output(self, model):
        # test that vec0 output variable is being read and has correct shape
        vec0 = model.get("vec0")
        assert isinstance(vec0, np.ndarray)
        assert vec0.shape == (len(model.tao.lat_list("*", "ele.name")), 6)

    def test_track_type_toggle_updates_beam_state(self, model):
        # start in beam tracking mode and verify beam outputs are populated
        model.set({"track_type": 1})
        tracked = model.get(["track_type", "qf_beam", "qd_beam"])
        assert tracked["track_type"] == 1
        assert isinstance(tracked["qf_beam"], ParticleGroup)
        assert isinstance(tracked["qd_beam"], ParticleGroup)

        # switching back to single-particle mode should clear beam dumps
        model.set({"track_type": 0})
        assert model.get("track_type") == 0

        # In single mode update_state removes the beam dumps from the list of supported variables
        for var in ["qf_beam", "qd_beam"]:
            with pytest.raises(ValueError):
                model.get(var)

        assert model.initial_particles is None
        assert model.final_particles is None

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
        names = model.get("name")
        assert isinstance(names, np.ndarray)
        assert names.dtype == object
        assert len(names) == len(model.tao.lat_list("*", "ele.name"))

    def test_comb_ds_save_setting(self, model):
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
        model.set({"track_type": 1})
        supported = model.supported_variables
        expected.update(set(TAO_COMB_OUTPUT_UNITS.keys()))
        assert expected.issubset(set(supported.keys()))

        # try to get a comb output variable before setting comb_ds_save and check that it is empty
        comb_output = model.get("x.beta")
        assert isinstance(comb_output, np.ndarray)
        assert len(comb_output) == 23

        # setting track_type back to single should remove comb output variables
        model.set({"track_type": 0})
        supported = model.supported_variables
        expected.difference_update(set(TAO_COMB_OUTPUT_UNITS.keys()))
        assert expected.issubset(set(supported.keys()))

        # try to get a comb output -- should raise error since they should no longer be supported
        with pytest.raises(ValueError):
            model.get("x.beta")

        # adding an initial beam should update the length of the comb output variables
        model.set({"track_type": 1})
        model.initial_particles = ParticleGroup(TEST_BEAM_PATH)

        comb_output = model.get("x.beta")
        assert isinstance(comb_output, np.ndarray)
        assert len(comb_output) == 23

    def test_getting_all_variables(self, model):
        variable_names = list(model.supported_variables.keys())

        for name in variable_names:
            model.get(name)

        # run the model in beam tracking mode and then try to get all variables again, including beam variables
        model.set({"track_type": 1})
        for name in model.supported_variables.keys():
            model.get(name)

    def test_setting_initial_particles_updates_state(self, model):
        model.set({"track_type": 1})
        particles = ParticleGroup(TEST_BEAM_PATH)

        # add a dummy value that should be updated when the state is updated after setting the initial particles
        model._state["mat6"] = np.zeros((len(model.tao.lat_list("*", "ele.name")), 6, 6))

        # after setting the initial particles, the model state should be updated to reflect the new beam
        model.initial_particles = particles
        assert model.initial_particles == particles
        assert not np.all(model._state["mat6"] == 0)
