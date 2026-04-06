import pytest
import numpy as np

from pytao import Tao

from lume_bmad.model import LUMEBmadModel
from lume_bmad.utils import TAO_COMB_OUTPUT_UNITS
from lume.variables import NDVariable, ScalarVariable
from lume_bmad.transformer import BasicTransformer
from beamphysics import ParticleGroup

class TestModel:
    @pytest.fixture
    def model(self):
        control_variables = {
            "qf:B1_GRADIENT": ScalarVariable(name="qf:B1_GRADIENT", units="1/m^2"),
            "qd:B1_GRADIENT": ScalarVariable(name="qd:B1_GRADIENT", units="1/m^2"),
        }
        transformer = BasicTransformer({})
        tao = Tao(init_file="tests/fodo.init", noplot=True)

        model = LUMEBmadModel(tao, control_variables, {}, transformer, dump_locations=["qf", "qd"])
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

        output_beam = model.get(["output_beam"])["output_beam"]
        assert isinstance(output_beam, ParticleGroup)
        assert output_beam.n_particle == 1000

        # set the input beam element name and set the input beam
        model.input_beam_element_name = "qf"
        model.set({"input_beam": output_beam})

        # read the input beam back and check that it is the same as the output beam
        input_beam = model.get(["input_beam"])["input_beam"]
        assert input_beam == output_beam

        # test that beam is being dumped at the correct locations
        for ele in ["qf", "qd"]:
            beam = model.get([f"{ele}_beam"])[f"{ele}_beam"]
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
                    hist, _ = beam.histogramdd("x", "y", bins=(100, 100), range=((-0.1, 0.1), (-0.1, 0.1)))
                    return hist
                else:
                    return super().get_tao_property(tao, control_name)


        model = LUMEBmadModel(
            Tao(init_file="tests/fodo.init", noplot=True), 
            control_variables,
            output_variables, 
            ScreenTransformer({}), 
            dump_locations=["qf", "qd"]
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
        tracked = model.get(["track_type", "input_beam", "output_beam", "qf_beam", "qd_beam"])
        assert tracked["track_type"] == 1
        assert isinstance(tracked["input_beam"], ParticleGroup)
        assert isinstance(tracked["output_beam"], ParticleGroup)
        assert isinstance(tracked["qf_beam"], ParticleGroup)
        assert isinstance(tracked["qd_beam"], ParticleGroup)

        # switching back to single-particle mode should clear beam dumps
        model.set({"track_type": 0})
        assert model.get(["track_type"])["track_type"] == 0

        # In single mode update_state sets beam objects to None. Public get() currently
        # validates ParticleGroupVariable types, so use _get to inspect stored state.
        single_state = model._get(["input_beam", "output_beam", "qf_beam", "qd_beam"])
        assert single_state["input_beam"] is None
        assert single_state["output_beam"] is None
        assert single_state["qf_beam"] is None
        assert single_state["qd_beam"] is None

        with pytest.raises(TypeError):
            model.get(["input_beam"])

    def test_supported_variables_contains_model_interfaces(self, model):
        supported = model.supported_variables
        expected = {
            "qf:B1_GRADIENT",
            "qd:B1_GRADIENT",
            "input_beam",
            "output_beam",
            "qf_beam",
            "qd_beam",
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
            "input_beam",
            "output_beam",
            "qf_beam",
            "qd_beam",
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

        # setting track_type back to single should remove comb output variables
        model.set({"track_type": 0})
        supported = model.supported_variables
        expected.difference_update(set(TAO_COMB_OUTPUT_UNITS.keys()))
        assert expected.issubset(set(supported.keys()))
