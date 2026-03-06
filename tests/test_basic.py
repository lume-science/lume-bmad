import pytest
from lume_bmad.model import LUMEBmadModel
from lume.variables import ScalarVariable
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

        model = LUMEBmadModel("tests/fodo.init", control_variables, {}, transformer)
        return model

    def test_model_initialization(self, model):

        # test that beam dump locations are set correctly
        assert model.start_element == "BEGINNING"
        assert model.end_element == "END"

        assert model.tao.beam(0)["saved_at"] == "BEGINNING, END"

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


