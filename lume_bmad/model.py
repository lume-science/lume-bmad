from os import getcwd
from typing import Any
from lume.model import LUMEModel
from lume.variables import ScalarVariable, Variable, ParticleGroupVariable
from pytao import Tao
from lume_bmad.utils import (
    evaluate_tao,
    get_tao_lat_list_outputs,
    get_beam_info,
    get_particle_group_at_element,
)
from lume_bmad.transformer import BmadTransformer
from beamphysics.interfaces.bmad import write_bmad

class LUMEBmadModel(LUMEModel):
    """
    Subclasses LUMEModel to create a Bmad model given a tao init file.

    Attributes
    ----------
    device_to_element: dict[str, str]
        Mapping between device PV names and Bmad element names.
    element_to_device: dict[str, str]
        Mapping between Bmad element names and device PV names.
    control_variables: dict[str, Variable]
        Dictionary of control variables that can be read/written.
    read_only_variables: dict[str, Variable]
        Dictionary of read-only output variables.
    supported_variables: dict[str, Variable]
        Dictionary of all supported variables (control + read-only).

    """

    def __init__(
        self,
        init_file: str,
        control_variables: dict[str, Variable],
        output_variables: dict[str, Variable],
        transformer: BmadTransformer,
    ):
        """
        Initialize the Bmad model.

        Arguments
        ---------
        init_file: str
            Path to the Tao init file.
        control_variables: dict[str, Variable]
            Dictionary of control variables.
        output_variables: dict[str, Variable]
            Dictionary of output variables.
        transformer: BmadTransformer
            Transformer object for mapping between control variable names and Bmad element names + attributes.

        """

        self.tao = Tao(f"-init {init_file} -noplot")

        # import control and output variables
        self._control_variables = control_variables
        self._read_only_variables = output_variables

        # add both control and read-only variables to the list of model variables
        self._variables = {
            **self._control_variables, 
            **self._read_only_variables,
            "input_beam": ParticleGroupVariable(name="input_beam"),
            "output_beam": ParticleGroupVariable(name="output_beam", read_only=True),
        }

        # add track_type variable to control variables to allow toggling between single particle and beam tracking
        self._variables.update(
            {"track_type": ScalarVariable(name="track_type")}
        )

        # create transformer for mapping between control names and bmad names
        self.transformer = transformer

        # set dumping of beam distributions at the beginning and end of the lattice
        self.tao.cmd(f"set beam saved_at = {self.start_element}, {self.end_element}")

        # get initial state of the model
        self._state = {}
        self.update_state()

        self._initial_state = self._state.copy()

    def _get(self, names: list[str]) -> dict[str, Any]:
        """
        Internal method to retrieve current values for specified variables.

        Parameters
        ----------
        names : list[str]
            List of variable names to retrieve

        Returns
        -------
        dict[str, Any]
            Dictionary mapping variable names to their current values
        """
        return {name: self._state[name] for name in names}

    def _set(self, values: dict[str, Any]) -> None:
        """
        Internal method to set input variables and compute outputs.

        This method:
        1. Updates input variables in the state
        2. Performs calculations to update output variables
        3. Stores results in the state

        Parameters
        ----------
        values : dict[str, Any]
            Dictionary of variable names and values to set
        """
        # handle setting track_type separately since it is not a simple Tao property
        if "track_type" in values.keys():
            if values.pop("track_type", None) == 1:
                self.tao.cmd("set global track_type = beam")
            else:
                self.tao.cmd("set global track_type = single")

        # handle setting the input beam separately
        if "input_beam" in values.keys():
            input_beam = values.pop("input_beam")
            fname = getcwd()+"/input_beam.particles"
            write_bmad(input_beam, fname, p0c=input_beam["mean_p"])
            self.tao.cmd(f"set beam_init position_file = {fname}")

        # map pvdata to tao commands and evaluate
        tao_cmds = self.transformer.get_tao_commands(self.tao, values)
        evaluate_tao(self.tao, tao_cmds)

        # update state with new input / output values
        self.update_state()

    def update_state(self) -> None:
        """
        Update the model state by reading all supported variables.
        """
        # iterate through all supported variables to get their current values and update the state
        for name in self.supported_variables.keys():
            # handle reading the input / output beam distributions
            if name in ["input_beam", "output_beam"]:
                if self.tao.tao_global()["track_type"] == "beam":
                    # get element at track start
                    element_name = self.start_element if name == "input_beam" else self.end_element
                    self._state[name] = get_particle_group_at_element(
                            self.tao, element_name
                        )
                else:
                    self._state[name] = None
            
            elif name == "track_type":
                self._state[name] = 1 if self.tao.tao_global()["track_type"] == "beam" else 0
            else:
                # for other variables, use the transformer to get the value from Tao
                self._state[name] = self.transformer.get_tao_property(self.tao, name)

    @property
    def control_name_to_bmad(self):
        """mapping between control variable PV names and Bmad element names + attributes"""
        return self.transformer.control_name_to_bmad

    @property
    def control_variables(self):
        """dictionary of control variables"""
        return self._control_variables

    @property
    def read_only_variables(self):
        """dictionary of read-only output variables"""
        return self._read_only_variables
    
    @property
    def start_element(self):
        """name of element at which beam is initialized for tracking"""
        return self.tao.beam(0)["track_start"] if self.tao.beam(0)["track_start"] != "" else "BEGINNING"
    
    @property
    def end_element(self):
        """name of element at which beam is tracked to"""
        return self.tao.beam(0)["track_end"] if self.tao.beam(0)["track_end"] != "" else "END"

    @property
    def supported_variables(self):
        """dictionary of all supported variables"""
        return self._variables

    def reset(self):
        """Reset the model to its initial state."""
        self.set(self._initial_state)
