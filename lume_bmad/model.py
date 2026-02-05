from typing import Any
from lume.model import LUMEModel
# from pytao import Tao

from lume_bmad.utils import (
    SLAC2BmadTransformer,
    import_control_variables,
    import_output_variables,
    evaluate_tao,
    get_tao_lat_list_outputs,
)


class LUMEBmadModel(LUMEModel):
    """
    Subclasses LUMEModel to create a Bmad model given a tao init file.

    Attributes
    ----------
    device_to_element: dict[str, str]
        Mapping between device PV names and Bmad element names.
    element_to_device: dict[str, str]
        Mapping between Bmad element names and device PV names.
    control_variables: dict[str, ScalarVariable]
        Dictionary of control variables that can be read/written.
    read_only_variables: dict[str, ScalarVariable]
        Dictionary of read-only output variables.
    supported_variables: dict[str, ScalarVariable]
        Dictionary of all supported variables (control + read-only).

    """

    def __init__(
        self,
        init_file: str,
        control_variable_file: str,
        output_variable_file: str,
    ):
        """
        Initialize the Bmad model.

        Arguments
        ---------
        init_file: str
            Path to the Tao init file.
        control_variable_file: str
            Path to the YAML file containing control variable definitions.
        output_variable_file: str
            Path to the YAML file containing output variable definitions.

        """

        # self.tao = Tao(f"-init {init_file} -noplot")

        # import control and output variables
        self._control_variables = {}
        self._read_only_variables = {}

        # Define control system variables
        self._control_variables, self._control_name_to_bmad = import_control_variables(
            control_variable_file
        )

        # Define read-only output variables
        self._read_only_variables = import_output_variables(output_variable_file)

        # add both control and read-only variables to the list of model variables
        self._variables = {**self._control_variables, **self._read_only_variables}

        # create transformer for mapping between control names and bmad names
        self.transformer = SLAC2BmadTransformer(self._control_name_to_bmad)

        # get initial state of the model
        self._state = {}
        # self.update_state()

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

        # map pvdata to tao commands and evaluate
        tao_cmds = self.transformer.get_tao_commands(self.tao, values, "cu_hxr")
        evaluate_tao(self.tao, tao_cmds)

        # update state with new input / output values
        self.update_state()

    def update_state(self) -> None:
        """
        Update the model state by reading all supported variables.

        """

        # handle reading all of the control variables
        control_names = list(self.control_variables.keys())  # get list of PV names
        for name in control_names:
            self._state[name] = self.transformer.get_tao_property(self.tao, name)

        # handle reading twiss functions and rmats at all elements for output variables
        self._state.update(get_tao_lat_list_outputs(self.tao))

        # handle reading other read-only output variables
        # TODO: implement other read-only variable types as needed

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
    def supported_variables(self):
        """dictionary of all supported variables"""
        return self._variables

    def reset(self):
        """Reset the model to its initial state."""
        self.set(self._initial_state)
