from os import getcwd
import logging
import warnings
from beamphysics import ParticleGroup
import numpy as np
from typing import Any
from lume.model import LUMEModel
from lume.staged_model import InitialParticlesMixIn, FinalParticlesMixIn
from lume.variables import ScalarVariable, Variable, ParticleGroupVariable
from pytao import Tao
from lume_bmad.utils import (
    evaluate_tao,
    get_tao_output_variables,
    TAO_OUTPUT_UNITS,
    TAO_COMB_OUTPUT_UNITS,
    get_tao_comb_output_variables,
)
from lume.actions import ActionModel, ActionVariable
from lume_bmad.actions import TrackTypeAction, CombStatVariable, BeamAtElementVariable

logger = logging.getLogger(__name__)


class LUMEBmadModel(ActionModel, InitialParticlesMixIn, FinalParticlesMixIn):
    """
    Subclasses ActionModel to create a Bmad model given a tao init file.

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
        tao: Tao,
        action_variables: list[ActionVariable],
        dump_locations: list[str] = [],
        comb_ds_save: float = 0.1,
    ):
        """
        Initialize the Bmad model.

        Arguments
        ---------
        tao: Tao
            Tao object initialized with the desired init file.
        action_variables: list[ActionVariable]
            List of action variables.
        dump_locations: list[str], optional
            List of element names at which to dump the beam distribution in addition to the start and end of the lattice.
        comb_ds_save: float, optional
            Length frequency of dumping tracked beam parameters for tao comb command. Default is 0.1 m.

        """
        super().__init__(simulator=tao, action_variables=action_variables)

        self.comb_ds_save = comb_ds_save
        logger.debug("Initializing LUMEBmadModel with comb_ds_save=%s", comb_ds_save)

        # Add model parameters read_only_variables
        model_output_variables = get_tao_output_variables(self.simulator)

        for var in model_output_variables:
            self.register_action_variable(var)


        # add track_type variable to control variables to allow toggling between single particle and beam tracking
        self.register_action_variable(TrackTypeAction())

        # set dumping of beam distributions at the beginning and end of the lattice
        self._dump_locations = dump_locations
        elements = ",".join(dump_locations + ["BEGINNING", "END"])
        self.simulator.cmd(f"set beam saved_at = {elements}")

        # Register dynamic outputs that depend on the current tracking mode.
        self._refresh_dynamic_action_variables()

        # get initial state of the model
        self._state = {}
        self.update_state()

        self._initial_state = self._state.copy()
        logger.info(
            "Initialized LUMEBmadModel ",
        )

    def _refresh_dynamic_action_variables(self) -> None:
        """Synchronize comb and dumped-beam action variables with the current track type."""
        in_beam_mode = self.simulator.tao_global()["track_type"] == "beam"

        comb_names = list(TAO_COMB_OUTPUT_UNITS.keys())
        if in_beam_mode:
            for variable in get_tao_comb_output_variables(self.simulator):
                self.register_action_variable(variable)
        else:
            for parameter_name in comb_names:
                if parameter_name in self.supported_variables:
                    self.unregister_action_variable(parameter_name)

        for element_name in self._dump_locations:
            beam_variable_name = f"{element_name}_beam"
            if in_beam_mode:
                self.register_action_variable(
                    BeamAtElementVariable(name=beam_variable_name, element_name=element_name)
                )
            elif beam_variable_name in self.supported_variables:
                self.unregister_action_variable(beam_variable_name)

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
        logger.debug("_set called with keys: %s", list(values.keys()))

        # turn tao eager mode off to speed up setting multiple variables
        self.simulator.cmd("set global lattice_calc_on = F")

        # set control variables using their respective set methods
        super()._set(values)

        # after setting all variables, turn eager mode back on
        self.simulator.cmd("set global lattice_calc_on = T")

        # In beam mode, ensure comb outputs are allocated before reading them.
        if self.simulator.tao_global()["track_type"] == "beam":
            self.simulator.cmd(f"set beam comb_ds_save = {self.comb_ds_save}")

        # track_type toggles the set of supported read-only outputs.
        self._refresh_dynamic_action_variables()

        # update state with new input / output values
        self.update_state()

    def update_state(self) -> None:
        """
        Update the model state by reading all supported variables.
        """
        logger.debug("Updating model state for %d supported variables", len(self.supported_variables))
        for name in self.supported_variables.keys():
            try:
                self._state[name] = self.supported_variables[name]._get(self.simulator)
            except Exception as e:
                logger.error("Error getting variable %s: %s", name, str(e))
                raise e
        
        logger.debug("Model state update complete")

    @property
    def start_element(self):
        """name of element at which beam is initialized for tracking"""
        return (
            self.simulator.beam(0)["track_start"]
            if self.simulator.beam(0)["track_start"] != ""
            else self.simulator.lat_list("*", "ele.name")[0]
        )

    @property
    def end_element(self):
        """name of element at which beam is tracked to"""
        return (
            self.simulator.beam(0)["track_end"]
            if self.simulator.beam(0)["track_end"] != ""
            else self.simulator.lat_list("*", "ele.name")[-1]
        )

    @property
    def initial_particles(self) -> ParticleGroup:
        """initial particle distribution for tracking"""
        if self.simulator.tao_global()["track_type"] == "beam":
            return self.simulator.particles(self.start_element)
        else:
            return None

    @property
    def tao(self) -> Tao:
        """access to the underlying Tao simulator object"""
        return self.simulator

    @initial_particles.setter
    def initial_particles(self, particles: ParticleGroup):
        """set the initial particle distribution for tracking"""
        if self.simulator.tao_global()["track_type"] == "beam":
            fname = getcwd() + "/input_beam.h5"
            logger.debug("Setting initial particles from %s", fname)
            particles.write(fname)
            self.simulator.cmd(f"set beam_init position_file = {fname}")

            # after setting the initial particles, we need to update the comb variables
            # and the model state to reflect the new particle distribution and any changes to output 
            # variables that depend on the input beam
            self.simulator.cmd(f"set beam comb_ds_save = {self.comb_ds_save}")
            self._refresh_dynamic_action_variables()

            # update the model state
            self.update_state()
        else:
            raise ValueError(
                "Cannot set initial_particles when track_type is not 'beam'"
            )

    @property
    def final_particles(self) -> ParticleGroup:
        """final particle distribution after tracking"""
        if self.simulator.tao_global()["track_type"] == "beam":
            return self.simulator.particles(self.end_element)
        else:
            return None

    def reset(self):
        """Reset the model to its initial state."""
        logger.info("Resetting model to initial state")
        self.set(self._initial_state)
