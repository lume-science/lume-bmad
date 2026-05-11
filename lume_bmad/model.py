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
)
from lume_bmad.transformer import BmadTransformer


logger = logging.getLogger(__name__)


class LUMEBmadModel(LUMEModel, InitialParticlesMixIn, FinalParticlesMixIn):
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
        tao: Tao,
        control_variables: dict[str, Variable],
        output_variables: dict[str, Variable],
        transformer: BmadTransformer,
        dump_locations: list[str] = [],
        comb_ds_save: float = 0.1,
    ):
        """
        Initialize the Bmad model.

        Arguments
        ---------
        tao: Tao
            Tao object initialized with the desired init file.
        control_variables: dict[str, Variable]
            Dictionary of control variables.
        output_variables: dict[str, Variable]
            Dictionary of output variables.
        transformer: BmadTransformer
            Transformer object for mapping between control variable names and Bmad element names + attributes.
        dump_locations: list[str], optional
            List of element names at which to dump the beam distribution in addition to the start and end of the lattice.
        comb_ds_save: float, optional
            Length frequency of dumping tracked beam parameters for tao comb command. Default is 0.1 m.

        """

        self.tao = tao
        self.comb_ds_save = comb_ds_save
        logger.debug("Initializing LUMEBmadModel with comb_ds_save=%s", comb_ds_save)

        # Add model parameters read_only_variables
        model_output_variables = get_tao_output_variables(self.tao)

        # import control and output variables
        self._control_variables = control_variables
        self._read_only_variables = model_output_variables | output_variables

        # add both control and read-only variables to the list of model variables
        self._variables = {
            **self._control_variables,
            **self._read_only_variables,
        }

        # add track_type variable to control variables to allow toggling between single particle and beam tracking
        self._variables.update({"track_type": ScalarVariable(name="track_type")})

        # create transformer for mapping between control names and bmad names
        self.transformer = transformer

        # set dumping of beam distributions at the beginning and end of the lattice
        self._dump_locations = dump_locations
        elements = ",".join(dump_locations + ["BEGINNING", "END"])
        self.tao.cmd(f"set beam saved_at = {elements}")

        # get initial state of the model
        self._state = {}
        self.update_state()

        self._initial_state = self._state.copy()
        logger.info(
            "Initialized LUMEBmadModel with %d control vars, %d read-only vars, %d total vars",
            len(self._control_variables),
            len(self._read_only_variables),
            len(self._variables),
        )

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

        # handle setting track_type separately since it is not a simple Tao property
        if "track_type" in values.keys():
            if values["track_type"] == 1:
                logger.debug("Switching Tao track_type to beam")
                output = self.tao.cmd("set global track_type = beam")

                # set comb length for tracking outputs
                self.tao.cmd(f"set beam comb_ds_save = {self.comb_ds_save}")
                tao_model_output_variables = get_tao_output_variables(self.tao)
                self._variables.update(tao_model_output_variables)

                # set particle group variables
                self._variables.update(
                    {
                        f"{ele}_beam": ParticleGroupVariable(
                            name=f"{ele}_beam", read_only=True
                        )
                        for ele in self._dump_locations
                    }
                )

            else:
                logger.debug("Switching Tao track_type to single")
                output = self.tao.cmd("set global track_type = single")

                # remove comb output variables when switching back to single particle tracking
                for var in TAO_COMB_OUTPUT_UNITS.keys():
                    if var in self._variables.keys():
                        self._variables.pop(var)

                # remove particle group variables
                for var in [f"{ele}_beam" for ele in self._dump_locations]:
                    if var in self._variables.keys():
                        self._variables.pop(var)

            if len(output) > 0:
                logger.warning("Warning while setting track_type: %s", "".join(output))
                warnings.warn(f"Warning while setting track_type: {''.join(output)}")
            values.pop("track_type")

        # handle setting the input beam separately
        if "input_beam" in values.keys():
            input_beam = values.pop("input_beam")
            fname = getcwd() + "/input_beam.h5"
            logger.debug("Writing input beam to %s", fname)
            input_beam.write(fname)
            self.tao.cmd(f"set beam_init position_file = {fname}")

            # reset comb length for tracking outputs
            self.tao.cmd(f"set beam comb_ds_save = {self.comb_ds_save}")
            tao_model_output_variables = get_tao_output_variables(self.tao)
            self._variables.update(tao_model_output_variables)

        # map pvdata to tao commands and evaluate
        tao_cmds = self.transformer.get_tao_commands(self.tao, values)
        logger.debug("Evaluating %d Tao commands", len(tao_cmds))
        evaluate_tao(self.tao, tao_cmds)

        # update state with new input / output values
        self.update_state()

    def update_state(self) -> None:
        """
        Update the model state by reading all supported variables.
        """
        logger.debug("Updating model state for %d supported variables", len(self.supported_variables))

        # iterate through all supported variables to get their current values and update the state
        for name in self.supported_variables.keys():
            # handle reading the input / output beam distributions
            if name in [f"{ele}_beam" for ele in self._dump_locations]:
                if self.tao.tao_global()["track_type"] == "beam":
                    element_name = name.split("_beam")[0]
                    self._state[name] = self.tao.particles(element_name)
                else:
                    self._state[name] = None

            # handle reading the track_type variable
            elif name == "track_type":
                self._state[name] = (
                    1 if self.tao.tao_global()["track_type"] == "beam" else 0
                )

            # handle reading all of the per-element tao tracking outputs
            elif name in TAO_OUTPUT_UNITS.keys():
                lat_values = self.tao.lat_list("*", "ele." + name)
                if name == "name":
                    # Keep element names as object dtype to avoid fixed-width unicode dtypes.
                    self._state[name] = np.asarray(lat_values, dtype=object)
                elif name == "mat6":
                    # reshape mat6 output to be (element_count, 6, 6)
                    self._state[name] = np.asarray(lat_values).reshape(-1, 6, 6)
                elif name == "vec0":
                    # reshape vec0 output to be (element_count, 6)
                    self._state[name] = np.asarray(lat_values).reshape(-1, 6)
                else:
                    self._state[name] = np.asarray(lat_values)

            # handle reading the tao comb outputs for beam tracking
            elif name in TAO_COMB_OUTPUT_UNITS.keys():
                # for comb outputs, get the value from the tao bunch_comb command
                if self.tao.tao_global()["track_type"] == "beam":
                    self._state[name] = np.asarray(self.tao.bunch_comb(name))
                else:
                    self._state[name] = None

            else:
                # for other variables, use the transformer to get the value from Tao
                self._state[name] = self.transformer.get_tao_property(self.tao, name)

        logger.debug("Model state update complete")

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
        return (
            self.tao.beam(0)["track_start"]
            if self.tao.beam(0)["track_start"] != ""
            else self.tao.lat_list("*", "ele.name")[0]
        )

    @property
    def end_element(self):
        """name of element at which beam is tracked to"""
        return (
            self.tao.beam(0)["track_end"]
            if self.tao.beam(0)["track_end"] != ""
            else self.tao.lat_list("*", "ele.name")[-1]
        )

    @property
    def initial_particles(self) -> ParticleGroup:
        """initial particle distribution for tracking"""
        if self.tao.tao_global()["track_type"] == "beam":
            return self.tao.particles(self.start_element)
        else:
            return None

    @initial_particles.setter
    def initial_particles(self, particles: ParticleGroup):
        """set the initial particle distribution for tracking"""
        if self.tao.tao_global()["track_type"] == "beam":
            fname = getcwd() + "/input_beam.h5"
            logger.debug("Setting initial particles from %s", fname)
            particles.write(fname)
            self.tao.cmd(f"set beam_init position_file = {fname}")

            # after setting the initial particles, we need to update the comb variables
            # and the model state to reflect the new particle distribution and any changes to output 
            # variables that depend on the input beam
            self.tao.cmd(f"set beam comb_ds_save = {self.comb_ds_save}")
            tao_model_output_variables = get_tao_output_variables(self.tao)
            self._variables.update(tao_model_output_variables)

            # update the model state
            self.update_state()
        else:
            raise ValueError(
                "Cannot set initial_particles when track_type is not 'beam'"
            )

    @property
    def final_particles(self) -> ParticleGroup:
        """final particle distribution after tracking"""
        if self.tao.tao_global()["track_type"] == "beam":
            return self.tao.particles(self.end_element)
        else:
            return None

    @property
    def supported_variables(self):
        """dictionary of all supported variables"""
        return self._variables

    def reset(self):
        """Reset the model to its initial state."""
        logger.info("Resetting model to initial state")
        self.set(self._initial_state)
