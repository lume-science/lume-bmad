from typing import Any
import numpy as np

from lume.actions import Action, ReadOnlyActionMixin, WritableActionMixin
from lume.variables import ScalarVariable, NDVariable, EnumVariable, ParticleGroupVariable
from pytao import Tao

import logging

logger = logging.getLogger(__name__)

class EleScalarVariable(ScalarVariable, WritableActionMixin):
    """Action that operates on a single scalar variable in the Bmad model."""

    def _get(self, simulator: Tao) -> Any:
        return simulator.ele_gen_attribs(self.name.split(":")[0])[self.name.split(":")[1]]

    def _set(self, simulator: Tao, value: Any) -> None:
        element_name, property_name = self.name.split(":")
        logger.debug(f"Setting {self.name} to {value}")
        simulator.cmd(f"set ele {element_name} {property_name} = {value}")
    

class StatVariable(NDVariable, ReadOnlyActionMixin):
    """Action that operates on a single statistic in the Bmad model."""

    statistic_name: str

    def _get(self, simulator: Tao) -> Any:
        if self.statistic_name == "s_ele":
            name = "s"
        else:
            name = self.statistic_name
        value = simulator.lat_list("*", f"ele.{name}")
        arr = np.asarray(value, dtype=self.dtype)

        # Tao returns some outputs as flat arrays; reshape to the declared variable shape.
        if tuple(arr.shape) != self.shape:
            arr = arr.reshape(self.shape)

        return arr


class CombStatVariable(NDVariable, ReadOnlyActionMixin):
    """Action that operates on a single comb statistic in the Bmad model."""

    statistic_name: str

    def _get(self, simulator: Tao) -> Any:
        if simulator.tao_global()["track_type"] == "beam":
            return simulator.bunch_comb(self.statistic_name)
        else:
            raise ValueError(
                "CombStatAction can only be used when track_type is 'beam'."
            )
    

class TrackTypeAction(EnumVariable, WritableActionMixin):
    """Action that gets or sets the track type in the Bmad model."""
    name: str = "track_type"
    options: list[str] = ["single", "beam"]

    def _get(self, simulator: Tao) -> Any:
        return simulator.tao_global()["track_type"]

    def _set(self, simulator: Tao, value: Any) -> None:
        logger.debug(f"Setting track_type to {value}")
        if value not in self.options:
            raise ValueError(f"track_type must be one of {self.options}.")
        simulator.cmd(f"set global track_type = {value}")


class BeamAtElementVariable(ParticleGroupVariable, ReadOnlyActionMixin):
    """Read-only action that returns a dumped beam at a given element."""

    element_name: str

    def _get(self, simulator: Tao) -> Any:
        if simulator.tao_global()["track_type"] != "beam":
            raise ValueError("Beam outputs are only available when track_type is 'beam'.")
        return simulator.particles(self.element_name)


class ScreenVariable(NDVariable, ReadOnlyActionMixin):
    """Read-only action that returns a screen image at a given element."""

    element_name: str
    pixel_size: float  # default pixel size in meters

    def _get(self, simulator: Tao) -> Any:
        if simulator.tao_global()["track_type"] != "beam":
            return np.zeros(self.shape)

        beam = simulator.particles(self.element_name)

        # simple screen that counts number of particles
        half_width = np.array(self.shape) * self.pixel_size / 2
        hist, _ = beam.histogramdd(
            "x", "y", bins=self.shape, range=((-half_width[0], half_width[0]), (-half_width[1], half_width[1]))
        )
        return hist