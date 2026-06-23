from dataclasses import dataclass
from typing import Any
import numpy as np

from lume.actions import ReadOnlyActionMixin, WritableActionMixin
from lume.variables import (
    ScalarVariable,
    IntVariable,
    NDVariable,
    EnumVariable,
    ParticleGroupVariable,
)
from pytao import Tao

import logging

logger = logging.getLogger(__name__)


class EleScalarVariable(ScalarVariable, WritableActionMixin):
    """Action that operates on a single scalar variable in the Bmad model."""
    element_name: str
    property_name: str

    def _get(self, simulator: Tao) -> Any:
        return simulator.ele_gen_attribs(self.element_name)[self.property_name]

    def _set(self, simulator: Tao, value: Any) -> None:
        logger.debug(f"Setting {self.name} to {value}")
        simulator.cmd(f"set ele {self.element_name} {self.property_name} = {value}")


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
                "CombStatVariable can only be used when track_type is 'beam'."
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
            raise ValueError(
                "Beam outputs are only available when track_type is 'beam'."
            )
        return simulator.particles(self.element_name)


@dataclass(frozen=True)
class ScreenSpec:
    """Definition of a screen detector geometry in the lattice."""

    element_name: str
    shape: tuple[int, int]
    pixel_size: float

    @property
    def half_width(self) -> tuple[float, float]:
        """Half-width used to build the histogram range."""
        return (
            (self.shape[0] * self.pixel_size) / 2,
            (self.shape[1] * self.pixel_size) / 2,
        )


class _ScreenSpecVariableMixin:
    """Shared ScreenSpec conversion utilities for screen-derived variables."""

    element_name: str
    pixel_size: float
    shape: tuple[int, int]

    @property
    def screen_spec(self) -> ScreenSpec:
        """Canonical screen specification for this variable."""
        return ScreenSpec(
            element_name=self.element_name,
            shape=self.shape,
            pixel_size=self.pixel_size,
        )

    @classmethod
    def _from_screen_spec(
        cls,
        *,
        name: str,
        screen_spec: ScreenSpec,
        **kwargs,
    ):
        """Construct a screen-derived variable from a shared ScreenSpec."""
        return cls(
            name=name,
            element_name=screen_spec.element_name,
            pixel_size=screen_spec.pixel_size,
            shape=screen_spec.shape,
            **kwargs,
        )


class ScreenImageVariable(_ScreenSpecVariableMixin, NDVariable, ReadOnlyActionMixin):
    """Read-only action that returns a screen image at a given element. The image is normalized to a unit scale"""

    pixel_size: float  # default pixel size in meters
    read_only: bool = True

    @classmethod
    def from_screen_spec(cls, name: str, screen_spec: ScreenSpec, **kwargs):
        """Build a screen-image variable from a shared screen specification."""
        return cls._from_screen_spec(
            name=name,
            screen_spec=screen_spec,
            **kwargs,
        )

    def _get(self, simulator: Tao) -> Any:
        if simulator.tao_global()["track_type"] != "beam":
            return np.zeros(self.shape)  # empty image for non-beam tracks

        beam = simulator.particles(self.element_name)

        # simple screen that counts number of particles
        half_width = self.screen_spec.half_width
        hist, _ = beam.histogramdd(
            "x",
            "y",
            bins=self.shape,
            range=((-half_width[0], half_width[0]), (-half_width[1], half_width[1])),
        )

        # normalize to unit scale
        hist /= np.max(hist) if np.max(hist) > 0 else 1.0

        return hist


class ScreenResolutionVariable(
    _ScreenSpecVariableMixin, ScalarVariable, ReadOnlyActionMixin
):
    """Read-only action that returns the pixel size in meters."""

    pixel_size: float  # default pixel size in meters
    read_only: bool = True

    @classmethod
    def from_screen_spec(cls, name: str, screen_spec: ScreenSpec, **kwargs):
        """Build a screen-resolution variable from a shared screen specification."""
        return cls._from_screen_spec(
            name=name,
            screen_spec=screen_spec,
            **kwargs,
        )

    def _get(self, simulator: Tao) -> Any:
        _ = simulator
        return self.screen_spec.pixel_size


class ScreenImageShapeVariable(
    _ScreenSpecVariableMixin, IntVariable, ReadOnlyActionMixin
):
    """Read-only action that returns the pixel shape of the screen image."""

    index: int  # index of the dimension to return (0 for x, 1 for y)
    read_only: bool = True

    @classmethod
    def from_screen_spec(cls, name: str, screen_spec: ScreenSpec, **kwargs):
        """Build a screen-image-size variable from a shared screen specification."""
        return cls._from_screen_spec(
            name=name,
            screen_spec=screen_spec,
            **kwargs,
        )

    def _get(self, simulator: Tao) -> Any:
        _ = simulator
        return self.screen_spec.shape[self.index]
