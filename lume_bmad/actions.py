from typing import Any

from lume.actions import Action, WritableAction
from lume.variables import ScalarVariable, NDVariable
from pytao import Tao

class BmadAction(Action[Tao]):
    """Base class for actions that operate on a Tao object."""

class WritableBmadAction(WritableAction[Tao]):
    """Base class for writable actions that operate on a Tao object."""


class EleAction(WritableBmadAction):
    """Action that operates on a single element in the Bmad model."""

    var: ScalarVariable
    element_name: str
    attribute: str

    def _get(self, simulator: Tao) -> Any:
        return simulator.ele_gen_attribs(self.element_name)[self.attribute]
    
    def _set(self, simulator: Tao, value: Any) -> None:
        simulator.cmd(f"set {self.element_name} {self.attribute} = {value}")

class StatAction(BmadAction):
    """Action that operates on a single statistic in the Bmad model."""

    var: NDVariable
    statistic_name: str

    def _get(self, simulator: Tao) -> Any:
        return simulator.lat_list("*", f"ele.{self.statistic_name}")

class BCTRLAction(WritableBmadAction):
    """Action that operates on a single control variable in the Bmad model."""

    var: ScalarVariable
    element_name: str

    def _get(self, simulator: Tao) -> Any:
        attrs = simulator.ele_gen_attribs(self.element_name)
        return -attrs["b1_gradient"] * attrs["length"] * 10
    
    def _set(self, simulator: Tao, value: Any) -> None:
        attrs = simulator.ele_gen_attribs(self.element_name)
        k1 = -value / (attrs["length"] * 10)
        simulator.cmd(f"set {self.element_name} k1 = {k1}")