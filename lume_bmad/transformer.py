from typing import Any
from pytao import Tao
from abc import ABC, abstractmethod


class BmadTransformer(ABC):
    """
    Class that handles transformations between control system
    names and values and bmad element names and attributes.

    Should be subclassed for each specific accelerator facility to handle
    any necessary unit conversions or special cases in the mapping
    between control variables and Bmad properties.

    Attributes
    ----------
    control_name_to_bmad: dict[str, str]
        Mapping between control variable names and Bmad element names.
        Example: {"QUAD:IN20:511": "Q1"}

    """

    def __init__(self, control_name_to_bmad: dict[str, str]):
        self._control_name_to_bmad = control_name_to_bmad

    @property
    def control_name_to_bmad(self) -> dict[str, str]:
        return self._control_name_to_bmad

    @abstractmethod
    def get_tao_property(self, tao: Tao, control_name: str):
        """
        Get a property of an element from Bmad via Tao and
        return its value in control system (EPICS) units.

        Parameters
        ----------
        tao: Tao
            Tao instance to use for retrieving the property value.
        control_name: str
            Name of the control variable for which to retrieve the corresponding Bmad property.

        Returns
        -------
        Any
            Value of the requested property.

        """
        pass

    @abstractmethod
    def get_tao_commands(
        self, tao: Tao, pvdata: dict[str, Any], beam_path: str
    ) -> list[str]:
        """
        Get Tao commands to set a property of an element in Bmad via Tao. Handle
        mapping control names to element attributes and any necessary unit conversions as needed.

        Parameters
        ----------
        tao: Tao
            Instance of the Tao class.
        pvdata: dict[str, Any]
            Dictionary of control variable names and values to set
        beam_path: str
            Beam path in the Bmad lattice (e.g. "cu_hxr")

        Returns
        -------
        list[str]
            List of Tao commands to execute

        """
        pass


class BasicTransformer(BmadTransformer):
    """
    Basic implementation of the BmadTransformer that assumes control variable names are the same as Bmad element names
    and that values can be set directly without any unit conversions or special handling.

    This can be used as a template for implementing more complex transformers for specific facilities.

    Assumes control variable names are in the format "ELEMENT:ATTRIBUTE" where ELEMENT is the name of the Bmad element and
    ATTRIBUTE is the name of the attribute to set for that element. For example, "QUAD:K1" would
    refer to the K1 attribute of the QUAD element.

    """

    def get_tao_property(self, tao: Tao, control_name: str):
        element_name = control_name.split(":")[0]
        attr = control_name.split(":")[1]
        return tao.ele_gen_attribs(element_name)[attr]

    def get_tao_commands(self, tao: Tao, pvdata: dict[str, Any]) -> list[str]:
        commands = []
        for control_name, value in pvdata.items():
            bmad_name = control_name.split(":")[0]
            attr = control_name.split(":")[1]
            commands.append(f"set ele {bmad_name} {attr} = {value}")
        return commands
