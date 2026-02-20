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
