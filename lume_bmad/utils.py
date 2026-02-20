import yaml
from lume.variables import ScalarVariable
from typing import Any
from lume_bmad.transformer import BmadTransformer
from pytao import Tao

# from lcls_live.datamaps import get_datamaps


TAO_OUTPUT_UNITS = {
    "ele.name": "",
    "ele.ix_ele": "",
    "ele.ix_branch": "",
    "ele.a.beta": "m",
    "ele.a.alpha": "",
    "ele.a.eta": "m",
    "ele.a.etap": "",
    "ele.a.gamma": "1/m",
    "ele.a.phi": "",
    "ele.b.beta": "m",
    "ele.b.alpha": "",
    "ele.b.eta": "m",
    "ele.b.etap": "",
    "ele.b.gamma": "1/m",
    "ele.b.phi": "",
    "ele.x.eta": "m",
    "ele.x.etap": "",
    "ele.y.eta": "m",
    "ele.y.etap": "",
    "ele.s": "m",
    "ele.l": "m",
    "ele.e_tot": "eV",
    "ele.p0c": "eV",
    "ele.mat6": "",
    "ele.vec0": "m",
}

###############################################################
# Utility functions for importing control and output variables
################################################################


def import_control_variables(control_variable_file: str):
    """
    Import control variables from a YAML file and define them as Variable instances.
    Also get the mapping between device PV names and Bmad element names.

    TODO: move SLAC specific mapping and unit conversions to slac-tools

    Parameters
    ----------
    control_variable_file: str
        Path to the YAML file containing control variable definitions.

    Returns
    -------
    dict[str, Variable]
        Dictionary of pv variables to Variable instances.
    dict[str, str]
        Mapping between PV names and Bmad element names + attributes.
    """

    control_name_to_bmad = {}
    var_dict = {}

    with open(control_variable_file, "r") as file:
        control_variable = yaml.safe_load(file)

    # handle quadrupoles
    quads = control_variable.get("quad", [])
    for quad in quads:
        pv_name = quad["pvname"]

        # map pv to bmad element name and attribute
        control_name_to_bmad[pv_name] = " ".join(
            [quad["bmad_name"], quad["bmad_attribute"]]
        )
        var_dict[pv_name] = ScalarVariable(
            name=pv_name,
            value_range=(quad["min_value"], quad["max_value"]),
            unit="kG",
            read_only=False,
        )

    return var_dict, control_name_to_bmad


def import_output_variables(output_variable_file: str):
    """
    Import output variables from a YAML file and define them as Variable instances.
    Note that output variables are read-only.

    TODO: move SLAC specific mapping and unit conversions to slac-tools

    Parameters
    ----------
    output_variable_file: str
        Path to the YAML file containing output variable definitions.

    Returns
    -------
    dict[str, Variable]
        Dictionary of output variables mapped by their names.
    """

    out_dict = {}

    with open(output_variable_file, "r") as file:
        output_variables = yaml.safe_load(file)

    for ele in output_variables.keys():
        for attr in output_variables[ele].keys():
            name = attr.replace("ele", "").replace(".", "_")
            name = ele + name + "_"
            out_dict[name] = ScalarVariable(
                name=name,
                unit=TAO_OUTPUT_UNITS[attr],
                read_only=True,
            )

    return out_dict


###############################################################
# Utility classes / functions for Bmad/Tao interaction
###############################################################


class SLAC2BmadTransformer(BmadTransformer):
    def get_tao_property(self, tao: Tao, control_name: str):
        """
        Get a property of an element from Bmad via Tao and
        return its value in control system (EPICS) units.

        # TODO: implment other variable types as needed,
        # utilize future datamaps functionality from lcls-live or database

        Parameters
        ----------
        tao: Tao
            Instance of the Tao class.
        control_name: str
            Name of the control variable to retrieve.

        Returns
        -------
        Any
            Value of the requested property.

        """

        # Map control name to element and attribute
        element, attr = self.control_name_to_bmad[control_name].split(" ")
        ele_attr = tao.ele_gen_attribs(element)
        if attr == "b1_gradient":
            # convert from Bmad units to EPICS units
            return ele_attr["B1_GRADIENT"] * ele_attr["L"] * 10
        else:
            return ele_attr[attr]

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
        tao_cmds = []
        for name, value in pvdata.items():
            element, attr = self.control_name_to_bmad[name].split(" ")
            if attr == "b1_gradient":
                # convert from EPICS units to Bmad units
                ele_attr = tao.ele_gen_attribs(element)
                bmad_value = value / (ele_attr["L"] * 10)
            else:
                bmad_value = value
            tao_cmd = f"set ele {element} {attr} = {bmad_value}"
            tao_cmds.append(tao_cmd)

        return tao_cmds


def evaluate_tao(tao: Tao, tao_cmds: list[str]) -> None:
    """
    Evaluate tao commands, toggles lattice_calculation OFF/ON
    between command list

    Parameters
    ----------
    tao: Tao
        Instance of the Tao class.
    tao_cmds: list[str]
        List of Tao commands to execute.

    Returns
    -------
    None

    """
    tao.cmd("set global lattice_calc_on = F")
    tao.cmds(tao_cmds)
    tao.cmd("set global lattice_calc_on = T")


def get_tao_lat_list_outputs(tao: Tao) -> dict[str, list[Any]]:
    """
    Returns dictionary of Tao output values including element name, twiss and rmats at all elements.

    Parameters
    ----------
    tao: Tao
        Instance of the Tao class.

    Returns
    -------
    dict[str, list[Any]]
        Dictionary mapping output variable names to their values at each element in a lattice.
    """
    # populate output dictionary
    outputs = {}
    lattice_elements = tao.lat_list("*", "ele.name")
    for k in TAO_OUTPUT_UNITS.keys():
        output = tao.lat_list("*", k)
        outputs.update(
            {
                ele + k.replace("ele.", "").replace(".", "_") + "_": val
                for ele, val in zip(lattice_elements, output)
            }
        )

    return outputs
