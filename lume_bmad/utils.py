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
