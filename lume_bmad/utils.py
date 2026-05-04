import numpy as np
import yaml
from lume.variables import ScalarVariable, NDVariable
from typing import Any
from pytao import Tao

# from lcls_live.datamaps import get_datamaps 


TAO_OUTPUT_UNITS = {
    "name": "",
    "ix_ele": "",
    "ix_branch": "",
    "a.beta": "m",
    "a.alpha": "",
    "a.eta": "m",
    "a.etap": "",
    "a.gamma": "1/m",
    "a.phi": "",
    "b.beta": "m",
    "b.alpha": "",
    "b.eta": "m",
    "b.etap": "",
    "b.gamma": "1/m",
    "b.phi": "",
    "l": "m",
    "e_tot": "eV",
    "p0c": "eV",
    "mat6": "",
    "vec0": "m",
}

TAO_COMB_OUTPUT_UNITS = {
    "s": "m",
    "x": "m",
    "px": "rad",
    "y": "m",
    "py": "rad",
    "t": "s",
    "x.beta": "m",
    "x.alpha": "",
    "x.eta": "m",
    "x.etap": "",
    "x.emit": "m*rad",
    "x.norm_emit": "m*rad",
    "y.beta": "m",
    "y.alpha": "",
    "y.eta": "m",
    "y.etap": "",
    "y.emit": "m*rad",
    "y.norm_emit": "m*rad",
    "n_particle_live": "",
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
        output = tao.lat_list("*", "ele." + k)
        if k == "name":
            outputs[k] = np.asarray(output, dtype=object)
        else:
            outputs.update(
                {
                    ele + k.replace("ele", ""): val
                    for ele, val in zip(lattice_elements, output)
                }
        )

    return outputs


def get_beam_info(tao: Tao) -> dict[str, list[Any]]:
    """
    Returns dictionary of beam tracking information
    
    Parameters
    ----------
    tao: Tao
        Instance of the Tao class.

    Returns
    -------
    dict[str, list[Any]]
        Dictionary mapping beam tracking beam or single particle and beam saved at element list

    """
    beam_info = {}
    lines = tao.cmd('python show beam')
    track_type = [l.split('=') for l in lines if "global%track_type" in l][0][1]
    beam_info['track_type'] =  track_type[2:-1]
    saved_at = [l.split('=') for l in lines if "saved_at" in l][0][1]
    saved_at = saved_at.strip(' "').split(',')
    beam_info['saved_at'] = [s.strip(' ') for s in saved_at]

    return beam_info


def get_tao_output_variables(tao:Tao) ->dict[str, NDVariable]:
    """
    returns dictionary of output variables

    Parameters
    ----------
    tao: Tao
        Instance of the Tao class.

    Returns
    -------
    dict[str, NDVariable]
        A dictionary of NDVariables.

    """
    elements = tao.lat_list("*", "ele.name")
    element_count = len(elements)
    out_dict = {}

    # handle lat_list outputs first
    for parameter_name in TAO_OUTPUT_UNITS.keys():
        if parameter_name in ["name"]:
            # Avoid fixed-width unicode dtypes (<U0, <U12, ...) so any name length is valid.
            data_type_ = object
        elif parameter_name in ["ix_ele"]:
            data_type_ = np.int32
        else:
            data_type_ = float

        if parameter_name == "mat6":
            shape = (element_count, 6, 6)
        elif parameter_name == "vec0":
            shape = (element_count, 6)
        else:
            shape = (element_count,)


        out_dict[parameter_name] = NDVariable(
            name=parameter_name,
            shape = shape,
            unit=TAO_OUTPUT_UNITS[parameter_name],
            read_only=True,
            dtype=data_type_,
        )

    # handle comb outputs
    if tao.tao_global()["track_type"] == "beam":
        s = tao.bunch_comb('s')
        shape = s.shape
        for parameter_name in TAO_COMB_OUTPUT_UNITS.keys():
            out_dict[parameter_name] = NDVariable(
                name=parameter_name,
                shape=shape,
                unit=TAO_COMB_OUTPUT_UNITS[parameter_name],
                read_only=True,
                dtype=float,
            )

    return out_dict


def rmat_get(tao, element_a, element_b, design = False):
    """
    Returns dictionary with Rmat from a to b
    
    Parameters
    ----------
    tao: Tao
        Instance of the Tao class.

    Returns
    -------
    array (6,6) containing Rmat

    """
    if design:
        element_a = element_a + "|design"
    return tao.matrix(element_a, element_b)['mat6']
