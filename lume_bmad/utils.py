import yaml
from lume.variables import ScalarVariable, NDVariable
from typing import Any
from pytao import Tao
from beamphysics.interfaces.bmad import write_bmad
from pmd_beamphysics import ParticleGroup

# from lcls_live.datamaps import get_datamaps 


TAO_OUTPUT_UNITS = {
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
    "x.eta": "m",
    "x.etap": "",
    "y.eta": "m",
    "y.etap": "",
    "s": "m",
    "l": "m",
    "e_tot": "eV",
    "p0c": "eV",
    "mat6": "",
    "vec0": "m",
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


def get_particle_group_at_element(tao:Tao, element, file_name=None):
    """
    returns a ParticleGroup and optionally writes Bmad particles to file
    """
    parameters =  \
        ["x", "px", "y", "py", "z", "pz", "p0c", "charge", "state", "t", "status"]
    data_bunch1 = {"species": "electron"}
    for par in parameters:
        data_bunch1[par] = tao.bunch1(element, par)
    P = ParticleGroup.from_bmad(data_bunch1)
    if file_name:
        write_bmad(P, file_name, p0c = data_bunch1["p0c"][0])
        print('Wrote Bmad beam file')
    return P


def get_tao_output_variables(tao:Tao) ->dict[str, list[Any]]:
    """
    returns dictionary of output variables

    Parameters
    ----------
    tao: Tao
        Instance of the Tao class.

    Returns
    -------
    dict[str, list[Any]]
    a dictionary of NDVariables

    """
    elements = tao.lat_list("*", "ele.name")
    element_count = len(elements)
    out_dict = {}

    for parameter_name in TAO_OUTPUT_UNITS.keys():
            out_dict[parameter_name] = NDVariable(
                name=parameter_name,
                shape = (1, element_count),
                unit=TAO_OUTPUT_UNITS[parameter_name],
                read_only=True,
            )
    return out_dict

def get_tao_output_parameters():
    return list(TAO_OUTPUT_UNITS.keys())
