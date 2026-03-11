from lume.variables import NDVariable
from typing import Any
from pytao import Tao
from beamphysics.interfaces.bmad import write_bmad
from pmd_beamphysics import ParticleGroup


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
