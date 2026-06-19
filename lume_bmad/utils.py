from lume_bmad.actions import CombStatVariable, StatVariable
import numpy as np
import yaml
from lume.variables import ScalarVariable
from pytao import Tao

# from lcls_live.datamaps import get_datamaps


TAO_OUTPUT_UNITS = {
    "name": "",
    "s_ele": "m",
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
    "p0c": "eV/c",
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


def get_tao_stat_output_variables(tao: Tao) -> list[StatVariable]:
    """
    Returns list of StatVariable instances for Tao lattice outputs.

    Parameters
    ----------
    tao: Tao
        Instance of the Tao class.

    Returns
    -------
    list[StatVariable]
        A list of StatVariable instances.

    """
    elements = tao.lat_list("*", "ele.name")
    element_count = len(elements)
    out_list = []

    for parameter_name in TAO_OUTPUT_UNITS.keys():
        if parameter_name in ["name"]:
            # Avoid fixed-width unicode dtypes (<U0, <U12, ...) so any name length is valid.
            data_type_ = np.dtype(object)
        elif parameter_name in ["ix_ele"]:
            data_type_ = np.dtype(np.int32)
        else:
            data_type_ = np.dtype(float)

        if parameter_name == "mat6":
            shape = (element_count, 6, 6)
        elif parameter_name == "vec0":
            shape = (element_count, 6)
        else:
            shape = (element_count,)

        out_list.append(
            StatVariable(
                name=parameter_name,
                statistic_name=parameter_name,
                shape=shape,
                unit=TAO_OUTPUT_UNITS[parameter_name],
                read_only=True,
                dtype=data_type_,
            )
        )

    return out_list


def get_tao_comb_output_variables(tao: Tao) -> list[CombStatVariable]:
    """
    Returns list of CombStatVariable instances for Tao comb outputs.

    Parameters
    ----------
    tao: Tao
        Instance of the Tao class.

    Returns
    -------
    list[CombStatVariable]
        A list of CombStatVariable instances.

    """
    out_list = []

    # handle comb outputs
    if tao.tao_global()["track_type"] == "beam":
        s = tao.bunch_comb("s")
        shape = s.shape
        for parameter_name in TAO_COMB_OUTPUT_UNITS.keys():
            out_list.append(
                CombStatVariable(
                    name=parameter_name,
                    statistic_name=parameter_name,
                    shape=shape,
                    unit=TAO_COMB_OUTPUT_UNITS[parameter_name],
                    read_only=True,
                    dtype=np.dtype(float),
                )
            )

    return out_list


def get_tao_output_variables(tao: Tao) -> list[StatVariable | CombStatVariable]:
    """
    Returns all available Tao output variables, including lattice and comb outputs.

    Parameters
    ----------
    tao: Tao
        Instance of the Tao class.

    Returns
    -------
    list[StatVariable | CombStatVariable]
        A list of StatVariable and CombStatVariable instances.

    """
    return [
        *get_tao_stat_output_variables(tao),
        *get_tao_comb_output_variables(tao),
    ]


def rmat_get(tao, element_a, element_b, design=False):
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
    return tao.matrix(element_a, element_b)["mat6"]
