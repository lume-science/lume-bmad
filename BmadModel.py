
from typing import Any
import numpy as np
import yaml
from lume.model import LUMEModel
from lume.variables import ScalarVariable
from pytao import Tao
from bmad_model_utils import (get_tao_cmds,  get_tao_output, get_bmad_bdes,
                              get_bmad_klys, tao_output_units, add_klys_STATUS
)

INIT = f'-init $LCLS_LATTICE/bmad/models/cu_hxr/tao.init -noplot'


class BmadModel(LUMEModel):
    """
    Bmad model for CU_HXR 
    
    """
    
    def __init__(self):
        self.tao = Tao(INIT)
        
        # Define control system variables
        with open('hxr_input.yaml', 'r') as file:
            control_variable = yaml.safe_load(file)

        var_dict, control_pvname = {}, []
        device_to_element, element_to_device = {}, {}
        quads = control_variable.get('quad')
        for quad in quads:
            device_to_element[quad['pvname']] = quad['bmad_name']
            element_to_device[quad['bmad_name']] = quad['pvname']
            control_pvname.append(quad['pvname'])
            name = quad['pvname'].replace(':','_')
            var_dict[name] = ScalarVariable(
                name=name,
                value_range=(quad['min_value'], quad['max_value']),
                unit=quad['bmad_unit'],
                read_only=False
            )
                
        #KLYS (Move this to CU_ specific function later)
        klys_keys = [k for k in control_variable.keys() if k[0] =='K']
        for k in klys_keys:
            klys = control_variable.get(k)[0]
            control_pvname.append(klys['ampl_act_pvname'])
            name = klys['ampl_des_pvname'].replace(':','_') 
            var_dict[name] = ScalarVariable(
                name=name,
                value_range=(0, 500),
                unit='MeV',
                read_only=False
            )
            name = klys['phase_act_pvname'].replace(':','_')
            control_pvname.append(klys['phase_des_pvname'])
            var_dict[name] = ScalarVariable(
                name=name,
                value_range=(0, 360),
                unit='Deg_S',
                read_only=False
            )
            name = klys['accelerate_pvname'].replace(':','_')
            control_pvname.append(klys['accelerate_pvname'])
            var_dict[name] = ScalarVariable(
                name=name,
                value_range=(0, 1),
                unit='',
                read_only=False
            ) 
            

        # Define supported read only variables
        with open('hxr_output.yaml', 'r') as file:
            output_variables = yaml.safe_load(file)
                
        out_dict = {}
        for ele in output_variables.keys():
            for attr in output_variables[ele].keys():
                name = attr.replace('ele', '').replace('.','_')                
                name = ele + name + '_'
                out_dict[name] = ScalarVariable(
                     name=name,
                     default_value=0,
                     unit=tao_output_units[attr], 
                     read_only=True
                 )

        self._state = {}
        self.device_to_element = device_to_element
        self.element_to_device = element_to_device
        self.klys_variables = [val for val in control_variable
                               if val.startswith('K')]
        self._pvname = control_pvname        
        self._variables = var_dict
        self._variables.update(out_dict)
        self._initilaize_state()

                
    @property
    def supported_variables(self) -> dict[str, ScalarVariable]:
        """Return the dictionary of supported variables."""
        return self._variables
    
    def _get(self, names: list[str]) -> dict[str, Any]:
        """
        Internal method to retrieve current values for specified variables.
        
        Parameters
        ----------
        names : list[str]
            List of variable names to retrieve
            
        Returns
        -------
        dict[str, Any]
            Dictionary mapping variable names to their current values
        """
        return {name: self._state[name] for name in names}
    
    def _set(self, values: dict[str, Any]) -> None:
        """
        Internal method to set input variables and compute outputs.
        
        This method:
        1. Updates input variables in the state
        2. Performs calculations to update output variables
        3. Stores results in the state
        
        Parameters
        ----------
        values : dict[str, Any]
            Dictionary of variable names and values to set
        """
        # Update input values in state
        pvdata = {}
        for name, value in values.items():
            if not name.endswith('_'):
                pvdata[name.replace('_', ':')] = value
            self._state[name] = value

        #pvdata = add_klys_STATUS(pvdata)
        print(pvdata)
        tao_cmds = get_tao_cmds(pvdata, 'cu_hxr');
        output_tao = evaluate_tao(self.tao, tao_cmds)
        _update_state(output_tao)
                     
    
    def reset(self) -> None:
        """Reset the model to its initial state."""
        self._state = self._initial_state.copy()
        _set(self._state)

    def _tao_output(self, output) -> dict[str, Any]:
        """
        Takes output dictionary containing modeled parameters data in 
        vector from and returns dictionary of scalars; key is element
        name, value is dictoinary modeled attributes 
        e.g. output_data['END']['ele.a.beta']
        """
        output_keys = output.keys()
        output_data = {}
        for indx in output['ele.ix_ele']:
            element = output['ele.name'][indx] 
            output_data[element] = []
            ele_dict = {}
            for out_key in output_keys:
                val = output[out_key][indx]
                if isinstance(val, (np.integer, np.floating)):
                    val = float(val)
                ele_dict[out_key] = val
            output_data[element] = ele_dict
        return output_data



    def write_output_yaml(output_data: dict[str, Any]) -> None:
        with open('hxr_output.yaml', 'w') as yaml_file:
            yaml.dump(output_data, yaml_file,
                default_flow_style=False, sort_keys=False)


    def _initilaize_state(self):
        """ 
        Updates state with Bmad modeled attributes and values of 
        control variables
        """
        output_tao = get_tao_output(self.tao)
        #_update_state(self, output_tao)
        for control_var in self._variables:
            if control_var.endswith('_'):
                continue
            pvname = control_var.replace('_', ':')
            if pvname.startswith('QUAD'):
                element = self.device_to_element[pvname]
                val = get_bmad_bdes(self.tao, element)
                self._state[control_var] = val
        #for klys in self.klys_variables:
        #    ampl, phas, in_use = get_bmad_klys(self.tao, klys)
        #TODO update these values in _state       
                     
        #MOVE this to _init
    def _update_state(self, output_tao):
        """
        write tao output to state
        """
        #GET
        output_data =  _tao_output(output)
        for ele in output_data.keys():
            for attr in output_data[ele].keys():
                var = attr.replace('ele', '').replace('.','_')                
                name =  ele + var + '_'
                self._state[name] = output_data[ele][attr]


                    
                     
                     

    
