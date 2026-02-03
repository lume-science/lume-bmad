from pytao import Tao
import bmad_modeling as mod
import json
import numpy as np
import yaml
from epics import caget

OPTIONS = ' -noplot' 
INIT = f'-init $LCLS_LATTICE/bmad/models/cu_hxr/tao.init {OPTIONS}'
tao = Tao(INIT)

"""
see https://docs.google.com/document/d/14d-ruEJ11zRfsiWoxtEGd3U4o-aBX6MUSZIkazlGiy8/edit?usp=sharing
"""


hxr = mod.BmadModeling('cu_hxr', 'DES') #mdl_obj
output_design = mod.get_output(tao)

def get_value_range(pv):
    bmin, bmax = None, None
    device = pv[0:4]
    if device in ['QUAD','BEND']:
        bmin = caget(pv.replace('BDES','BMIN'))
        bmax = caget(pv.replace('BDES','BMAX'))
    return [bmin, bmax]

#make JSON file
datamaps_keys = hxr.all_data_maps.keys()

data = {}
for dm_key in datamaps_keys:
    data[dm_key] = []
    if dm_key[0] != 'K':
        sub_keys = hxr.all_data_maps[dm_key].data.keys().to_list()
        number_of_elements = len(hxr.all_data_maps[dm_key].data[sub_keys[0]])
        for indx in range(0, number_of_elements):
            ele_dict = {}
            for sk in sub_keys:
                val =  hxr.all_data_maps[dm_key].data[sk][indx]
                if isinstance(val, (np.integer, np.floating)):
                    val = float(val)
                ele_dict[sk] = val
                if sk == 'pvname':
                    value_range = get_value_range(val)
                    ele_dict['min_value'] = value_range[0]
                    ele_dict['max_value'] = value_range[1]
            data[dm_key].append(ele_dict)
    if dm_key[0] == 'K':
        klys_datamap = hxr.all_data_maps[dm_key]
        data[dm_key].append(klys_datamap.asdict())

        
with open('hxr_input.yaml', 'w') as yaml_file:
    yaml.dump(data, yaml_file, default_flow_style=False, sort_keys=False)

           


"""
output dictionary
"""
outkeys = ['ele.name', 'ele.ix_ele', 'ele.ix_branch', 'ele.a.beta',
               'ele.a.alpha', 'ele.a.eta', 'ele.a.etap', 'ele.a.gamma',
               'ele.a.phi', 'ele.b.beta', 'ele.b.alpha', 'ele.b.eta',
               'ele.b.etap', 'ele.b.gamma', 'ele.b.phi', 'ele.x.eta',
               'ele.x.etap', 'ele.y.eta', 'ele.y.etap', 'ele.s', 'ele.l',
               'ele.e_tot', 'ele.p0c', 'ele.mat6', 'ele.vec0']

output = mod.get_output(tao)
output_keys = output.keys()

out_data = {}
for indx in output['ele.ix_ele']:
    element = output['ele.name'][indx] 
    out_data[element] = []
    ele_dict = {}
    for out_key in output_keys:
        val = output[out_key][indx]
        if isinstance(val, (np.integer, np.floating)):
            val = float(val)
        ele_dict[out_key] = val
    out_data[element] = ele_dict

with open('hxr_output.yaml', 'w') as yaml_file:
    yaml.dump(out_data, yaml_file,
        default_flow_style=False, sort_keys=False)
