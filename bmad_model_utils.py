from lcls_live.datamaps import get_datamaps

tao_output_units = {
    'ele.name':'', 'ele.ix_ele':'', 'ele.ix_branch':'', 'ele.a.beta':'m',
    'ele.a.alpha':'', 'ele.a.eta':'m', 'ele.a.etap':'', 'ele.a.gamma':'1/m',
    'ele.a.phi':'', 'ele.b.beta':'m', 'ele.b.alpha':'', 'ele.b.eta':'m',
    'ele.b.etap':'', 'ele.b.gamma':'1/m', 'ele.b.phi':'', 'ele.x.eta':'m',
    'ele.x.etap':'', 'ele.y.eta':'m', 'ele.y.etap':'', 'ele.s':'m', 'ele.l':'m',
    'ele.e_tot':'eV', 'ele.p0c':'eV', 'ele.mat6':'', 'ele.vec0':'m'}


def evaluate_tao(tao, tao_cmds):
    """
    Evaluate tao commands, toggles lattice_calculation OFF/ON
    between command list
    """
    tao.cmd("set global lattice_calc_on = F")
    tao.cmds(tao_cmds)
    tao.cmd("set global lattice_calc_on = T")
    output = get_output(tao)
    return output


def get_tao_output(tao):
    """
    Returns dictionary of modeled parameters, including element name,
    twiss and Rmats
    """
    outkeys = ['ele.name', 'ele.ix_ele', 'ele.ix_branch', 'ele.a.beta',
               'ele.a.alpha', 'ele.a.eta', 'ele.a.etap', 'ele.a.gamma',
               'ele.a.phi', 'ele.b.beta', 'ele.b.alpha', 'ele.b.eta',
               'ele.b.etap', 'ele.b.gamma', 'ele.b.phi', 'ele.x.eta',
               'ele.x.etap', 'ele.y.eta', 'ele.y.etap', 'ele.s', 'ele.l',
               'ele.e_tot', 'ele.p0c', 'ele.mat6', 'ele.vec0']
    for key in outkeys:
        output = {k: tao.lat_list("*", k) for k in outkeys}
    return output


def get_bmad_bdes(tao, element, b1_gradient=[]):
    """Returns BDES from Bmad B1_GRADIENT or given gradient"""
    ele_attr = tao.ele_gen_attribs(element)
    if not b1_gradient:
        b1_gradient = ele_attr["B1_GRADIENT"]
    return -b1_gradient * ele_attr["L"] * 10


def get_tao_cmds(pvdata, beam_path):
    """
    Returns tao commands from pvdata, if data_source is DES, 
    calls use_klys_when_beam off for Cu Linac
    """
    all_data_maps = get_datamaps(beam_path)
    lines_quads, lines_rf = [], []
    #pvdata = add_klys_STATUS(pvdata)
    for dm_key, map in all_data_maps.items():
        if dm_key.startswith("K"):
            acc_pv = map.accelerate_pvname
            if acc_pv == "":
                continue
            lines_rf += map.as_tao(pvdata)
        if dm_key == "cavities":
            lines_rf = map.as_tao(pvdata)
        if dm_key == "quad":
            lines_quads += map.as_tao(pvdata)
    if "NOT" in mdl_obj.data_source and "cu" in beam_path:
        new_lines = []
        for cmd in lines_rf:
            if "in_use" in cmd:
                if "K21_1" in cmd or "K21_2" in cmd:
                    new_lines.append(cmd)  # L1 always on Beam Code 1
                    continue
                ele = cmd.split()[2]
                [sector, station] = ele[1:].split("_")
                pv = f"KLYS:LI{sector}:{station}1:"
                f"BEAMCODE{mdl_obj.beam_code}_STAT"
                cmd_words = cmd.split()
                cmd_words[-1] = str(pvdata[pv])
                new_lines.append(" ".join(cmd_words))
            else:
                new_lines.append(cmd)
    return lines_rf + lines_quads


def get_bmad_klys(tao, klys):
    cmd = f'show lat {klys} -attr ENLD_MEV -attr PHASE_DEG -attr in_use '
    + '-no_label_lines'
    result = tao.cmd(cmd)[0].split()
    return  result[5], result[6], result[7]


def add_klys_STATUS(pvdata):
    new_data = {}
    for pv in pvdata.keys():
        if pv.endswith('ENLD'):            
            new_data[pv.replace('ENLD','STAT')] = 1
            new_data[pv.replace('ENLD','SWRD')] = 1
            new_data[pv.replace('ENLD','HDSC')] = 1
            new_data[pv.replace('ENLD','DSTA')] = (1,1)
    return pvdata.update(new_data)

