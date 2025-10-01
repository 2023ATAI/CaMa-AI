#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on  April  24  08:42 2025
@Author: Qingliang Li: liqingliang@ccsfu.edu.cn （Email）
@Co-author1: Zhongwang Wei:  weizhw6@mail.sysu.edu.cn（Email）
@Co-author2: Kaixuan Cai:  caikx22@mails.jlu.edu.cn（Email）
@Co-author3: Cheng Zhang:  zc24@mails.jlu.edu.cn（Email）
@purpose:  CaMa-Flood default stand-alone driver (python)
Licensed under the Apache License, Version 2.0.

* CONTAINS:
! -- CMF_DRV_INPUT    : Set namelist & logfile
! -- CMF_DRV_INIT     : Initialize        CaMa-Flood
! -- CMF_DRV_END      : Finalize          CaMa-Flood
"""

from cmf_drv_control_mod import CMF_DRV_INPUT,CMF_DRV_INIT,CMF_DRV_END
import cmf_drv_advance_mod
import torch
from fortran_tensor_3D import Ftensor_3D
from parkind1 import Parkind1
import yaml
import argparse
from  run_yml_1 import run_config_1
from  run_yml_2 import run_config_2
from  run_yml_3 import run_config_3
from cmf_utils_mod import CMF_UTILS_MOD
import time
from cmf_ctrl_maps_mod import CMF_MAPS_NMLIST_MOD


# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
def CALC_ADPSTP(DT_DEF, CC_VARS, CC_NMLIST, CM_NMLIST, Datatype, config):
    """
    Dynamically determine the minimum stable time step (DT_MIN) for simulation based on flow velocity, distance,
    and slope. This value is used to update the global simulation time step (DT) and the number of iterations (NT)
    to satisfy the CFL condition (Courant–Friedrichs–Lewy stability criterion).
    """
    device = config['device']
    log_filename = config['RDIR'] + config['LOGOUT']
    DT_MIN = torch.tensor(DT_DEF, device=device)
    NR_Index = torch.arange(1, CM_NMLIST.NSEQRIV + 1, device=device)
    I_E_M = (CM_NMLIST.I2MASK[NR_Index, 1] == 0).nonzero(as_tuple=True)[0]
    if not I_E_M is None:
        CC_VARS.DDPH[NR_Index[I_E_M], 1] = torch.maximum(CC_VARS.D2RIVDPH[NR_Index[I_E_M]-1, 0],
                                                         torch.tensor(0.01, dtype=Datatype.JPRB, device=device))
        CC_VARS.DDST[NR_Index[I_E_M], 1] = CM_NMLIST.D2NXTDST[NR_Index[I_E_M], 1]
        DT_MIN_temp = torch.min(CC_NMLIST.PCADP * CC_VARS.DDST[NR_Index[I_E_M], 1] *
                                (CC_NMLIST.PGRV * CC_VARS.DDPH[NR_Index[I_E_M], 1]) ** (-0.5))
        DT_MIN = torch.minimum(DT_MIN_temp, DT_MIN)
    #   Calculate the minimum time step for river channel cells
    NRA_Index = torch.arange(CM_NMLIST.NSEQRIV + 1, CM_NMLIST.NSEQALL + 1, device=device)
    I_P_M = (CM_NMLIST.I2MASK[NRA_Index, 1] == 0).nonzero(as_tuple=True)[0]
    if not I_P_M is None:
        CC_VARS.DDPH[NRA_Index[I_P_M], 1] = torch.maximum(CC_VARS.D2RIVDPH[NRA_Index[I_P_M]-1, 0],
                                                          torch.tensor(0.01, dtype=Datatype.JPRB, device=device))
        CC_VARS.DDST[NRA_Index[I_P_M], 1] = CC_NMLIST.PDSTMTH
        DT_MIN_temp = torch.min(CC_NMLIST.PCADP * CC_VARS.DDST[NRA_Index[I_P_M], 1] *
                                (CC_NMLIST.PGRV * CC_VARS.DDPH[NRA_Index[I_P_M], 1]) ** (-0.5))
        DT_MIN = torch.minimum(DT_MIN_temp, DT_MIN)

    CC_NMLIST.NT = int(DT_DEF / DT_MIN - 0.01) + 1
    CC_NMLIST.DT = torch.tensor(DT_DEF, dtype=torch.float64) * torch.tensor(CC_NMLIST.NT, dtype=torch.float32).pow(-1)

    if CC_NMLIST.NT > 2:
        with open(log_filename, 'a') as log_file:
            # Write settings to log
            log_file.write(
                f"\nADPSTP: NT={CC_NMLIST.NT:4d}, DT_DEF={DT_DEF:10.2f}, DT_MIN={DT_MIN:10.2f}, DT={CC_NMLIST.DT:10.2f}\n")
            log_file.flush()
            log_file.close()
    return CC_VARS, CC_NMLIST


# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
def main_init_data (config):
# ----------------------------------------------------------------------------------------------------------------------
    print(torch.cuda.is_available())
    Datatype = Parkind1()

    # ----------------------------------------------------------------------------------------------------------------------

    CM_NMLIST                           =               CMF_MAPS_NMLIST_MOD(config, Datatype)
    CM_NMLIST.CMF_MAPS_NMLISTT(config)

    # Read input river map
    CM_NMLIST.CMF_RIVMAP_INIT(Datatype, config)
    CU                                  =               CMF_UTILS_MOD(Datatype, CM_NMLIST)
    # Set topography
    CM_NMLIST.CMF_TOPO_INIT(Datatype, CU, config)

    return CM_NMLIST, CU
# ----------------------------------------------------------------------------------------------------------------------
def main_cmf (config, CM_NMLIST, CU):
# ----------------------------------------------------------------------------------------------------------------------
    Datatype = Parkind1()

    # ----------------------------------------------------------------------------------------------------------------------
    #!*** 1a. Namelist handling
    (CC_NMLIST,      CT_NMLIST,         CF_NMLIST,   CR_NMLIST,       CO_NMLIST) \
                                        =               CMF_DRV_INPUT               (config,    Datatype)


    #!*** 1b. INITIALIZATION
    (CC_NMLIST,             CT_NMLIST,             CM_NMLIST,       CF_NMLIST,          CR_NMLIST,
     CO_NMLIST,             CC_VAR) \
                                        =               (CMF_DRV_INIT
                                                         (CC_NMLIST, CT_NMLIST, CM_NMLIST,    CF_NMLIST,   CR_NMLIST,
                                                          CO_NMLIST, CU,        config,       Datatype    ))

    #!*** 1c. allocate data buffer for input forcing
    ISTEPADV               =           int(CT_NMLIST.NSTEPS/(CC_NMLIST.DTIN / CC_NMLIST.DT))

    ZBUFF_up               =           torch.zeros((ISTEPADV, CC_NMLIST.NXIN, CC_NMLIST.NYIN),
                                                dtype=Datatype.JPRB,device=config['device'])
    ZBUFF_up               =           Ftensor_3D(ZBUFF_up, start_depth=1, start_row=1, start_col=1)


    ZBUFF_sub              =           torch.zeros((ISTEPADV, CC_NMLIST.NXIN, CC_NMLIST.NYIN),
                                                dtype=Datatype.JPRB, device=config['device'])
    ZBUFF_sub              =           Ftensor_3D(ZBUFF_sub, start_depth=1, start_row=1, start_col=1)
# ----------------------------------------------------------------------------------------------------------------------
    # 2. MAIN TEMPORAL LOOP / TIME-STEP (NSTEPS calculated by DRV_INIT)

    ZBUFF_up, ZBUFF_sub    =            (CF_NMLIST.CMF_FORCING_GET
                                        (CC_NMLIST, CT_NMLIST, Datatype, ZBUFF_up, ZBUFF_sub, CU, config))


    # !*  2b Interporlate runoff & send to CaMa-Flood
    CC_VAR                 =            (CF_NMLIST.CMF_FORCING_PUT
                                        (CC_NMLIST, CM_NMLIST, ZBUFF_up, ZBUFF_sub, config, Datatype, CC_VAR))




    ISTEPADV               =           int(CC_NMLIST.DTIN / CC_NMLIST.DT)


    DT_DEF = CC_NMLIST.DT
    if CC_NMLIST.LADPSTP:  # ! adoptive time step
        CC_VARS, CC_NMLIST = CALC_ADPSTP(DT_DEF, CC_VAR, CC_NMLIST, CM_NMLIST, Datatype, config)

    # ----------------------------------------------------------------------------------------------------------------------
    #     for speed up
    CC_VAR.D2SFCELV                 =           CC_VAR.D2SFCELV.raw()
    CM_NMLIST.D2RIVELV              =           CM_NMLIST.D2RIVELV.raw()
    CC_VARS.D2SFCELV_PRE            =           CC_VARS.D2SFCELV_PRE.raw()
    CC_VARS.D2RIVDPH_PRE            =           CC_VARS.D2RIVDPH_PRE.raw()
    CC_VARS.D2FLDDPH_PRE            =           CC_VARS.D2FLDDPH_PRE.raw()
    CM_NMLIST.I1NEXT                =           CM_NMLIST.I1NEXT.raw()
    CM_NMLIST.D2DWNELV              =           CM_NMLIST.D2DWNELV.raw()
    CC_VARS.D2DWNELV_PRE            =           CC_VARS.D2DWNELV_PRE.raw()
    CM_NMLIST.D2NXTDST              =           CM_NMLIST.D2NXTDST.raw()
    CC_VARS.D2RIVOUT_PRE            =           CC_VARS.D2RIVOUT_PRE.raw()
    CM_NMLIST.D2RIVMAN              =           CM_NMLIST.D2RIVMAN.raw()
    CC_VARS.D2RIVOUT                =           CC_VARS.D2RIVOUT.raw()
    CC_VARS.D2FLDSTO_PRE            =           CC_VARS.D2FLDSTO_PRE.raw()
    CC_VARS.D2FLDOUT_PRE            =           CC_VARS.D2FLDOUT_PRE.raw()
    CC_VARS.D2FLDOUT                =           CC_VARS.D2FLDOUT.raw()
    CM_NMLIST.D2ELEVTN              =           CM_NMLIST.D2ELEVTN.raw()
    CC_VARS.D2STORGE                =           CC_VARS.D2STORGE.raw()
    CM_NMLIST.D2RIVHGT              =           CM_NMLIST.D2RIVHGT.raw()
    CC_VARS.D1PTHFLW                =           CC_VARS.D1PTHFLW.raw()
    CM_NMLIST.PTH_UPST              =           CM_NMLIST.PTH_UPST.raw()
    CM_NMLIST.PTH_DOWN              =           CM_NMLIST.PTH_DOWN.raw()
    CM_NMLIST.I2MASK                =           CM_NMLIST.I2MASK.raw()
    CC_VARS.D1PTHFLW_PRE            =           CC_VARS.D1PTHFLW_PRE.raw()
    CM_NMLIST.PTH_WTH               =           CM_NMLIST.PTH_WTH.raw()
    CM_NMLIST.PTH_MAN               =           CM_NMLIST.PTH_MAN.raw()
    CC_VARS.D1PTHFLWSUM             =           CC_VARS.D1PTHFLWSUM.raw()
    CC_VARS.D2RIVINF                =           CC_VARS.D2RIVINF.raw()
    CC_VARS.D2FLDINF                =           CC_VARS.D2FLDINF.raw()
    CC_VARS.D2PTHOUT                =           CC_VARS.D2PTHOUT.raw()
    CC_VARS.D2GDWRTN                =           CC_VARS.D2GDWRTN.raw()
    CC_VARS.D2ROFSUB                =           CC_VARS.D2ROFSUB.raw()
    CC_VARS.P2GDWSTO                =           CC_VARS.P2GDWSTO.raw()
    CC_VARS.D2OUTFLW                =           CC_VARS.D2OUTFLW.raw()
    CC_VARS.D2RUNOFF                =           CC_VARS.D2RUNOFF.raw()

    # ----------------------------------------------------------------------------------------------------------------------

    for ISTEP in range(1, CT_NMLIST.NSTEPS + 1, ISTEPADV):


        #!*** 2c  Advance CaMa-Flood model for ISTEPADV
        ZTT0                                =               (cmf_drv_advance_mod.ADVANCE
                                                             (ISTEPADV,     CC_NMLIST ,     CM_NMLIST,    CT_NMLIST,
                                                              config,       Datatype,       CU,           CC_VAR,
                                                              CO_NMLIST,    CR_NMLIST,      config['device']))

    #!*  2c Prepare forcing for optional sediment transport in stand-alone mode
    if CC_NMLIST.LSEDOUT:
        print("The 'cmf_sed_forcing' code in 81-th Line for main_cmf.py is needed to improved")
        print("The 'cmf_sed_forcing' code in 82-th Line for main_cmf.py is needed to improved")

# ----------------------------------------------------------------------------------------------------------------------

    #   !*** 3a. finalize CaMa-Flood
    CMF_DRV_END     (config,CC_NMLIST,CF_NMLIST,CM_NMLIST,CO_NMLIST,CT_NMLIST,ZTT0)

    return config
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='../gosh/test_US_06min.yml', help='Path to config file')
    args = parser.parse_args()
    torch.use_deterministic_algorithms(True)
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    config = run_config_1(config)

    CM_NMLIST, CU  =  main_init_data(config)

    while config["IYR"] <= config["YEND"]:
        config, start_time = run_config_2(config)
        config = main_cmf(config, CM_NMLIST, CU)
        config = run_config_3(start_time, config)
# ----------------------------------------------------------------------------------------------------------------------