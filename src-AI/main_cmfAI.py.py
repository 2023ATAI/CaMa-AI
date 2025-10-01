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
from cmf_ctrl_nmlist_mod import CMF_CTRL_NMLIST_MOD
from cmf_ctrl_maps_mod import CMF_MAPS_NMLIST_MOD



def main_init_data (config):
# ----------------------------------------------------------------------------------------------------------------------
    print(torch.cuda.is_available())
    Datatype = Parkind1()

    log_filename = config['RDIR'] + config['LOGOUT']
    #!*** 1. CaMa-Flood configulation namelist
    CC_NMLIST                    =       CMF_CTRL_NMLIST_MOD             (config,    Datatype)
    CC_NMLIST.log_settings                                               (config)
    # ----------------------------------------------------------------------------------------------------------------------

    CM_NMLIST                           =               CMF_MAPS_NMLIST_MOD(config, Datatype, CC_NMLIST)
    CM_NMLIST.CMF_MAPS_NMLISTT(config, CC_NMLIST)

    # Read input river map
    CM_NMLIST.CMF_RIVMAP_INIT(CC_NMLIST,         log_filename,            Datatype,         config)
    CU                                  =               CMF_UTILS_MOD(Datatype, CC_NMLIST, CM_NMLIST)
    # Set topography
    CM_NMLIST.CMF_TOPO_INIT(CC_NMLIST,       log_filename,      Datatype,         CU)

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


    # DT_DEF = CC_NMLIST.DT
    # if CC_NMLIST.LADPSTP:  # ! adoptive time step
    #     CC_VARS, CC_NMLIST = CALC_ADPSTP(DT_DEF, CC_VAR, CC_NMLIST, CM_NMLIST, Datatype, config)

    # ----------------------------------------------------------------------------------------------------------------------
    #     for speed up
    CC_VAR.D2SFCELV                 =           CC_VAR.D2SFCELV.raw()
    CM_NMLIST.D2RIVELV              =           CM_NMLIST.D2RIVELV.raw()
    CC_VAR.D2SFCELV_PRE            =           CC_VAR.D2SFCELV_PRE.raw()
    CC_VAR.D2RIVDPH_PRE            =           CC_VAR.D2RIVDPH_PRE.raw()
    CC_VAR.D2FLDDPH_PRE            =           CC_VAR.D2FLDDPH_PRE.raw()
    CM_NMLIST.I1NEXT                =           CM_NMLIST.I1NEXT.raw()
    CM_NMLIST.D2DWNELV              =           CM_NMLIST.D2DWNELV.raw()
    CC_VAR.D2DWNELV_PRE            =           CC_VAR.D2DWNELV_PRE.raw()
    CM_NMLIST.D2NXTDST              =           CM_NMLIST.D2NXTDST.raw()
    CC_VAR.D2RIVOUT_PRE            =           CC_VAR.D2RIVOUT_PRE.raw()
    CM_NMLIST.D2RIVMAN              =           CM_NMLIST.D2RIVMAN.raw()
    CC_VAR.D2RIVOUT                =           CC_VAR.D2RIVOUT.raw()
    CC_VAR.D2FLDSTO_PRE            =           CC_VAR.D2FLDSTO_PRE.raw()
    CC_VAR.D2FLDOUT_PRE            =           CC_VAR.D2FLDOUT_PRE.raw()
    CC_VAR.D2FLDOUT                =           CC_VAR.D2FLDOUT.raw()
    CM_NMLIST.D2ELEVTN              =           CM_NMLIST.D2ELEVTN.raw()
    CC_VAR.D2STORGE                =           CC_VAR.D2STORGE.raw()
    CM_NMLIST.D2RIVHGT              =           CM_NMLIST.D2RIVHGT.raw()
    CC_VAR.D1PTHFLW                =           CC_VAR.D1PTHFLW.raw()
    CM_NMLIST.PTH_UPST              =           CM_NMLIST.PTH_UPST.raw()
    CM_NMLIST.PTH_DOWN              =           CM_NMLIST.PTH_DOWN.raw()
    CM_NMLIST.I2MASK                =           CM_NMLIST.I2MASK.raw()
    CC_VAR.D1PTHFLW_PRE            =           CC_VAR.D1PTHFLW_PRE.raw()
    CM_NMLIST.PTH_WTH               =           CM_NMLIST.PTH_WTH.raw()
    CM_NMLIST.PTH_MAN               =           CM_NMLIST.PTH_MAN.raw()
    CC_VAR.D1PTHFLWSUM             =           CC_VAR.D1PTHFLWSUM.raw()
    CC_VAR.D2RIVINF                =           CC_VAR.D2RIVINF.raw()
    CC_VAR.D2FLDINF                =           CC_VAR.D2FLDINF.raw()
    CC_VAR.D2PTHOUT                =           CC_VAR.D2PTHOUT.raw()
    CC_VAR.D2GDWRTN                =           CC_VAR.D2GDWRTN.raw()
    CC_VAR.D2ROFSUB                =           CC_VAR.D2ROFSUB.raw()
    CC_VAR.P2GDWSTO                =           CC_VAR.P2GDWSTO.raw()
    CC_VAR.D2OUTFLW                =           CC_VAR.D2OUTFLW.raw()
    CC_VAR.D2RUNOFF                =           CC_VAR.D2RUNOFF.raw()
    CM_NMLIST.PTH_ELV               =           CM_NMLIST.PTH_ELV.raw()
    CM_NMLIST.PTH_DST               =           CM_NMLIST.PTH_DST.raw()
    # ----------------------------------------------------------------------------------------------------------------------
    ISTEP_   =   1
    for ISTEP in range(1, CT_NMLIST.NSTEPS + 1, ISTEPADV):
        CC_VAR.D2RUNOFF[:, 0]              =           CC_VAR.D2RUNOFF_year[ISTEP_, :]
        CC_VAR.D2ROFSUB[:, 0]              =           CC_VAR.D2ROFSUB_year[ISTEP_, :]
        #!*** 2c  Advance CaMa-Flood model for ISTEPADV
        ZTT0                                =               (cmf_drv_advance_mod.ADVANCE
                                                             (ISTEPADV,     CC_NMLIST ,     CM_NMLIST,    CT_NMLIST,
                                                              config,       Datatype,       CU,           CC_VAR,
                                                              CO_NMLIST,    CR_NMLIST,      config['device']))
        ISTEP_      =       ISTEP_  +   1
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
    config, start_time = run_config_2(config)
    CM_NMLIST, CU  =  main_init_data(config)

    while config["IYR"] <= config["YEND"]:
        config, start_time = run_config_2(config)
        config = main_cmf(config, CM_NMLIST, CU)
        config = run_config_3(start_time, config)
# ----------------------------------------------------------------------------------------------------------------------