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

def main_cmAI (config):
# ----------------------------------------------------------------------------------------------------------------------
    print(torch.cuda.is_available())
    Datatype = Parkind1()

    # ----------------------------------------------------------------------------------------------------------------------
    #!*** 1a. Namelist handling
    (CC_NMLIST,      CT_NMLIST,    CM_NMLIST,      CF_NMLIST,   CR_NMLIST,       CO_NMLIST) \
                                        =               CMF_DRV_INPUT               (config,    Datatype)


    #!*** 1b. INITIALIZATION
    (CC_NMLIST,             CT_NMLIST,             CM_NMLIST,       CF_NMLIST,          CR_NMLIST,
     CO_NMLIST,             CC_VAR) \
                                        =               (CMF_DRV_INIT
                                                         (CC_NMLIST, CT_NMLIST, CM_NMLIST,    CF_NMLIST,   CR_NMLIST,
                                                          CO_NMLIST, config,    Datatype    ))

    #!*** 1c. allocate data buffer for input forcing
    ZBUFF               =           torch.zeros((CC_NMLIST.NXIN, CC_NMLIST.NYIN, 2),
                                                dtype=Datatype.JPRB,device=config['device'])
    ZBUFF               =           Ftensor_3D(ZBUFF, start_depth=1, start_row=1, start_col=1)

# ----------------------------------------------------------------------------------------------------------------------
    # 2. MAIN TEMPORAL LOOP / TIME-STEP (NSTEPS calculated by DRV_INIT)

    ISTEPADV            =           int(CC_NMLIST.DTIN / CC_NMLIST.DT)

    CU                  =           CMF_UTILS_MOD          (Datatype,  CC_NMLIST, CM_NMLIST)

    for ISTEP in range(1, CT_NMLIST.NSTEPS + 1, ISTEPADV):
        #   !*  2a Read forcing from file, This is only relevant in Stand-alone mode
        # ! cost test
        if CC_NMLIST.MODTTEST:
            start_moduel                          =       time.time()

        ZBUFF           =           (CF_NMLIST.CMF_FORCING_GET
                                            (CC_NMLIST,      CT_NMLIST,    Datatype,      ZBUFF,      CU,     config))
        # ! cost test
        if CC_NMLIST.MODTTEST:
            end_moduel = time.time()
            CT_NMLIST.test_cost.setdefault('CMF_FORCING_GET', []).append(end_moduel - start_moduel)

        # ! cost test
        if CC_NMLIST.MODTTEST:
            start_moduel = time.time()

       # !*  2b Interporlate runoff & send to CaMa-Flood
        CC_VAR          =                   (CF_NMLIST.CMF_FORCING_PUT
                                             (CC_NMLIST,      CM_NMLIST,     ZBUFF,    config,     Datatype,   CC_VAR))

        # ! cost test
        if CC_NMLIST.MODTTEST:
            end_moduel = time.time()
            CT_NMLIST.test_cost.setdefault('CMF_FORCING_PUT', []).append(end_moduel - start_moduel)

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
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='../gosh/test_US_06min.yml', help='Path to config file')
    args = parser.parse_args()
    torch.use_deterministic_algorithms(True)
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    config = run_config_1(config)
    while config["IYR"] <= config["YEND"]:
        config, start_time = run_config_2(config)
        config = main_cmAI(config)
        config = run_config_3(start_time, config)
# ----------------------------------------------------------------------------------------------------------------------