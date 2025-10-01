#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on  April  24  08:42 2025
@Author: Qingliang Li: liqingliang@ccsfu.edu.cn （Email）
@Co-author1: Zhongwang Wei:  weizhw6@mail.sysu.edu.cn（Email）
@Co-author2: Kaixuan Cai:  caikx22@mails.jlu.edu.cn（Email）
@purpose:  call CaMa-Flood physics (python)
Licensed under the Apache License, Version 2.0.

* CONTAINS:
! -- CMF_PROG_INIT      : Initialize Prognostic variables (include restart data handling)
! -- CMF_DIAG_INIT      : Initialize Diagnostic variables
"""
import  os
import torch
import time
from cmf_calc_diag_mod import CMF_DIAG_AVEMAX_ADPSTP,  CMF_DIAG_RESET_ADPSTP, CMF_DIAG_GETAVE_ADPSTP
import cmf_calc_outflw_mod
from cmf_calc_pthout_mod import CMF_CALC_PTHOUT
from cmf_calc_stonxt_mod import CMF_CALC_STONXT
import cmf_calc_fldstg_mod
os.environ['PYTHONWARNINGS']='ignore::FutureWarning'
os.environ['PYTHONWARNINGS']='ignore::RuntimeWarning'

def CMF_PHYSICS_ADVANCE(CC_NMLIST, CM_NMLIST, CT_NMLIST ,log_filename, device, Datatype, CU, CC_VARS):
    """
    """

    def CALC_VARS_PRE(CC_VARS):
        #   ! for river mouth
        RM_Index                                =           torch.arange(0, CM_NMLIST.NSEQMAX)
        CC_VARS.D2RIVOUT_PRE[RM_Index,0]        =           CC_VARS.D2RIVOUT[RM_Index,0]        # !! save outflow (t)
        CC_VARS.D2RIVDPH_PRE[RM_Index,0]        =           CC_VARS.D2RIVDPH[RM_Index,0]        # !! save depth   (t)
        CC_VARS.D2FLDOUT_PRE[RM_Index,0]        =           CC_VARS.D2FLDOUT[RM_Index,0]        # !! save outflow   (t)
        CC_VARS.D2FLDSTO_PRE[RM_Index,0]        =           CC_VARS.P2FLDSTO[RM_Index,0]        # !! save outflow   (t)


        if CC_NMLIST.LPTHOUT:
            CC_VARS.D1PTHFLW_PRE[:,:]          =            CC_VARS.D1PTHFLW[:,:]
        return CC_VARS
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def CALC_WATBAL(IT, CU, CT_NMLIST, CC_VARS, CC_NMLIST, Datatype, device, log_filename):

        DORD = torch.tensor(1e-9, dtype=Datatype.JPRB, device=device)
        # ------------------------------------------------------------------------------------------------------------------
        PKMIN = int(CT_NMLIST.KMIN + IT * CC_NMLIST.DT / 60)
        PYYYYMMDD, PHHMM = CU.MIN2DATE(PKMIN, CT_NMLIST.YYYY0, CT_NMLIST.MM0, CT_NMLIST.DD0)
        PYEAR, PMON, PDAY = CU.SPLITDATE(PYYYYMMDD)
        PHOUR, PMIN = CU.SPLITHOUR(PHHMM)

        # ! poisitive error when water appears from somewhere, negative error when water is lost to somewhere
        # !! water ballance error1 (discharge calculation)   [m3]
        DERROR = - (
                    CC_VARS.P0GLBSTOPRE - CC_VARS.P0GLBSTONXT + CC_VARS.P0GLBRIVINF - CC_VARS.P0GLBRIVOUT)  # !! flux  calc budget error
        # !! water ballance error2 (flood stage calculation) [m3]
        DERROR2 = - (CC_VARS.P0GLBSTOPRE2 - CC_VARS.P0GLBSTONEW2)  # !! flux  calc budget error
        with open(log_filename, 'a') as log_file:
            log_file.write(
                f"{PYEAR:04}/{PMON:02}/{PDAY:02}_{PHOUR:02}:{PMIN:02}"
                f"{IT:6d} flx: "
                f"{(CC_VARS.P0GLBSTOPRE * DORD).item():12.3f}"
                f"{(CC_VARS.P0GLBSTONXT * DORD).item():12.3f}"
                f"{(CC_VARS.P0GLBSTONEW * DORD).item():12.3f}"
                f"{(DERROR * DORD).item():12.3e}  "
                f"{(CC_VARS.P0GLBRIVINF * DORD).item():12.3f}"
                f"{(CC_VARS.P0GLBRIVOUT * DORD).item():12.3f} stg: "
                f"{(CC_VARS.P0GLBSTOPRE2 * DORD).item():12.3f}"
                f"{(CC_VARS.P0GLBSTONEW2 * DORD).item():12.3f}"
                f"{(DERROR2 * DORD).item():12.3e}  "
                f"{(CC_VARS.P0GLBRIVSTO * DORD).item():12.3f}"
                f"{(CC_VARS.P0GLBFLDSTO * DORD).item():12.3f}"
                f"{(CC_VARS.P0GLBFLDARE * DORD).item():12.3f}\n"
            )
            log_file.flush()
            log_file.close()
        return
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    DT_DEF              =           CC_NMLIST.DT
    # !=== 0. calculate river and floodplain stage (for DT calc & )
    CM_NMLIST,CC_VARS   =           CMF_PHYSICS_FLDSTG(CM_NMLIST,CC_NMLIST,CC_VARS,device,Datatype)

    CC_NMLIST.NT = 1

    CC_VARS             =           CMF_DIAG_RESET_ADPSTP(log_filename, CC_VARS, CT_NMLIST, CC_NMLIST, CM_NMLIST, device, Datatype)          #   !! average & max calculation: reset

    #   !! ==========
    for IT in range (1,CC_NMLIST.NT+1):
        CC_NMLIST.TESTIT        =       IT
        # !=== 1. Calculate river discharge
        if CC_NMLIST.LKINE:
            print("The 'CMF_CALC_OUTFLW_KINE' code in 93-th Line for cmf_ctrl_physics_mod.py is needed to improved")
            print("The 'CMF_CALC_OUTFLW_KINE' code in 93-th Line for cmf_ctrl_physics_mod.py is needed to improved")
        elif CC_NMLIST.LSLPMIX:
            print("The 'CMF_CALC_OUTFLW_KINEMIX' code in 96-th Line for cmf_ctrl_physics_mod.py is needed to improved")
            print("The 'CMF_CALC_OUTFLW_KINEMIX' code in 97-th Line for cmf_ctrl_physics_mod.py is needed to improved")
        else:
            print(f'"====  start  CMF_CALC_OUTFLW====')
            print(f'IT:    {IT}')
            #  --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

            CC_VARS     =       cmf_calc_outflw_mod.CMF_CALC_OUTFLW(CC_NMLIST, CM_NMLIST, CC_VARS , device, Datatype)

            #  --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

        if not CC_NMLIST.LFLDOUT:           # !! OPTION: no high-water channel flow
            CC_VARS.D2FLDOUT[:,:]           =       torch.zeros((CM_NMLIST.NSEQMAX, 1), dtype=Datatype._JPRB,device=device)
            CC_VARS.D2FLDOUT_PRE[:, :]      =       torch.zeros((CM_NMLIST.NSEQMAX, 1), dtype=Datatype._JPRB, device=device)

        # --- v4.12: damout before pthout for water buget error
        if CC_NMLIST.LDAMOUT:       #   !! reservoir operation
            print("The 'CMF_DAMOUT_CALC' code in 110-th Line for cmf_ctrl_physics_mod.py is needed to improved")
            print("The 'CMF_DAMOUT_CALC' code in 111-th Line for cmf_ctrl_physics_mod.py is needed to improved")

        # ! --- Bifurcation channel flow
        if CC_NMLIST.LPTHOUT:
            if CC_NMLIST.LLEVEE:
                print("The 'CMF_LEVEE_OPT_PTHOUT' code in 122-th Line for cmf_ctrl_physics_mod.py is needed to improved")
                print("The 'CMF_LEVEE_OPT_PTHOUT' code in 123-th Line for cmf_ctrl_physics_mod.py is needed to improved")
            else:
            #  --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

                CC_VARS   =      CMF_CALC_PTHOUT(CC_NMLIST, CM_NMLIST, CC_VARS , device, Datatype)


            #  --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        # --- Water budget adjustment and calculate inflow

        CC_VARS         =       cmf_calc_outflw_mod.CMF_CALC_INFLOW(CC_NMLIST, CM_NMLIST, CC_VARS, device, Datatype)


        #  --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        if CC_NMLIST.LDAMOUT:       #   !! reservoir operation
            print("The 'CMF_DAMOUT_CALC' code in 116-th Line for cmf_ctrl_physics_mod.py is needed to improved")
            print("The 'CMF_DAMOUT_CALC' code in 117-th Line for cmf_ctrl_physics_mod.py is needed to improved")

        # ! --- save value for next tstet
        CC_VARS           =     CALC_VARS_PRE(CC_VARS)
        #  --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        # !=== 2.  Calculate the storage in the next time step in FTCS diff. eq.

        CC_VARS           =     CMF_CALC_STONXT (CC_NMLIST, CM_NMLIST, CC_VARS , device, Datatype)

        #  --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        #!=== option for ILS coupling
        ##ifdef ILS
        #  --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        # !=== 3. calculate river and floodplain staging

        CM_NMLIST, CC_VARS      =        CMF_PHYSICS_FLDSTG(CM_NMLIST, CC_NMLIST, CC_VARS, device, Datatype)

        #  --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        # !=== 4.  write water balance monitoring to IOFILE

        CALC_WATBAL(IT,CU,CT_NMLIST,CC_VARS,CC_NMLIST,Datatype,device,log_filename)

        #  --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        # !=== 5. calculate averages, maximum
        # ! cost test
        if CC_NMLIST.MODTTEST:
            start_moduel = time.time()

        CC_VARS                 =        CMF_DIAG_AVEMAX_ADPSTP     (CC_NMLIST,CC_VARS,CM_NMLIST,device)

        # ! cost test
        if CC_NMLIST.MODTTEST:
            end_moduel = time.time()
            CT_NMLIST.test_cost.setdefault('CMF_DIAG_AVEMAX_ADPSTP', []).append(end_moduel - start_moduel)
        #  --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        #!=== option for ILS coupling
        ##ifdef ILS
    CC_NMLIST.DT = DT_DEF             #!! reset
    print(f'====  reset DT====:    {CC_NMLIST.DT}')
    #  --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # !=== 5. calculate averages, maximum
    # ! cost test
    if CC_NMLIST.MODTTEST:
        start_moduel = time.time()

    CC_VARS                     =        CMF_DIAG_GETAVE_ADPSTP   (log_filename, CC_VARS, CT_NMLIST, CC_NMLIST, device, Datatype)        #   !! average & max calculation: finalize

    # ! cost test
    if CC_NMLIST.MODTTEST:
        end_moduel = time.time()
        CT_NMLIST.test_cost.setdefault('CMF_DIAG_GETAVE_ADPSTP', []).append(end_moduel - start_moduel)
    #  --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # ! --- Optional: calculate instantaneous discharge (only at the end of outer time step)
    if CC_NMLIST.LOUTINS:                   # !! reservoir operation
        print("The 'CMF_CALC_OUTINS' code in 149-th Line for cmf_ctrl_physics_mod.py is needed to improved")
        print("The 'CMF_CALC_OUTINS' code in 150-th Line for cmf_ctrl_physics_mod.py is needed to improved")

    return CC_VARS, CC_NMLIST, CM_NMLIST

# --------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------
def CMF_PHYSICS_FLDSTG(CM_NMLIST,CC_NMLIST,CC_VARS,device, Datatype):
    """
    ! flood stage scheme selecter
    """
    if CC_NMLIST.LLEVEE:
        # !! levee floodstage (Vector processor option not available)
        print("The 'CMF_LEVEE_FLDSTG' code in xxx-th Line for cmf_ctrl_physics_mod.py is needed to improved")
        print("The 'CMF_LEVEE_FLDSTG' code in xxx-th Line for cmf_ctrl_physics_mod.py is needed to improved")
    else:
        if CC_NMLIST.LSTG_ES:
            #!! Alternative subroutine optimized for vector processor
            print("The 'CMF_OPT_FLDSTG_ES' code in xxx-th Line for cmf_ctrl_physics_mod.py is needed to improved")
            print("The 'CMF_OPT_FLDSTG_ES' code in xxx-th Line for cmf_ctrl_physics_mod.py is needed to improved")
        else:
            CM_NMLIST,CC_VARS = cmf_calc_fldstg_mod.CMF_CALC_FLDSTG_DEF(CM_NMLIST,CC_NMLIST,CC_VARS,device, Datatype)                 #!! Default

    return CM_NMLIST,CC_VARS


