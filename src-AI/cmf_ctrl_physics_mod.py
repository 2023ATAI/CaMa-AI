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

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def CALC_ADPSTP(DT_DEF, CC_VARS, CC_NMLIST, CM_NMLIST, Datatype,log_filename):
        """
        Dynamically determine the minimum stable time step (DT_MIN) for simulation based on flow velocity, distance,
        and slope. This value is used to update the global simulation time step (DT) and the number of iterations (NT)
        to satisfy the CFL condition (Courant–Friedrichs–Lewy stability criterion).
        """
        DT_MIN = torch.tensor(DT_DEF, device=device)
        NR_Index = torch.arange(0, CM_NMLIST.NSEQRIV, device=device)
        I_E_M = (CM_NMLIST.I2MASK[NR_Index, 0] == 0).nonzero(as_tuple=True)[0]
        if not I_E_M is None:
            CC_VARS.DDPH[NR_Index[I_E_M], 0] = torch.maximum(CC_VARS.D2RIVDPH[NR_Index[I_E_M], 0],
                                                             torch.tensor(0.01, dtype=Datatype.JPRB, device=device))
            CC_VARS.DDST[NR_Index[I_E_M], 0] = CM_NMLIST.D2NXTDST[NR_Index[I_E_M], 0]
            DT_MIN_temp = torch.min(CC_NMLIST.PCADP * CC_VARS.DDST[NR_Index[I_E_M], 0] *
                                    (CC_NMLIST.PGRV * CC_VARS.DDPH[NR_Index[I_E_M], 0]) ** (-0.5))
            DT_MIN = torch.minimum(DT_MIN_temp, DT_MIN)
        #   Calculate the minimum time step for river channel cells
        NRA_Index = torch.arange(CM_NMLIST.NSEQRIV, CM_NMLIST.NSEQALL, device=device)
        I_P_M = (CM_NMLIST.I2MASK[NRA_Index, 0] == 0).nonzero(as_tuple=True)[0]
        if not I_P_M is None:
            CC_VARS.DDPH[NRA_Index[I_P_M], 0] = torch.maximum(CC_VARS.D2RIVDPH[NRA_Index[I_P_M], 0],
                                                              torch.tensor(0.01, dtype=Datatype.JPRB, device=device))
            CC_VARS.DDST[NRA_Index[I_P_M], 0] = CC_NMLIST.PDSTMTH
            DT_MIN_temp = torch.min(CC_NMLIST.PCADP * CC_VARS.DDST[NRA_Index[I_P_M], 0] *
                                    (CC_NMLIST.PGRV * CC_VARS.DDPH[NRA_Index[I_P_M], 0]) ** (-0.5))
            DT_MIN = torch.minimum(DT_MIN_temp, DT_MIN)

        CC_NMLIST.NT = int(DT_DEF / DT_MIN - 0.01) + 1
        CC_NMLIST.DT = torch.tensor(DT_DEF, dtype=torch.float64) * torch.tensor(CC_NMLIST.NT, dtype=torch.float32).pow(
            -1)

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
    DT_DEF = CC_NMLIST.DT
    # !=== 0. calculate river and floodplain stage (for DT calc & )
    CM_NMLIST, CC_VARS  =           CMF_PHYSICS_FLDSTG(CM_NMLIST, CC_NMLIST, CC_VARS, device, Datatype)

    CC_NMLIST.NT        =           1

    CC_VARS, CC_NMLIST  =           CALC_ADPSTP(DT_DEF, CC_VARS, CC_NMLIST, CM_NMLIST, Datatype, log_filename)

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
            # print(f'"====  start  CMF_CALC_OUTFLW====')
            # print(f'IT:    {IT}')
            #  --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

            CC_VARS     =       cmf_calc_outflw_mod.CMF_CALC_OUTFLW(CC_NMLIST, CM_NMLIST, CC_VARS , device, Datatype)

            #  --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

        if not CC_NMLIST.LFLDOUT:           # !! OPTION: no high-water channel flow
            CC_VARS.D2FLDOUT[:,:]           =       torch.zeros((CM_NMLIST.NSEQMAX, 1), dtype=Datatype._JPRB,device=device)
            CC_VARS.D2FLDOUT_PRE[:, :]      =       torch.zeros((CM_NMLIST.NSEQMAX, 1), dtype=Datatype._JPRB, device=device)


        # ! --- Bifurcation channel flow
        if CC_NMLIST.LPTHOUT:
            CC_VARS   =      CMF_CALC_PTHOUT(CC_NMLIST, CM_NMLIST, CC_VARS , device, Datatype)

            #  --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        # --- Water budget adjustment and calculate inflow
        CC_VARS         =       cmf_calc_outflw_mod.CMF_CALC_INFLOW(CC_NMLIST, CM_NMLIST, CC_VARS, device, Datatype)

        #  --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
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

        CC_VARS                 =        CMF_DIAG_AVEMAX_ADPSTP     (CC_NMLIST,CC_VARS,CM_NMLIST,device)

        #  --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        #!=== option for ILS coupling
        ##ifdef ILS
    CC_NMLIST.DT = DT_DEF
    # print(f'====  reset DT====:    {DT_DEF}')
    #  --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # !=== 5. calculate averages, maximum

    CC_VARS                     =        CMF_DIAG_GETAVE_ADPSTP   (log_filename, CC_VARS, CT_NMLIST, CC_NMLIST, device, Datatype)        #   !! average & max calculation: finalize

    #  --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    return CC_VARS, CC_NMLIST, CM_NMLIST

# --------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------
def CMF_PHYSICS_FLDSTG(CM_NMLIST,CC_NMLIST,CC_VARS,device, Datatype):
    """
    ! flood stage scheme selecter
    """
    CM_NMLIST,CC_VARS = cmf_calc_fldstg_mod.CMF_CALC_FLDSTG_DEF(CM_NMLIST,CC_NMLIST,CC_VARS,device, Datatype)                 #!! Default

    return CM_NMLIST,CC_VARS


