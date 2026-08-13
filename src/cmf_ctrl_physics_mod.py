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
from cmf_calc_diag_mod import CMF_DIAG_AVEMAX_ADPSTP,  CMF_DIAG_RESET_ADPSTP, CMF_DIAG_GETAVE_ADPSTP
import cmf_calc_outflw_mod
from cmf_calc_pthout_mod import CMF_CALC_PTHOUT
from cmf_calc_stonxt_mod import CMF_CALC_STONXT
import cmf_calc_fldstg_cpu
import cmf_calc_fldstg_cuda
os.environ['PYTHONWARNINGS']='ignore::FutureWarning'
os.environ['PYTHONWARNINGS']='ignore::RuntimeWarning'


def CMF_PHYSICS_ADVANCE(CC_NMLIST, CM_NMLIST, CT_NMLIST ,log_filename, device, Datatype, CU, CC_VARS, config):
    """
    """
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def CALC_ADPSTP(DT_DEF, CC_VARS, CC_NMLIST, CM_NMLIST, Datatype):
        """
        Dynamically determine the minimum stable time step (DT_MIN) for simulation based on flow velocity, distance,
        and slope. This value is used to update the global simulation time step (DT) and the number of iterations (NT)
        to satisfy the CFL condition (Courant–Friedrichs–Lewy stability criterion).
        """
        if torch.is_tensor(DT_DEF):
            DT_DEF_T                                =           DT_DEF.to(dtype=Datatype.JPRB, device=device)
        else:
            DT_DEF_T                                =           torch.tensor(DT_DEF, dtype=Datatype.JPRB, device=device)
        DT_MIN                                      =           DT_DEF_T.clone()
        NR_Index                                    =           torch.arange(1, CM_NMLIST.NSEQRIV + 1,device=device)
        I_E_M                                       =           (CM_NMLIST.I2MASK[NR_Index, 1] == 0).nonzero(as_tuple=True)[0]
        if not I_E_M is None:
            CC_VARS.DDPH[NR_Index[I_E_M], 1]            =           torch.maximum(CC_VARS.D2RIVDPH[NR_Index[I_E_M], 1],
                                                                              torch.tensor(0.01, dtype=Datatype.JPRB,device=device))
            CC_VARS.DDST[NR_Index[I_E_M], 1]            =           CM_NMLIST.D2NXTDST[NR_Index[I_E_M], 1]
            DT_MIN_temp                                 =           torch.min(CC_NMLIST.PCADP*CC_VARS.DDST[NR_Index[I_E_M],1]*
                                                                              (CC_NMLIST.PGRV * CC_VARS.DDPH[NR_Index[I_E_M] ,1])**(-0.5))
            DT_MIN                                      =           torch.minimum(DT_MIN_temp, DT_MIN)
        #   Calculate the minimum time step for river channel cells
        NRA_Index                                   =           torch.arange(CM_NMLIST.NSEQRIV+1, CM_NMLIST.NSEQALL+1,device=device)
        I_P_M                                       =           (CM_NMLIST.I2MASK[NRA_Index, 1] == 0).nonzero(as_tuple=True)[0]
        if not I_P_M is None:
            CC_VARS.DDPH[NRA_Index[I_P_M], 1]            =           torch.maximum(CC_VARS.D2RIVDPH[NRA_Index[I_P_M], 1],
                                                                                torch.tensor(0.01, dtype=Datatype.JPRB, device=device))
            CC_VARS.DDST[NRA_Index[I_P_M], 1]            =           CC_NMLIST.PDSTMTH
            DT_MIN_temp                                  =           torch.min(CC_NMLIST.PCADP*CC_VARS.DDST[NRA_Index[I_P_M],1]*
                                                                                            (CC_NMLIST.PGRV * CC_VARS.DDPH[NRA_Index[I_P_M] ,1])**(-0.5))
            DT_MIN                                       =           torch.minimum(DT_MIN_temp, DT_MIN)

        DT_DEF_FLOAT                                =           float(DT_DEF_T.detach().cpu().item())
        DT_MIN_FLOAT                                =           float(DT_MIN.detach().cpu().item())
        CC_NMLIST.NT                                =           int(DT_DEF_FLOAT / DT_MIN_FLOAT - 0.01) + 1
        CC_NMLIST.DT                                =           DT_DEF_FLOAT / float(CC_NMLIST.NT)

        if CC_NMLIST.NT > 2:
            with open(log_filename, 'a') as log_file:
                # Write settings to log
                log_file.write(
                    f"\nADPSTP: NT={CC_NMLIST.NT:4d}, DT_DEF={DT_DEF_FLOAT:10.2f}, DT_MIN={DT_MIN_FLOAT:10.2f}, DT={CC_NMLIST.DT:10.2f}\n")
                log_file.flush()
                log_file.close()
        return CC_VARS, CC_NMLIST
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def CALC_VARS_PRE(CC_VARS):
        #   ! for river mouth
        RM_Index                                =           torch.arange(1, CM_NMLIST.NSEQMAX + 1, device=device)
        CC_VARS.D2RIVOUT_PRE[RM_Index,1]        =           CC_VARS.D2RIVOUT[RM_Index,1]        # !! save outflow (t)
        CC_VARS.D2RIVDPH_PRE[RM_Index,1]        =           CC_VARS.D2RIVDPH[RM_Index,1]        # !! save depth   (t)
        CC_VARS.D2FLDOUT_PRE[RM_Index,1]        =           CC_VARS.D2FLDOUT[RM_Index,1]        # !! save outflow   (t)
        CC_VARS.D2FLDSTO_PRE[RM_Index,1]        =           CC_VARS.P2FLDSTO[RM_Index,1]        # !! save outflow   (t)


        if CC_NMLIST.LPTHOUT:
            CC_VARS.D1PTHFLW_PRE[:,:]          =            CC_VARS.D1PTHFLW[:,:]
        return CC_VARS
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def CALC_WATBAL(IT, CU, CT_NMLIST, CC_VARS, CC_NMLIST, Datatype, device, log_filename):

        DORD = torch.tensor(1e-9, dtype=Datatype.JPRB, device=device)
        # ------------------------------------------------------------------------------------------------------------------
        DT_SECONDS = CC_NMLIST.DT.detach().cpu().item() if torch.is_tensor(CC_NMLIST.DT) else CC_NMLIST.DT
        PKMIN = int(CT_NMLIST.KMIN + IT * DT_SECONDS / 60.0)
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

    if CC_NMLIST.LADPSTP:           # ! adoptive time step

        CC_VARS, CC_NMLIST      =       CALC_ADPSTP(DT_DEF, CC_VARS, CC_NMLIST, CM_NMLIST, Datatype)
    CC_VARS             =           CMF_DIAG_RESET_ADPSTP(log_filename, CC_VARS, CT_NMLIST, CC_NMLIST, CM_NMLIST, device, Datatype)          #   !! average & max calculation: reset

    #   !! ==========
    for IT in range (1,CC_NMLIST.NT+1):
        # !=== 1. Calculate river discharge
        if CC_NMLIST.LKINE:
            raise RuntimeError("LKINE is not supported in the formal CaMa-PyTorch v1.0 CPU/CUDA release.")
        elif CC_NMLIST.LSLPMIX:
            raise RuntimeError("LSLPMIX is not supported in the formal CaMa-PyTorch v1.0 CPU/CUDA release.")
        else:
            CC_VARS     =       cmf_calc_outflw_mod.CMF_CALC_OUTFLW(CC_NMLIST, CM_NMLIST, CC_VARS , device, Datatype)
            #  --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

        if not CC_NMLIST.LFLDOUT:           # !! OPTION: no high-water channel flow
            CC_VARS.D2FLDOUT[:,:]           =       torch.zeros((CM_NMLIST.NSEQMAX, 1), dtype=Datatype._JPRB,device=device)
            CC_VARS.D2FLDOUT_PRE[:, :]      =       torch.zeros((CM_NMLIST.NSEQMAX, 1), dtype=Datatype._JPRB, device=device)

        # --- v4.12: damout before pthout for water buget error
        if CC_NMLIST.LDAMOUT:       #   !! reservoir operation
            raise RuntimeError("LDAMOUT is not supported in the formal CaMa-PyTorch v1.0 CPU/CUDA release.")

        # ! --- Bifurcation channel flow
        if CC_NMLIST.LPTHOUT:
            if CC_NMLIST.LLEVEE:
                raise RuntimeError("LLEVEE PTHOUT is not supported in the formal CaMa-PyTorch v1.0 CPU/CUDA release.")
            else:
            #  --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
                CC_VARS   =      CMF_CALC_PTHOUT(CC_NMLIST, CM_NMLIST, CC_VARS , device, Datatype)
            #  --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        # --- Water budget adjustment and calculate inflow
        CC_VARS         =       cmf_calc_outflw_mod.CMF_CALC_INFLOW(CC_NMLIST, CM_NMLIST, CC_VARS, device, Datatype)
        #  --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        if CC_NMLIST.LDAMOUT:       #   !! reservoir operation
            raise RuntimeError("LDAMOUT is not supported in the formal CaMa-PyTorch v1.0 CPU/CUDA release.")

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
    CC_NMLIST.DT = DT_DEF             #!! reset
    #  --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # !=== 5. calculate averages, maximum
    CC_VARS                     =        CMF_DIAG_GETAVE_ADPSTP   (log_filename, CC_VARS, CT_NMLIST, CC_NMLIST, device, Datatype)        #   !! average & max calculation: finalize

    #  --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # ! --- Optional: calculate instantaneous discharge (only at the end of outer time step)
    if CC_NMLIST.LOUTINS:                   # !! reservoir operation
        raise RuntimeError("LOUTINS is not supported in the formal CaMa-PyTorch v1.0 CPU/CUDA release.")

    return CC_VARS, CC_NMLIST, CM_NMLIST

# --------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------
def CMF_PHYSICS_FLDSTG(CM_NMLIST,CC_NMLIST,CC_VARS,device, Datatype):
    """
    ! flood stage scheme selecter
    """
    if CC_NMLIST.LLEVEE:
        raise RuntimeError("LLEVEE FLDSTG is not supported in the formal CaMa-PyTorch v1.0 CPU/CUDA release.")
    if CC_NMLIST.LSTG_ES:
        raise RuntimeError("LSTG_ES is not supported in the formal CaMa-PyTorch v1.0 CPU/CUDA release.")

    backend = torch.device(device).type
    if backend == "cpu":
        CM_NMLIST, CC_VARS = cmf_calc_fldstg_cpu.CMF_CALC_FLDSTG_CPU(
            CM_NMLIST, CC_NMLIST, CC_VARS, device, Datatype
        )
    elif backend == "cuda":
        CM_NMLIST, CC_VARS = cmf_calc_fldstg_cuda.CMF_CALC_FLDSTG_CUDA(
            CM_NMLIST, CC_NMLIST, CC_VARS, device, Datatype
        )
    else:
        raise RuntimeError(
            f"Unsupported CaMa-PyTorch FLDSTG backend: {backend!r}. "
            "Use CPU or CUDA."
        )

    return CM_NMLIST,CC_VARS
