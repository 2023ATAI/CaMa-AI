#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on  April  24  08:42 2025
@Author: Qingliang Li: liqingliang@ccsfu.edu.cn （Email）
@Co-author1: Zhongwang Wei:  weizhw6@mail.sysu.edu.cn（Email）
@Co-author2: Kaixuan Cai:  caikx22@mails.jlu.edu.cn（Email）
@purpose:  Manage average and max diagnostic vars for output in CaMa-Flood (python)
Licensed under the Apache License, Version 2.0.

!* CONTAINS:
! -- CMF_DIAG_AVE_MAX   : Add / Max of diagnostic variables at time step
! -- CMF_DIAG_AVERAGE   : Calculate time-average of Diagnostic Variables
! -- CMF_DIAG_RESET     : Reset Diagnostic Variables (Average & Maximum )

____________________________________________________________________________________
New:
! -- CMF_DIAG_AVEMAX_ADPSTP   : Add / Max of diagnostic variables within adaptive time step
! -- CMF_DIAG_GETAVE_ADPSTP   : Calculate time-average of Diagnostic Variables for adaptive steps
! -- CMF_DIAG_RESET_ADPSTP    : Reset Diagnostic Variables (Average & Maximum ) for adaptive steps
!
! -- CMF_DIAG_AVEMAX_OUTPUT   : Add / Max of diagnostic variables at time step for output time step
! -- CMF_DIAG_GETAVE_OUTPUT   : Calculate time-average of Diagnostic Variables for output
! -- CMF_DIAG_RESET_OUTPUT    : Reset Diagnostic Variables (Average & Maximum ) for output
"""


import  os

import torch

os.environ['PYTHONWARNINGS']='ignore::FutureWarning'
os.environ['PYTHONWARNINGS']='ignore::RuntimeWarning'
def CMF_DIAG_RESET_ADPSTP(log_filename, CC_VARS, CT_NMLIST, CC_NMLIST, CM_NMLIST, device, Datatype):
    # ------------------------------------------------------------------------------------------------------------------
    with open(log_filename, 'a') as log_file:
        log_file.write(f"CMF::DIAG_AVERAGE: reset, {CT_NMLIST.JYYYYMMDD}, {CT_NMLIST.JHHMM}\n")
        log_file.flush()
        log_file.close()


    CC_VARS.NADD_adp                        =               0
    NQ_Index                                =               torch.arange(1,CM_NMLIST.NSEQMAX+1)


    CC_VARS.D2RIVOUT_aAVG[NQ_Index,:]               =              torch.tensor(0, dtype=Datatype.JPRB,device=device)
    CC_VARS.D2FLDOUT_aAVG[NQ_Index,:]               =              torch.tensor(0, dtype=Datatype.JPRB,device=device)
    CC_VARS.D2OUTFLW_aAVG[NQ_Index,:]               =              torch.tensor(0, dtype=Datatype.JPRB,device=device)
    CC_VARS.D2RIVVEL_aAVG[NQ_Index,:]               =              torch.tensor(0, dtype=Datatype.JPRB,device=device)
    CC_VARS.D2PTHOUT_aAVG[NQ_Index,:]               =              torch.tensor(0, dtype=Datatype.JPRB,device=device)
    CC_VARS.D2GDWRTN_aAVG[NQ_Index,:]               =              torch.tensor(0, dtype=Datatype.JPRB,device=device)
    CC_VARS.D2RUNOFF_aAVG[NQ_Index,:]               =              torch.tensor(0, dtype=Datatype.JPRB,device=device)
    CC_VARS.D2ROFSUB_aAVG[NQ_Index,:]               =              torch.tensor(0, dtype=Datatype.JPRB,device=device)

    CC_VARS.D2STORGE_aMAX[NQ_Index,:]               =              torch.tensor(0, dtype=Datatype.JPRB,device=device)
    CC_VARS.D2OUTFLW_aMAX[NQ_Index,:]               =              torch.tensor(0, dtype=Datatype.JPRB,device=device)
    CC_VARS.D2RIVDPH_aMAX[NQ_Index,:]               =              torch.tensor(0, dtype=Datatype.JPRB,device=device)


    if CC_NMLIST.LDAMOUT:
        print("The 'D2DAMINF_AVG(:,:)' code in 82-th Line for cmf_calc_diag_mod.py is needed to improved")
        print("The 'D2DAMINF_AVG(:,:)' code in 83-th Line for cmf_calc_diag_mod.py is needed to improved")
    if CC_NMLIST.LWEVAP :
        print("The 'D2WEVAPEX_AVG(:,:)' code in 85-th Line for cmf_calc_diag_mod.py is needed to improved")
        print("The 'D2WEVAPEX_AVG(:,:)' code in 86-th Line for cmf_calc_diag_mod.py is needed to improved")

    NT_Index = torch.arange(1, CM_NMLIST.NPTHOUT+1,device=device)
    CC_VARS.D1PTHFLW_aAVG[NT_Index,:]                =              torch.tensor(0, dtype=Datatype.JPRB,device=device)
    CC_VARS.D1PTHFLWSUM_aAVG[NT_Index]               =              torch.tensor(0, dtype=Datatype.JPRB,device=device)

    return CC_VARS
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

def CMF_DIAG_AVEMAX_ADPSTP(CC_NMLIST,CC_VARS,CM_NMLIST,device):
    # ------------------------------------------------------------------------------------------------------------------
    CC_VARS.NADD_adp                            =               CC_VARS.NADD_adp + CC_NMLIST.DT

    NQ_Index                                 =               torch.arange(1, CM_NMLIST.NSEQMAX + 1, device=device)
    CC_VARS.D2RIVOUT_aAVG[NQ_Index,1]        =               CC_VARS.D2RIVOUT_aAVG[NQ_Index,1]    +   CC_VARS.D2RIVOUT[NQ_Index,1] * CC_NMLIST.DT
    CC_VARS.D2FLDOUT_aAVG[NQ_Index,1]        =               CC_VARS.D2FLDOUT_aAVG[NQ_Index,1]    +   CC_VARS.D2FLDOUT[NQ_Index,1] * CC_NMLIST.DT
    CC_VARS.D2RIVVEL_aAVG[NQ_Index,1]        =               CC_VARS.D2RIVVEL_aAVG[NQ_Index,1]    +   CC_VARS.D2RIVVEL[NQ_Index,1] * CC_NMLIST.DT
    CC_VARS.D2OUTFLW_aAVG[NQ_Index,1]        =               CC_VARS.D2OUTFLW_aAVG[NQ_Index,1]    +   CC_VARS.D2OUTFLW[NQ_Index,1] * CC_NMLIST.DT

    CC_VARS.D2PTHOUT_aAVG[NQ_Index,1]        =               (CC_VARS.D2PTHOUT_aAVG[NQ_Index,1]    +   CC_VARS.D2PTHOUT[NQ_Index,1] * CC_NMLIST.DT  -
                                                             CC_VARS.D2PTHINF[NQ_Index,1] * CC_NMLIST.DT)

    CC_VARS.D2GDWRTN_aAVG[NQ_Index,1]        =               CC_VARS.D2GDWRTN_aAVG[NQ_Index,1]    +   CC_VARS.D2GDWRTN[NQ_Index,1] * CC_NMLIST.DT
    CC_VARS.D2RUNOFF_aAVG[NQ_Index,1]        =               CC_VARS.D2RUNOFF_aAVG[NQ_Index,1]    +   CC_VARS.D2RUNOFF[NQ_Index,1] * CC_NMLIST.DT
    CC_VARS.D2ROFSUB_aAVG[NQ_Index,1]        =               CC_VARS.D2ROFSUB_aAVG[NQ_Index,1]    +   CC_VARS.D2ROFSUB[NQ_Index,1] * CC_NMLIST.DT

    CC_VARS.D2OUTFLW_aMAX[NQ_Index,1]        =               torch.maximum( CC_VARS.D2OUTFLW_aMAX[NQ_Index,1], torch.abs(CC_VARS.D2OUTFLW[NQ_Index,1]))
    CC_VARS.D2RIVDPH_aMAX[NQ_Index,1]        =               torch.maximum( CC_VARS.D2RIVDPH_aMAX[NQ_Index,1],          (CC_VARS.D2RIVDPH[NQ_Index,1]))
    CC_VARS.D2STORGE_aMAX[NQ_Index,1]        =               torch.maximum( CC_VARS.D2STORGE_aMAX[NQ_Index,1],          (CC_VARS.D2STORGE[NQ_Index,1]))

    if CC_NMLIST.LWEVAP:
        print("The 'D2WEVAPEX_AVG(ISEQ,1)' code in 48-th Line for cmf_calc_outflw_mod.py is needed to improved")
        print("The 'D2WEVAPEX_AVG(ISEQ,1)' code in 19-th Line for cmf_calc_outflw_mod.py is needed to improved")

    #   !! loop for optional variable (separated for computational efficiency)
    if CC_NMLIST.LDAMOUT:
        print("The 'D2DAMINF_AVG(ISEQ,1)' code in 53-th Line for cmf_calc_outflw_mod.py is needed to improved")
        print("The 'D2DAMINF_AVG(ISEQ,1)' code in 54-th Line for cmf_calc_outflw_mod.py is needed to improved")

    if CC_NMLIST.LPTHOUT:
        NO_Index                            =               torch.arange(1, CM_NMLIST.NPTHOUT + 1)
        CC_VARS.D1PTHFLW_aAVG[NO_Index,1]   =               CC_VARS.D1PTHFLW_aAVG[NO_Index,1]    +   CC_VARS.D1PTHFLW[NO_Index,1]  * CC_NMLIST.DT

    if CC_NMLIST.LSEDOUT:
        print("The 'd2rivvel_sed(ISEQ)' code in 61-th Line for cmf_calc_outflw_mod.py is needed to improved")
        print("The 'd2rivvel_sed(ISEQ)' code in 62-th Line for cmf_calc_outflw_mod.py is needed to improved")
    return CC_VARS
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def CMF_DIAG_GETAVE_ADPSTP(log_filename, CC_VARS, CT_NMLIST, CC_NMLIST, device, Datatype):
    # ------------------------------------------------------------------------------------------------------------------
    with open(log_filename, 'a') as log_file:
        log_file.write(f"CMF::DIAG_AVERAGE: time-average  {CC_VARS.NADD_adp}, {CT_NMLIST.JYYYYMMDD}, {CT_NMLIST.JHHMM}\n")
        log_file.flush()
        log_file.close()

    CC_VARS.D2RIVOUT_aAVG[:,:]               =               CC_VARS.D2RIVOUT_aAVG[:,:]       /       CC_VARS.NADD_adp
    CC_VARS.D2FLDOUT_aAVG[:,:]               =               CC_VARS.D2FLDOUT_aAVG[:,:]       /       CC_VARS.NADD_adp
    CC_VARS.D2OUTFLW_aAVG[:,:]               =               CC_VARS.D2OUTFLW_aAVG[:,:]       /       CC_VARS.NADD_adp
    CC_VARS.D2RIVVEL_aAVG[:,:]               =               CC_VARS.D2RIVVEL_aAVG[:,:]       /       CC_VARS.NADD_adp
    CC_VARS.D2PTHOUT_aAVG[:,:]               =               CC_VARS.D2PTHOUT_aAVG[:,:]       /       CC_VARS.NADD_adp
    CC_VARS.D2GDWRTN_aAVG[:,:]               =               CC_VARS.D2GDWRTN_aAVG[:,:]       /       CC_VARS.NADD_adp
    CC_VARS.D2RUNOFF_aAVG[:,:]               =               CC_VARS.D2RUNOFF_aAVG[:,:]       /       CC_VARS.NADD_adp
    CC_VARS.D2ROFSUB_aAVG[:,:]               =               CC_VARS.D2ROFSUB_aAVG[:,:]       /       CC_VARS.NADD_adp

    if CC_NMLIST.LDAMOUT:
        print("The 'D2DAMINF_aAVG(:,:)' code in 131-th Line for cmf_calc_diag_mod.py is needed to improved")
        print("The 'D2DAMINF_aAVG(:,:)' code in 132-th Line for cmf_calc_diag_mod.py is needed to improved")
    if CC_NMLIST.LWEVAP:
        print("The 'D2WEVAPEX_aAVG(:,:)' code in 134-th Line for cmf_calc_diag_mod.py is needed to improved")
        print("The 'D2WEVAPEX_aAVG(:,:)' code in 135-th Line for cmf_calc_diag_mod.py is needed to improved")

    CC_VARS.D1PTHFLW_aAVG[:,:]               =               CC_VARS.D1PTHFLW_aAVG[:,:]       /       CC_VARS.NADD_adp
    CC_VARS.D1PTHFLWSUM_aAVG[:]              =               CC_VARS.D1PTHFLWSUM_aAVG[:]  +  CC_VARS.D1PTHFLW_aAVG[:,:].sum(dim=1)  #   !! bifurcation height layer summation
    return CC_VARS
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def CMF_DIAG_RESET_OUTPUT(log_filename, CC_VARS, CT_NMLIST, CC_NMLIST, CM_NMLIST, device, Datatype):
    # ------------------------------------------------------------------------------------------------------------------
    with open(log_filename, 'a') as log_file:
        log_file.write(f"CMF::DIAG_AVERAGE: reset, {CT_NMLIST.JYYYYMMDD}, {CT_NMLIST.JHHMM}\n")
        log_file.flush()
        log_file.close()

    CC_VARS.NADD_out                        =               0
    NQ_Index                                =               torch.arange(1,CM_NMLIST.NSEQMAX+1)


    CC_VARS.D2RIVOUT_oAVG[NQ_Index,:]               =              torch.tensor(0, dtype=Datatype.JPRB,device=device)
    CC_VARS.D2FLDOUT_oAVG[NQ_Index,:]               =              torch.tensor(0, dtype=Datatype.JPRB,device=device)
    CC_VARS.D2OUTFLW_oAVG[NQ_Index,:]               =              torch.tensor(0, dtype=Datatype.JPRB,device=device)
    CC_VARS.D2RIVVEL_oAVG[NQ_Index,:]               =              torch.tensor(0, dtype=Datatype.JPRB,device=device)
    CC_VARS.D2PTHOUT_oAVG[NQ_Index,:]               =              torch.tensor(0, dtype=Datatype.JPRB,device=device)
    CC_VARS.D2GDWRTN_oAVG[NQ_Index,:]               =              torch.tensor(0, dtype=Datatype.JPRB,device=device)
    CC_VARS.D2RUNOFF_oAVG[NQ_Index,:]               =              torch.tensor(0, dtype=Datatype.JPRB,device=device)
    CC_VARS.D2ROFSUB_oAVG[NQ_Index,:]               =              torch.tensor(0, dtype=Datatype.JPRB,device=device)

    CC_VARS.D2STORGE_oMAX[NQ_Index,:]               =              torch.tensor(0, dtype=Datatype.JPRB,device=device)
    CC_VARS.D2OUTFLW_oMAX[NQ_Index,:]               =              torch.tensor(0, dtype=Datatype.JPRB,device=device)
    CC_VARS.D2RIVDPH_oMAX[NQ_Index,:]               =              torch.tensor(0, dtype=Datatype.JPRB,device=device)


    if CC_NMLIST.LDAMOUT:
        print("The 'D2DAMINF_AVG(:,:)' code in 82-th Line for cmf_calc_diag_mod.py is needed to improved")
        print("The 'D2DAMINF_AVG(:,:)' code in 83-th Line for cmf_calc_diag_mod.py is needed to improved")
    if CC_NMLIST.LWEVAP :
        print("The 'D2WEVAPEX_AVG(:,:)' code in 85-th Line for cmf_calc_diag_mod.py is needed to improved")
        print("The 'D2WEVAPEX_AVG(:,:)' code in 86-th Line for cmf_calc_diag_mod.py is needed to improved")

    NT_Index = torch.arange(1, CM_NMLIST.NPTHOUT+1)
    CC_VARS.D1PTHFLW_oAVG[NT_Index,:]                =              torch.tensor(0, dtype=Datatype.JPRB,device=device)

    return CC_VARS
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def CMF_DIAG_AVEMAX_OUTPUT (CC_NMLIST,CC_VARS,CM_NMLIST,device):
    # ------------------------------------------------------------------------------------------------------------------
    CC_VARS.NADD_out                         =               CC_VARS.NADD_out + CC_NMLIST.DT

    NQ_Index                                 =               torch.arange(1, CM_NMLIST.NSEQMAX + 1, device=device)
    CC_VARS.D2RIVOUT_oAVG[NQ_Index,1]        =               CC_VARS.D2RIVOUT_oAVG[NQ_Index,1]    +   CC_VARS.D2RIVOUT_aAVG[NQ_Index,1] * CC_NMLIST.DT
    CC_VARS.D2FLDOUT_oAVG[NQ_Index,1]        =               CC_VARS.D2FLDOUT_oAVG[NQ_Index,1]    +   CC_VARS.D2FLDOUT_aAVG[NQ_Index,1] * CC_NMLIST.DT
    CC_VARS.D2RIVVEL_oAVG[NQ_Index,1]        =               CC_VARS.D2RIVVEL_oAVG[NQ_Index,1]    +   CC_VARS.D2RIVVEL_aAVG[NQ_Index,1] * CC_NMLIST.DT
    CC_VARS.D2OUTFLW_oAVG[NQ_Index,1]        =               CC_VARS.D2OUTFLW_oAVG[NQ_Index,1]    +   CC_VARS.D2OUTFLW_aAVG[NQ_Index,1] * CC_NMLIST.DT

    CC_VARS.D2PTHOUT_oAVG[NQ_Index,1]        =               CC_VARS.D2PTHOUT_oAVG[NQ_Index,1]    +   CC_VARS.D2PTHOUT_aAVG[NQ_Index,1] * CC_NMLIST.DT

    CC_VARS.D2GDWRTN_oAVG[NQ_Index,1]        =               CC_VARS.D2GDWRTN_oAVG[NQ_Index,1]    +   CC_VARS.D2GDWRTN_aAVG[NQ_Index,1] * CC_NMLIST.DT
    CC_VARS.D2RUNOFF_oAVG[NQ_Index,1]        =               CC_VARS.D2RUNOFF_oAVG[NQ_Index,1]    +   CC_VARS.D2RUNOFF_aAVG[NQ_Index,1] * CC_NMLIST.DT
    CC_VARS.D2ROFSUB_oAVG[NQ_Index,1]        =               CC_VARS.D2ROFSUB_oAVG[NQ_Index,1]    +   CC_VARS.D2ROFSUB_aAVG[NQ_Index,1] * CC_NMLIST.DT

    CC_VARS.D2OUTFLW_oMAX[NQ_Index,1]        =               torch.maximum( CC_VARS.D2OUTFLW_oMAX[NQ_Index,1], torch.abs(CC_VARS.D2OUTFLW_aMAX[NQ_Index,1]))
    CC_VARS.D2RIVDPH_oMAX[NQ_Index,1]        =               torch.maximum( CC_VARS.D2RIVDPH_oMAX[NQ_Index,1],          (CC_VARS.D2RIVDPH_aMAX[NQ_Index,1]))
    CC_VARS.D2STORGE_oMAX[NQ_Index,1]        =               torch.maximum( CC_VARS.D2STORGE_oMAX[NQ_Index,1],          (CC_VARS.D2STORGE_aMAX[NQ_Index,1]))

    if CC_NMLIST.LWEVAP:
        print("The 'D2WEVAPEX_AVG(ISEQ,1)' code in 48-th Line for cmf_calc_outflw_mod.py is needed to improved")
        print("The 'D2WEVAPEX_AVG(ISEQ,1)' code in 19-th Line for cmf_calc_outflw_mod.py is needed to improved")

    #   !! loop for optional variable (separated for computational efficiency)
    if CC_NMLIST.LDAMOUT:
        print("The 'D2DAMINF_AVG(ISEQ,1)' code in 53-th Line for cmf_calc_outflw_mod.py is needed to improved")
        print("The 'D2DAMINF_AVG(ISEQ,1)' code in 54-th Line for cmf_calc_outflw_mod.py is needed to improved")

    NO_Index                            =               torch.arange(1, CM_NMLIST.NPTHOUT + 1)
    CC_VARS.D1PTHFLW_oAVG[NO_Index,1]   =               CC_VARS.D1PTHFLW_oAVG[NO_Index,1]    +   CC_VARS.D1PTHFLW_aAVG[NO_Index,1]  * CC_NMLIST.DT

    return CC_VARS
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def CMF_DIAG_GETAVE_OUTPUT(log_filename, CC_VARS, CT_NMLIST, CC_NMLIST, device, Datatype):
    # ------------------------------------------------------------------------------------------------------------------
    with open(log_filename, 'a') as log_file:
        log_file.write(f"CMF::DIAG_AVERAGE: time-average  {CC_VARS.NADD_out}, {CT_NMLIST.JYYYYMMDD}, {CT_NMLIST.JHHMM}\n")
        log_file.flush()
        log_file.close()

    CC_VARS.D2RIVOUT_oAVG[:,:]               =               CC_VARS.D2RIVOUT_oAVG[:,:]       /       torch.tensor(CC_VARS.NADD_out, dtype=Datatype.JPRB,device=device)
    CC_VARS.D2FLDOUT_oAVG[:,:]               =               CC_VARS.D2FLDOUT_oAVG[:,:]       /       torch.tensor(CC_VARS.NADD_out, dtype=Datatype.JPRB,device=device)
    CC_VARS.D2OUTFLW_oAVG[:,:]               =               CC_VARS.D2OUTFLW_oAVG[:,:]       /       torch.tensor(CC_VARS.NADD_out, dtype=Datatype.JPRB,device=device)
    CC_VARS.D2PTHOUT_oAVG[:,:]               =               CC_VARS.D2PTHOUT_oAVG[:,:]       /       torch.tensor(CC_VARS.NADD_out, dtype=Datatype.JPRB,device=device)
    CC_VARS.D2GDWRTN_oAVG[:,:]               =               CC_VARS.D2GDWRTN_oAVG[:,:]       /       torch.tensor(CC_VARS.NADD_out, dtype=Datatype.JPRB,device=device)
    CC_VARS.D2RUNOFF_oAVG[:,:]               =               CC_VARS.D2RUNOFF_oAVG[:,:]       /       torch.tensor(CC_VARS.NADD_out, dtype=Datatype.JPRB,device=device)
    CC_VARS.D2ROFSUB_oAVG[:,:]               =               CC_VARS.D2ROFSUB_oAVG[:,:]       /       torch.tensor(CC_VARS.NADD_out, dtype=Datatype.JPRB,device=device)

    if CC_NMLIST.LDAMOUT:
        print("The 'D2DAMINF_AVG(:,:)' code in 82-th Line for cmf_calc_diag_mod.py is needed to improved")
        print("The 'D2DAMINF_AVG(:,:)' code in 83-th Line for cmf_calc_diag_mod.py is needed to improved")
    if CC_NMLIST.LDAMOUT:
        print("The 'D2WEVAPEX_AVG(:,:)' code in 85-th Line for cmf_calc_diag_mod.py is needed to improved")
        print("The 'D2WEVAPEX_AVG(:,:)' code in 86-th Line for cmf_calc_diag_mod.py is needed to improved")

    CC_VARS.D1PTHFLW_oAVG[:,:]               =               CC_VARS.D1PTHFLW_oAVG[:,:]       /       torch.tensor(CC_VARS.NADD_out, dtype=Datatype.JPRB,device=device)
    return CC_VARS