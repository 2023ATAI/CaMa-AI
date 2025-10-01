#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on  April  24  08:42 2025
@Author: Qingliang Li: liqingliang@ccsfu.edu.cn （Email）
@Co-author1: Zhongwang Wei:  weizhw6@mail.sysu.edu.cn（Email）
@Co-author2: Kaixuan Cai:  caikx22@mails.jlu.edu.cn（Email）
@purpose:  CaMa-Flood physics for river&floodplain discharge (python)
Licensed under the Apache License, Version 2.0.

!* CONTAINS:
! -- CMF_CALC_OUTFLW
! -- CMF_CALC_INFLOW
! --
"""
import  os

import numpy as np
import torch
from fortran_tensor_3D import Ftensor_3D
from fortran_tensor_2D import Ftensor_2D
from fortran_tensor_1D import Ftensor_1D

os.environ['PYTHONWARNINGS']='ignore::FutureWarning'
os.environ['PYTHONWARNINGS']='ignore::RuntimeWarning'

def CMF_CALC_OUTFLW( CC_NMLIST, CM_NMLIST, CC_VARS , device, Datatype):
    """
    To compute river (D2RIVOUT) and floodplain (D2FLDOUT) discharge based on water surface slope, storage,
    and physical river/floodplain properties.
    """
    #   1. Preprocessing Water Surface Elevation
    CC_VARS.D2SFCELV[:CM_NMLIST.NSEQALL + 1, 1] \
                                        =           CM_NMLIST.D2RIVELV[:CM_NMLIST.NSEQALL + 1, 1] + CC_VARS.D2RIVDPH[:CM_NMLIST.NSEQALL + 1, 1]
    #   !! water surface elevation (t-1) [m]
    CC_VARS.D2SFCELV_PRE[:CM_NMLIST.NSEQALL + 1, 1]\
                                        =           CM_NMLIST.D2RIVELV[:CM_NMLIST.NSEQALL + 1, 1] + CC_VARS.D2RIVDPH_PRE[:CM_NMLIST.NSEQALL + 1, 1]
    #   !! floodplain depth (t-1)        [m]
    CC_VARS.D2FLDDPH_PRE[:CM_NMLIST.NSEQALL + 1, 1] \
                                        =           torch.maximum(CC_VARS.D2RIVDPH_PRE[:CM_NMLIST.NSEQALL + 1, 1] - CM_NMLIST.D2RIVHGT[:CM_NMLIST.NSEQALL + 1, 1],
                                                              torch.tensor(0.0, dtype=Datatype.JPRB, device=device))
    #   2. Loop Over River Cells
        # !Update downstream elevation
    RC_Index                            =           torch.arange(1, CM_NMLIST.NSEQRIV + 1,device=device)
    JSEQ                                =           CM_NMLIST.I1NEXT[RC_Index]      # ! next cell's pixel
    CM_NMLIST.D2DWNELV[RC_Index,1]      =           CC_VARS.D2SFCELV[JSEQ,1]
    CC_VARS.D2DWNELV_PRE[RC_Index,1]    =           CC_VARS.D2SFCELV_PRE[JSEQ,1]

    #   !! for normal cells
    DSFC                                =           torch.maximum(CC_VARS.D2SFCELV[RC_Index, 1],    CM_NMLIST.D2DWNELV[RC_Index, 1])
    DSLP                                =           ((CC_VARS.D2SFCELV[RC_Index, 1] - CM_NMLIST.D2DWNELV[RC_Index, 1]) *
                                                     CM_NMLIST.D2NXTDST[RC_Index, 1] ** (-1))
    #   !=== River Flow ===
    DFLW                                =           DSFC    -   CM_NMLIST.D2RIVELV[RC_Index, 1]             #   !!  flow cross-section depth
    DARE                                =           torch.maximum(CM_NMLIST.D2RIVWTH[RC_Index, 1] * DFLW,   #   !!  flow cross-section area
                                                                  torch.tensor(1e-10,dtype=Datatype.JPRB,device=device))

    DSFC_pr                             =           torch.maximum(CC_VARS.D2SFCELV_PRE[RC_Index, 1],    CC_VARS.D2DWNELV_PRE[RC_Index, 1])
    DFLW_pr                             =           DSFC_pr   -   CM_NMLIST.D2RIVELV[RC_Index, 1]
    DFLW_im                             =           torch.maximum((DFLW * DFLW_pr) ** 0.5,
                                                                  torch.tensor(1e-6,dtype=Datatype.JPRB,device=device))     #   !! semi implicit flow depth

    DOUT_pr                             =           CC_VARS.D2RIVOUT_PRE[RC_Index,1] * CM_NMLIST.D2RIVWTH[RC_Index, 1] ** (-1)  #   !! outflow (t-1) [m2/s] (unit width)
    DOUT                                =           (CM_NMLIST.D2RIVWTH[RC_Index, 1] * (DOUT_pr + CC_NMLIST.PGRV * CC_NMLIST.DT * DFLW_im * DSLP) *
                                                     (1 + CC_NMLIST.PGRV * CC_NMLIST.DT * CM_NMLIST.D2RIVMAN[RC_Index, 1] ** 2 * torch.abs(DOUT_pr) *
                                                     DFLW_im ** (-7/3)) ** (-1))
    DVEL                                =           CC_VARS.D2RIVOUT[RC_Index,1] * DARE ** (-1)


    Mask                                =           (DFLW_im > 1e-5) & (DARE > 1e-5)
    CC_VARS.D2RIVOUT[RC_Index, 1]       =           torch.where(Mask, DOUT, torch.tensor(0.0, dtype=Datatype.JPRB, device=device))
    CC_VARS.D2RIVVEL[RC_Index, 1]       =           torch.where(Mask, DVEL, torch.tensor(0.0, dtype=Datatype.JPRB, device=device))

    #!=== Floodplain Flow ===
    if CC_NMLIST.LFLDOUT:
        DFSTO                           =           CC_VARS.P2FLDSTO[RC_Index,1]
        DSFC                            =           torch.maximum(CC_VARS.D2SFCELV[RC_Index, 1], CM_NMLIST.D2DWNELV[RC_Index, 1])
        DSLP                            =           ((CC_VARS.D2SFCELV[RC_Index, 1] - CM_NMLIST.D2DWNELV[RC_Index, 1]) *
                                                     CM_NMLIST.D2NXTDST[RC_Index, 1] ** (-1))
        DSLP                            =           torch.maximum(-torch.tensor(0.005,dtype=Datatype.JPRB,device=device),
                                                    torch.minimum( torch.tensor(0.005,dtype=Datatype.JPRB,device=device), DSLP))    #   !! set max&min [instead of using weir equation for efficiency]


        DFLW                            =           torch.maximum(DSFC - CM_NMLIST.D2ELEVTN[RC_Index, 1],
                                                                  torch.tensor(0,dtype=Datatype.JPRB,device=device))
        DARE                            =           DFSTO * CM_NMLIST.D2RIVLEN[RC_Index,1] ** (-1)
        DARE                            =           torch.maximum(DARE -  CC_VARS.D2FLDDPH[RC_Index,1] * CM_NMLIST.D2RIVWTH[RC_Index,1],
                                                                  torch.tensor(0.0, dtype=Datatype.JPRB, device=device))    #   !! remove above river channel area


        DSFC_pr                         =           torch.maximum(CC_VARS.D2SFCELV_PRE[RC_Index, 1], CC_VARS.D2DWNELV_PRE[RC_Index, 1])
        DFLW_pr                         =           DSFC_pr -  CM_NMLIST.D2ELEVTN[RC_Index, 1]
        DFLW_im                         =           torch.maximum(
                                                        torch.maximum(  (DFLW * DFLW_pr),   torch.tensor(0, dtype=Datatype.JPRB, device=device)) ** 0.5,
                                                        torch.tensor(1e-6, dtype=Datatype.JPRB, device=device))


        DARE_pr                         =           CC_VARS.D2FLDSTO_PRE[RC_Index, 1] * CM_NMLIST.D2RIVLEN[RC_Index,1] ** (-1)
        DARE_pr                         =           torch.maximum(  DARE_pr -  CC_VARS.D2FLDDPH_PRE[RC_Index, 1] * CM_NMLIST.D2RIVWTH[RC_Index,1]
                                                                    ,torch.tensor(1e-6, dtype=Datatype.JPRB, device=device) )       #   !! remove above river channel area
        DARE_im                         =           torch.maximum( (DARE * DARE_pr) ** 0.5,   torch.tensor(1e-6, dtype=Datatype.JPRB, device=device) )


        DOUT_pr                         =           CC_VARS.D2FLDOUT_PRE [RC_Index, 1]
        DOUT                            =           ((DOUT_pr + CC_NMLIST.PGRV * CC_NMLIST.DT * DARE_im * DSLP) *
                                                     (1 + CC_NMLIST.PGRV * CC_NMLIST.DT * CC_NMLIST.PMANFLD ** 2 *
                                                      torch.abs(DOUT_pr) * DFLW_im ** (-4 / 3) * DARE_im ** (-1))
                                                     ** (-1))

        Mask                            =           (DFLW_im > 1e-5) & (DARE > 1e-5)    #   !! replace small depth location with zero
        CC_VARS.D2FLDOUT[RC_Index, 1]   =           torch.where(Mask, DOUT, torch.tensor(0.0, dtype=Datatype.JPRB, device=device))

        DOUT                            =           CC_VARS.D2FLDOUT[RC_Index, 1]
        Mask                            =           (CC_VARS.D2FLDOUT[RC_Index, 1] * CC_VARS.D2RIVOUT[RC_Index, 1] > 0)   #   !! river and floodplain different direction
        CC_VARS.D2FLDOUT[RC_Index, 1]   =           torch.where(Mask, DOUT, torch.tensor(0.0, dtype=Datatype.JPRB, device=device))


    #   !=== river mouth flow ===
    RMF_Index                           =           torch.arange(CM_NMLIST.NSEQRIV + 1, CM_NMLIST.NSEQALL + 1, device=device)
    DSLP                                =           (CC_VARS.D2SFCELV[RMF_Index, 1] - CM_NMLIST.D2DWNELV[RMF_Index, 1]) * CC_NMLIST.PDSTMTH ** (-1)
    if CC_NMLIST.LSLOPEMOUTH:
        print("The 'DSLP = D2ELEVSLOPE' code in 122-th Line for cmf_calc_putflw_mod.py is needed to improved")


    DFLW                                =           CC_VARS.D2RIVDPH[RMF_Index, 1]
    DARE                                =           CM_NMLIST.D2RIVWTH[RMF_Index, 1] * DFLW
    DARE                                =           torch.maximum(DARE,     # !!  flow cross-section area (min value for stability)
                                                              torch.tensor(1e-10, dtype=Datatype.JPRB, device=device))


    DFLW_pr                             =           CC_VARS.D2RIVDPH_PRE[RMF_Index, 1]
    DFLW_im                             =           torch.maximum( (DFLW * DFLW_pr) ** 0.5,
                                                                   torch.tensor(1e-6, dtype=Datatype.JPRB, device=device) )     #   !! semi implicit flow depth


    DOUT_pr                             =           CC_VARS.D2RIVOUT_PRE[RMF_Index,1] * CM_NMLIST.D2RIVWTH[RMF_Index, 1] ** (-1)
    DOUT                                =           (CM_NMLIST.D2RIVWTH[RMF_Index, 1] * (DOUT_pr + CC_NMLIST.PGRV * CC_NMLIST.DT * DFLW_im * DSLP) *
                                                     (1 + CC_NMLIST.PGRV * CC_NMLIST.DT * CM_NMLIST.D2RIVMAN[RMF_Index, 1] ** 2 * torch.abs(DOUT_pr) *
                                                     DFLW_im ** (-7/3))** (-1))
    DVEL                                =           CC_VARS.D2RIVOUT[RMF_Index, 1] * DARE ** (-1)


    Mask                                =           (DFLW_im > 1e-5) & (DARE > 1e-5)    #   !! replace small depth location with zero
    CC_VARS.D2RIVOUT[RMF_Index, 1]      =           torch.where(Mask, DOUT, torch.tensor(0.0, dtype=Datatype.JPRB, device=device))
    CC_VARS.D2RIVVEL[RMF_Index, 1]      =           torch.where(Mask, DVEL, torch.tensor(0.0, dtype=Datatype.JPRB, device=device))

    # !=== floodplain mouth flow ===
    if CC_NMLIST.LFLDOUT:
        DFSTO                           =           CC_VARS.P2FLDSTO[RMF_Index,1]
        DSLP                            =           (CC_VARS.D2SFCELV[RMF_Index, 1] - CM_NMLIST.D2DWNELV[RMF_Index, 1]) * CC_NMLIST.PDSTMTH ** (-1)
        DSLP                            =           torch.maximum(-torch.tensor(0.005, dtype=Datatype.JPRB, device=device),
                                                    torch.minimum( torch.tensor(0.005, dtype=Datatype.JPRB, device=device),DSLP))  # !! set max&min [instead of using weir equation for efficiency]
        if CC_NMLIST.LSLOPEMOUTH:
            print("The 'DSLP = D2ELEVSLOPE' code in 154-th Line for cmf_calc_putflw_mod.py is needed to improved")

        DFLW                            =           CC_VARS.D2SFCELV[RMF_Index, 1] - CM_NMLIST.D2ELEVTN[RMF_Index, 1]
        DARE                            =           (torch.maximum
                                                     (DFSTO * CM_NMLIST.D2RIVLEN[RMF_Index, 1] * (-1) -
                                                            CC_VARS.D2FLDDPH[RMF_Index, 1] * CM_NMLIST.D2RIVWTH[RMF_Index, 1],  # !! remove above channel
                                                     torch.tensor(0, dtype=Datatype.JPRB, device=device)))


        DFLW_pr                         =           CC_VARS.D2SFCELV_PRE[RMF_Index, 1] - CM_NMLIST.D2ELEVTN[RMF_Index, 1]
        DFLW_im                         =           torch.maximum(
                                                        torch.maximum(DFLW * DFLW_pr, -torch.tensor(0, dtype=Datatype.JPRB, device=device)) ** 0.5,
                                                        torch.tensor(1e-6, dtype=Datatype.JPRB, device=device))

        DARE_pr                         =           torch.maximum(
                                                        CC_VARS.D2FLDSTO_PRE[RMF_Index, 1] * CM_NMLIST.D2RIVLEN[RMF_Index, 1] ** (-1) -
                                                        CC_VARS.D2FLDDPH_PRE[RMF_Index, 1] * CM_NMLIST.D2RIVWTH[RMF_Index, 1],
                                                        torch.tensor(1e-6, dtype=Datatype.JPRB, device=device))
        DARE_im                         =           torch.maximum((DARE * DARE_pr) ** 0.5 , torch.tensor(1e-6, dtype=Datatype.JPRB, device=device))


        DOUT_pr                         =           CC_VARS.D2FLDOUT_PRE [RMF_Index, 1]
        DOUT                            =           ((DOUT_pr + CC_NMLIST.PGRV * CC_NMLIST.DT * DARE_im * DSLP) *
                                                     (1 + CC_NMLIST.PGRV * CC_NMLIST.DT * CC_NMLIST.PMANFLD ** 2 *
                                                      torch.abs(DOUT_pr) * DFLW_im ** (-4 / 3) * DARE_im ** (-1))
                                                     ** (-1))

        Mask                            =           (DFLW_im > 1e-5) & (DARE > 1e-5)  # !! replace small depth location with zero
        CC_VARS.D2FLDOUT[RMF_Index, 1]  =           torch.where(Mask, DOUT, torch.tensor(0.0, dtype=Datatype.JPRB, device=device))

        DOUT                            =           CC_VARS.D2FLDOUT[RMF_Index, 1]
        Mask                            =           (CC_VARS.D2FLDOUT[RMF_Index, 1] * CC_VARS.D2RIVOUT[RMF_Index, 1] > 0)   #   !! river and floodplain different direction
        CC_VARS.D2FLDOUT[RMF_Index, 1]  =           torch.where(Mask, DOUT, torch.tensor(0.0, dtype=Datatype.JPRB, device=device))


        RC_Index                        =           torch.arange(1, CM_NMLIST.NSEQRIV + 1, device=device)
        #   !! Storage change limitter to prevent sudden increase of "upstream" water level when backwardd flow (v423)
        DOUT                            =           torch.maximum((-CC_VARS.D2RIVOUT[RC_Index, 1] - CC_VARS.D2FLDOUT[RC_Index, 1]) * CC_NMLIST.DT,
                                                                  torch.tensor(1e-10, dtype=Datatype.JPRB, device=device))
        RATE                            =           torch.minimum(torch.tensor(0.05, dtype=Datatype.JPRB, device=device) * CC_VARS.D2STORGE[RC_Index, 1] / DOUT,
                                                                  torch.tensor(1, dtype=Datatype.JPRB, device=device))
        CC_VARS.D2RIVOUT[RC_Index, 1]  =           CC_VARS.D2RIVOUT[RC_Index, 1]  *  RATE
        CC_VARS.D2FLDOUT[RC_Index, 1]  =           CC_VARS.D2FLDOUT[RC_Index, 1]  *  RATE

    return CC_VARS
def CMF_CALC_INFLOW(CC_NMLIST, CM_NMLIST, CC_VARS , device, Datatype):
    """
    """
    # !*** 1. initialize & calculate P2STOOUT for normal cells
    P2RIVINF                    =           torch.zeros((CM_NMLIST.NSEQALL, 1), dtype=Datatype.JPRD, device=device)
    P2RIVINF                    =           Ftensor_2D(P2RIVINF, start_row=1, start_col=1)
    P2FLDINF                    =           torch.zeros((CM_NMLIST.NSEQALL, 1), dtype=Datatype.JPRD, device=device)
    P2FLDINF                    =           Ftensor_2D(P2FLDINF, start_row=1, start_col=1)
    P2PTHOUT                    =           torch.zeros((CM_NMLIST.NSEQALL, 1), dtype=Datatype.JPRD, device=device)
    P2PTHOUT                    =           Ftensor_2D(P2PTHOUT, start_row=1, start_col=1)
    P2STOOUT                    =           torch.zeros((CM_NMLIST.NSEQALL, 1), dtype=Datatype.JPRD, device=device)
    P2STOOUT                    =           Ftensor_2D(P2STOOUT, start_row=1, start_col=1)
    D2RATE                      =           torch.ones((CM_NMLIST.NSEQALL, 1), dtype=Datatype.JPRB, device=device)
    D2RATE                      =           Ftensor_2D(D2RATE, start_row=1, start_col=1)

    # !! for normal cells
    RC_Index                    =           torch.arange(1, CM_NMLIST.NSEQRIV + 1, device=device)
    JSEQ                        =           CM_NMLIST.I1NEXT[RC_Index]                      #   ! next cell's pixel
    OUT_R1                      =           torch.maximum( CC_VARS.D2RIVOUT[RC_Index, 1],
                                                          torch.tensor(0.0, dtype=Datatype.JPRB, device=device))
    OUT_R2                      =           torch.maximum(-CC_VARS.D2RIVOUT[RC_Index, 1],
                                                          torch.tensor(0.0, dtype=Datatype.JPRB, device=device))
    OUT_F1                      =           torch.maximum( CC_VARS.D2FLDOUT[RC_Index, 1],
                                                          torch.tensor(0.0, dtype=Datatype.JPRB, device=device))
    OUT_F2                      =           torch.maximum(-CC_VARS.D2FLDOUT[RC_Index, 1],
                                                          torch.tensor(0.0, dtype=Datatype.JPRB, device=device))
    DIUP                        =           (OUT_R1 + OUT_F1) * CC_NMLIST.DT
    DIDW                        =           (OUT_R2 + OUT_F2) * CC_NMLIST.DT

    # P2STOOUT[RC_Index,1]        =           P2STOOUT[RC_Index,1]    +   DIUP
    P2STOOUT.raw().index_add_              (0, RC_Index-1,  DIUP.unsqueeze(1))
    P2STOOUT.raw().index_add_              (0, JSEQ-1,      DIDW.unsqueeze(1))

    # !! for river mouth grids ------------
    RMF_Index                   =           torch.arange(CM_NMLIST.NSEQRIV + 1, CM_NMLIST.NSEQALL + 1, device=device)
    OUT_R1                      =           torch.maximum(CC_VARS.D2RIVOUT[RMF_Index, 1],
                                                          torch.tensor(0.0, dtype=Datatype.JPRB, device=device))
    OUT_F1                      =           torch.maximum(CC_VARS.D2FLDOUT[RMF_Index, 1],
                                                          torch.tensor(0.0, dtype=Datatype.JPRB, device=device))
    P2STOOUT[RMF_Index,1]       =           P2STOOUT[RMF_Index,1]   +   OUT_R1 * CC_NMLIST.DT   +   OUT_F1 * CC_NMLIST.DT
    #-------------------------------------------------------------------------------------------------------------------
    # !! for bifurcation channels ------------
    if CC_NMLIST.LPTHOUT:
        NT_Index                =          torch.arange(1, CM_NMLIST.NPTHOUT + 1, device=device)
        ISEQP                   =          CM_NMLIST.PTH_UPST[NT_Index]
        JSEQP                   =          CM_NMLIST.PTH_DOWN[NT_Index]
        #  !! Avoid calculation outside of domain
        ID_M                    =           ((ISEQP > 0) & (JSEQP > 0) &
                                            (CM_NMLIST.I2MASK[ISEQP, 1] <= 0) & (CM_NMLIST.I2MASK[JSEQP, 1] <= 0)).nonzero(as_tuple=True)[0]
        OUT_R1                  =           torch.maximum( CC_VARS.D1PTHFLWSUM[NT_Index[ID_M]], torch.tensor(0.0, dtype=Datatype.JPRB, device=device))
        OUT_R2                  =           torch.maximum(-CC_VARS.D1PTHFLWSUM[NT_Index[ID_M]], torch.tensor(0.0, dtype=Datatype.JPRB, device=device))
        DIUP                    =           (OUT_R1) * CC_NMLIST.DT
        DIDW                    =           (OUT_R2) * CC_NMLIST.DT
        P2STOOUT.raw().index_add_              (0, ISEQP[ID_M]-1,  DIUP.unsqueeze(1))
        P2STOOUT.raw().index_add_              (0, JSEQP[ID_M]-1,  DIDW.unsqueeze(1))
    #-------------------------------------------------------------------------------------------------------------------
    # !*** 2. modify outflow
    NQ_Index                    =          torch.arange(1, CM_NMLIST.NSEQALL + 1, device=device)
    PT_P_IX                     =          (P2STOOUT.raw() > 1.e-8).nonzero(as_tuple=True)[0]
    PT_P_ID                     =          NQ_Index[PT_P_IX]

    D2RATE[PT_P_ID, 1]          =          torch.minimum(
                                                ((CC_VARS.P2RIVSTO[PT_P_ID, 1]) + CC_VARS.P2FLDSTO[PT_P_ID, 1]) /
                                                P2STOOUT[PT_P_ID, 1],
                                                torch.tensor(1, dtype=Datatype.JPRD, device=device))
    # print(f'the index for D2RATE != 1:   {D2RATE.where(D2RATE.raw()!=1)[0]}')
    # !! normal pixels------
    RC_Index                    =           torch.arange(1, CM_NMLIST.NSEQRIV + 1, device=device)
    JSEQ                        =           CM_NMLIST.I1NEXT[RC_Index]                      #   ! next cell's pixel
    DR_P_                       =           (CC_VARS.D2RIVOUT[RC_Index,1] >= 0).nonzero(as_tuple=True)[0]
    DR_P_ID                     =           RC_Index[DR_P_]
    DR_N_ID                     =          (RC_Index)[~torch.isin((RC_Index), DR_P_ID)]
    CC_VARS.D2RIVOUT[DR_P_ID,1] =           CC_VARS.D2RIVOUT[DR_P_ID,1]  *  D2RATE[DR_P_ID,1]
    CC_VARS.D2FLDOUT[DR_P_ID,1] =           CC_VARS.D2FLDOUT[DR_P_ID,1]  *  D2RATE[DR_P_ID,1]
    CC_VARS.D2RIVOUT[DR_N_ID,1] =           CC_VARS.D2RIVOUT[DR_N_ID,1]  *  D2RATE[CM_NMLIST.I1NEXT[DR_N_ID],1]
    CC_VARS.D2FLDOUT[DR_N_ID,1] =           CC_VARS.D2FLDOUT[DR_N_ID,1]  *  D2RATE[CM_NMLIST.I1NEXT[DR_N_ID],1]


    P2RIVINF.raw().index_add_               (0, JSEQ-1, CC_VARS.D2RIVOUT[RC_Index,1].unsqueeze(1))    # !! total inflow to a grid (from upstream)

    P2FLDINF.raw().index_add_               (0, JSEQ-1, CC_VARS.D2FLDOUT[RC_Index,1].unsqueeze(1))

    # !! river mouth-----------------
    RMF_Index                   =           torch.arange(CM_NMLIST.NSEQRIV + 1, CM_NMLIST.NSEQALL + 1, device=device)
    CC_VARS.D2RIVOUT[RMF_Index,1]   =       CC_VARS.D2RIVOUT[RMF_Index,1]   *   D2RATE[RMF_Index,1]
    CC_VARS.D2FLDOUT[RMF_Index,1]   =       CC_VARS.D2FLDOUT[RMF_Index,1]   *   D2RATE[RMF_Index,1]

    # !! bifurcation channels --------
    if CC_NMLIST.LPTHOUT:
        NT_Index                    =       torch.arange(1, CM_NMLIST.NPTHOUT + 1, device=device)
        ISEQP                       =       CM_NMLIST.PTH_UPST[NT_Index]
        JSEQP                       =       CM_NMLIST.PTH_DOWN[NT_Index]
        #  !! Avoid calculation outside of domain
        ID_M                        =       ((ISEQP > 0) & (JSEQP > 0) &
                                            (CM_NMLIST.I2MASK[ISEQP, 1] <= 0) & (CM_NMLIST.I2MASK[JSEQP, 1] <= 0)).nonzero(as_tuple=True)[0]

        for ILEV in range(1, CM_NMLIST.NPTHLEV + 1):
            DW_P_M_ID                                           =       (CC_VARS.D1PTHFLW[NT_Index[ID_M], ILEV] >= 0).nonzero(as_tuple=True)[0]     # !! total outflow from each grid
            DW_N_M_ID                                           =       (CC_VARS.D1PTHFLW[NT_Index[ID_M], ILEV] <  0).nonzero(as_tuple=True)[0]
            CC_VARS.D1PTHFLW[NT_Index[ID_M][DW_P_M_ID], ILEV]   =       CC_VARS.D1PTHFLW[NT_Index[ID_M][DW_P_M_ID], ILEV] * D2RATE[ISEQP[ID_M][DW_P_M_ID],1]
            CC_VARS.D1PTHFLW[NT_Index[ID_M][DW_N_M_ID], ILEV]   =       CC_VARS.D1PTHFLW[NT_Index[ID_M][DW_N_M_ID], ILEV] * D2RATE[JSEQP[ID_M][DW_N_M_ID],1]


        DS_P_M_ID                                               =       (CC_VARS.D1PTHFLWSUM[NT_Index[ID_M]] >= 0).nonzero(as_tuple=True)[0]        # !! total outflow from each grid
        DS_N_M_ID                                               =       (CC_VARS.D1PTHFLWSUM[NT_Index[ID_M]] <  0).nonzero(as_tuple=True)[0]
        CC_VARS.D1PTHFLWSUM[NT_Index[ID_M][DS_P_M_ID]]          =       CC_VARS.D1PTHFLWSUM[NT_Index[ID_M][DS_P_M_ID]] * D2RATE[ISEQP[ID_M][DS_P_M_ID],1]
        CC_VARS.D1PTHFLWSUM[NT_Index[ID_M][DS_N_M_ID]]          =       CC_VARS.D1PTHFLWSUM[NT_Index[ID_M][DS_N_M_ID]] * D2RATE[JSEQP[ID_M][DS_N_M_ID],1]

        P2PTHOUT.raw().index_add_(0, ISEQP[ID_M] - 1,   CC_VARS.D1PTHFLWSUM[NT_Index[ID_M]].unsqueeze(1))
        P2PTHOUT.raw().index_add_(0, JSEQP[ID_M] - 1,  -CC_VARS.D1PTHFLWSUM[NT_Index[ID_M]].unsqueeze(1))


        CC_VARS.D2RIVINF[:CM_NMLIST.NSEQALL+1, 1]               =       P2RIVINF[:CM_NMLIST.NSEQALL+1, 1]   #   !! needed for SinglePrecisionMode
        CC_VARS.D2FLDINF[:CM_NMLIST.NSEQALL+1, 1]               =       P2FLDINF[:CM_NMLIST.NSEQALL+1, 1]
        CC_VARS.D2PTHOUT[:CM_NMLIST.NSEQALL+1, 1]               =       P2PTHOUT[:CM_NMLIST.NSEQALL+1, 1]

    return  CC_VARS