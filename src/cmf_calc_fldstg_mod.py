#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on  April  24  08:42 2025
@Author: Qingliang Li: liqingliang@ccsfu.edu.cn （Email）
@Co-author1: Zhongwang Wei:  weizhw6@mail.sysu.edu.cn（Email）
@Co-author2: Kaixuan Cai:  caikx22@mails.jlu.edu.cn（Email）
@purpose:  calculate river and floodplain staging (python)
Licensed under the Apache License, Version 2.0.

* CONTAINS:
! -- CMF_CALC_FLDSTG_DEF  !! default flood stage calculation
! -- CMF_OPT_FLDSTG_ES    !! optimized code for vector processor (such as Earth Simulator), activated using LSTG_ES=.TRUE. option).
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

def CMF_CALC_FLDSTG_DEF(CM_NMLIST,CC_NMLIST,CC_VARS,device, Datatype):
    DDPH_fil              =         torch.zeros((CM_NMLIST.NSEQALL), dtype=Datatype.JPRB, device=device)
    DDPH_fil              =         Ftensor_1D(DDPH_fil , start_index=1)
    DSTO_fil              =         torch.zeros((CM_NMLIST.NSEQALL), dtype=Datatype.JPRB, device=device)
    DSTO_fil              =         Ftensor_1D(DSTO_fil , start_index=1)
    DWTH_fil              =         torch.zeros((CM_NMLIST.NSEQALL), dtype=Datatype.JPRB, device=device)
    DWTH_fil              =         Ftensor_1D(DWTH_fil , start_index=1)
    DWTH_inc              =         torch.zeros((CM_NMLIST.NSEQALL), dtype=Datatype.JPRB, device=device)
    DWTH_inc              =         Ftensor_1D(DWTH_inc , start_index=1)
    DWTH_add              =         torch.zeros((CM_NMLIST.NSEQALL), dtype=Datatype.JPRB, device=device)
    DWTH_add              =         Ftensor_1D(DWTH_add, start_index=1)

    #   Estimate water depth and flood extent from water storage
    #        Solution for Equations (1) and (2) in [Yamazaki et al. 2011 WRR].
    FD_Index                    =        torch.arange(1, CM_NMLIST.NSEQALL + 1, device=device)
    #   DSTOALL: For each grid cell, the river water and floodplain water are first combined to obtain DSTOALL (total water storage).
    PSTOALL                     =       Ftensor_1D(CC_VARS.P2RIVSTO[FD_Index, 1] + CC_VARS.P2FLDSTO[FD_Index, 1], start_index=1)
    DSTOALL                     =       Ftensor_1D(PSTOALL.raw().to(dtype=Datatype.JPRB,device=device), start_index=1)
    CC_VARS.P0GLBSTOPRE2        =       sum((PSTOALL[FD_Index].view(-1).tolist()))

    #   [Case 1] When the water storage exceeds the maximum river channel capacity, Flooding Occurs.
    DD_P_M_ID_                  =       (PSTOALL  >  CM_NMLIST.D2RIVSTOMAX[FD_Index, 1]).nonzero(as_tuple=True)[0]
    DD_N_M_ID_                  =       (PSTOALL  <= CM_NMLIST.D2RIVSTOMAX[FD_Index, 1]).nonzero(as_tuple=True)[0]
    DD_P_M_ID                   =       FD_Index[DD_P_M_ID_]
    DD_N_M_ID                   =       FD_Index[DD_N_M_ID_]
    if torch.any(DD_P_M_ID):
        DSTO_fil [DD_P_M_ID]    =       CM_NMLIST.D2RIVSTOMAX  [DD_P_M_ID, 1]
        DWTH_fil [DD_P_M_ID]    =       CM_NMLIST.D2RIVWTH     [DD_P_M_ID, 1]
        #This reflects the influence of terrain slope — the flatter the terrain, the more easily the floodwater spreads.
        DWTH_inc [DD_P_M_ID]    =       CM_NMLIST.D2GRAREA[DD_P_M_ID, 1] / CM_NMLIST.D2RIVLEN[DD_P_M_ID, 1] * CM_NMLIST.DFRCINC
    #   The excess water is allocated to the floodplain
        for ISEQ in DD_P_M_ID:
            I = 1
            #   D2FLDSTOMAX(...) defines the maximum storage capacity at each hierarchical level, representing floodplain
            #   expansion across riverbanks with a fixed width increment per layer as follows
            while PSTOALL[ISEQ]          >       CM_NMLIST.D2FLDSTOMAX[ISEQ,1, I]  and     I <= CC_NMLIST.NLFP:
                DSTO_fil[ISEQ]           =       CM_NMLIST.D2FLDSTOMAX[ISEQ,1, I]
                DWTH_fil[ISEQ]           =       DWTH_fil[ISEQ].item() + DWTH_inc[ISEQ].item()
                DDPH_fil[ISEQ]           =       DDPH_fil[ISEQ].item() + CM_NMLIST.D2FLDGRD[ISEQ, 1, I].item() * DWTH_inc[ISEQ].item()
                I += 1
                #   If none of the floodplain levels can accommodate the excess water, overflow occurs
                if I > CC_NMLIST.NLFP:
                    break
            if  I> CC_NMLIST.NLFP:
                DSTO_add                 =       DSTOALL[ISEQ].item()   -   DSTO_fil[ISEQ].item()
                DWTH_add[ISEQ]           =       torch.tensor(0,dtype=Datatype.JPRB,device=device)
                CC_VARS.D2FLDDPH[ISEQ,1] =        DDPH_fil [ISEQ].item() + DSTO_add  / DWTH_fil [ISEQ].item() / CM_NMLIST.D2RIVLEN[ISEQ,1].item()
            else:
                DSTO_add                 =       DSTOALL[ISEQ].item()   -   DSTO_fil[ISEQ].item()
                DWTH_add [ISEQ]           =      (-DWTH_fil [ISEQ].item()   +
                                                 np.sqrt(DWTH_fil [ISEQ].item()**2 +2 * DSTO_add  / CM_NMLIST.D2RIVLEN[ISEQ,1].item() / CM_NMLIST.D2FLDGRD[ISEQ, 1, I].item())
                                                 )
                CC_VARS.D2FLDDPH[ISEQ,1]=       DDPH_fil [ISEQ].item() + CM_NMLIST.D2FLDGRD[ISEQ, 1, I].item() * DWTH_add[ISEQ].item()

        #   Update river water depth and storage
        #   P2RIVSTO represents the sum of the full river channel storage and the floodplain water volume converted
        #   into an equivalent river storage."
        CC_VARS.P2RIVSTO[DD_P_M_ID, 1]       =          (CM_NMLIST.D2RIVSTOMAX[DD_P_M_ID, 1] +
                                                        CM_NMLIST.D2RIVLEN[DD_P_M_ID, 1] * CM_NMLIST.D2RIVWTH[DD_P_M_ID, 1] * CC_VARS.D2FLDDPH[DD_P_M_ID, 1])

        CC_VARS.P2RIVSTO[DD_P_M_ID, 1]       =          torch.minimum(CC_VARS.P2RIVSTO[DD_P_M_ID, 1], PSTOALL[DD_P_M_ID])

        CC_VARS.D2RIVDPH[DD_P_M_ID, 1]       =          (CC_VARS.P2RIVSTO[DD_P_M_ID, 1] / CM_NMLIST.D2RIVLEN[DD_P_M_ID, 1]  / CM_NMLIST.D2RIVWTH[DD_P_M_ID, 1])

        CC_VARS.P2FLDSTO[DD_P_M_ID, 1]       =          PSTOALL [DD_P_M_ID] - CC_VARS.P2RIVSTO[DD_P_M_ID, 1]

        CC_VARS.P2FLDSTO[DD_P_M_ID, 1]       =          torch.maximum(CC_VARS.P2FLDSTO[DD_P_M_ID, 1], torch.tensor(0,dtype=Datatype.JPRD,device=device))
        #   Computation of floodplain area and its proportion
        CC_VARS.D2FLDFRC[DD_P_M_ID, 1]       =          ((-CM_NMLIST.D2RIVWTH[DD_P_M_ID, 1] + DWTH_fil[DD_P_M_ID] + DWTH_add[DD_P_M_ID]) /
                                                        (DWTH_inc[DD_P_M_ID] * CC_NMLIST.NLFP) )
        CC_VARS.D2FLDFRC[DD_P_M_ID, 1]       =          torch.clamp(CC_VARS.D2FLDFRC[DD_P_M_ID, 1], min=0, max=1)
        CC_VARS.D2FLDARE[DD_P_M_ID, 1]       =          (CM_NMLIST.D2GRAREA[DD_P_M_ID, 1] * CC_VARS.D2FLDFRC[DD_P_M_ID, 1])

    # else:
    #
    #     CC_VARS.P2RIVSTO[DD_N_M_ID, 1]       =       DSTOALL[DD_N_M_ID]
    #     CC_VARS.D2RIVDPH[DD_N_M_ID, 1]       =       DSTOALL[DD_N_M_ID] * (CM_NMLIST.D2RIVLEN[DD_N_M_ID, 1] ** -1) * (CM_NMLIST.D2RIVWTH[DD_N_M_ID, 1] ** -1)
    #     CC_VARS.D2RIVDPH[DD_N_M_ID, 1]       =       torch.maximum(CC_VARS.D2RIVDPH[DD_N_M_ID, 1], torch.tensor(0,dtype=Datatype.JPRB,device=device))
    #     CC_VARS.P2FLDSTO[DD_N_M_ID, 1]       =       torch.tensor(0,dtype=Datatype.JPRD,device=device)
    #     CC_VARS.D2FLDDPH[DD_N_M_ID, 1]       =       torch.tensor(0,dtype=Datatype.JPRB,device=device)
    #     CC_VARS.D2FLDFRC[DD_N_M_ID, 1]       =       torch.tensor(0,dtype=Datatype.JPRB,device=device)
    #     CC_VARS.D2FLDARE[DD_N_M_ID, 1]       =       torch.tensor(0,dtype=Datatype.JPRB,device=device)

    CC_VARS.P2RIVSTO[DD_N_M_ID, 1]      =       PSTOALL[DD_N_M_ID]
    CC_VARS.D2RIVDPH[DD_N_M_ID, 1]      =       DSTOALL[DD_N_M_ID] / CM_NMLIST.D2RIVLEN[DD_N_M_ID, 1] / CM_NMLIST.D2RIVWTH[DD_N_M_ID, 1]
    CC_VARS.D2RIVDPH[DD_N_M_ID, 1]      =       torch.maximum(CC_VARS.D2RIVDPH[DD_N_M_ID, 1], torch.tensor(0,dtype=Datatype.JPRB,device=device))
    CC_VARS.P2FLDSTO[DD_N_M_ID, 1]      =       torch.tensor(0,dtype=Datatype.JPRD,device=device)
    CC_VARS.D2FLDDPH[DD_N_M_ID, 1]      =       torch.tensor(0,dtype=Datatype.JPRB,device=device)
    CC_VARS.D2FLDFRC[DD_N_M_ID, 1]      =       torch.tensor(0,dtype=Datatype.JPRB,device=device)
    CC_VARS.D2FLDARE[DD_N_M_ID, 1]      =       torch.tensor(0,dtype=Datatype.JPRB,device=device)



    CC_VARS.D2SFCELV[FD_Index,1]            =           CM_NMLIST.D2RIVELV[FD_Index,1] + CC_VARS.D2RIVDPH[FD_Index,1]
    CC_VARS.D2STORGE[FD_Index, 1]           =           (CC_VARS.P2RIVSTO[FD_Index, 1] + CC_VARS.P2FLDSTO[FD_Index, 1]).to(Datatype.JPRB)
    # CC_VARS.P0GLBSTOPRE2                    =           DSTOALL.raw().sum()
    CC_VARS.P0GLBSTOPRE2                    =           sum(DSTOALL.raw().tolist())
    # CC_VARS.P0GLBSTONEW2                    =           P0GLBSTONEW2_.sum()
    CC_VARS.P0GLBSTONEW2                    =           sum(((CC_VARS.P2RIVSTO[FD_Index,1].view(-1)) + (CC_VARS.P2FLDSTO[FD_Index,1].view(-1))).tolist())
    # CC_VARS.P0GLBRIVSTO                     =           CC_VARS.P2RIVSTO.raw().sum()
    CC_VARS.P0GLBRIVSTO                     =           sum(CC_VARS.P2RIVSTO.raw().view(-1).tolist())
    # CC_VARS.P0GLBFLDSTO                     =           CC_VARS.P2FLDSTO.raw().sum()
    CC_VARS.P0GLBFLDSTO                     =           sum(CC_VARS.P2FLDSTO.raw().view(-1).tolist())
    # CC_VARS.P0GLBFLDARE                     =           CC_VARS.D2FLDARE.raw().sum()
    CC_VARS.P0GLBFLDARE                     =           sum(CC_VARS.D2FLDARE.raw().view(-1).tolist())

    print(f' ====  CMF_CALC_FLDSTG_DEF====')
    print(f'P0GLBSTOPRE2:   {CC_VARS.P0GLBSTOPRE2:,.15f}')
    print(f'P0GLBSTONEW2:   {CC_VARS.P0GLBSTONEW2:,.15f}')
    print(f'P0GLBRIVSTO:    {CC_VARS.P0GLBRIVSTO:,.15f}')
    print(f'P0GLBFLDSTO:    {CC_VARS.P0GLBFLDSTO:,.15f}')
    print(f'P0GLBFLDARE:    {CC_VARS.P0GLBFLDARE:,.15f}')


    return CM_NMLIST ,CC_VARS