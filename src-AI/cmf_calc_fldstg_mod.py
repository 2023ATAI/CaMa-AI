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
    DSTO_fil              =         torch.zeros((CM_NMLIST.NSEQALL), dtype=Datatype.JPRB, device=device)
    DWTH_fil              =         torch.zeros((CM_NMLIST.NSEQALL), dtype=Datatype.JPRB, device=device)
    DWTH_inc              =         torch.zeros((CM_NMLIST.NSEQALL), dtype=Datatype.JPRB, device=device)
    DWTH_add              =         torch.zeros((CM_NMLIST.NSEQALL), dtype=Datatype.JPRB, device=device)

    #   Estimate water depth and flood extent from water storage
    #        Solution for Equations (1) and (2) in [Yamazaki et al. 2011 WRR].
    FD_Index                    =        torch.arange(0, CM_NMLIST.NSEQALL, device=device)
    #   DSTOALL: For each grid cell, the river water and floodplain water are first combined to obtain DSTOALL (total water storage).
    PSTOALL = CC_VARS.P2RIVSTO[FD_Index, 0] + CC_VARS.P2FLDSTO[FD_Index, 0]
    DSTOALL = PSTOALL.to(dtype=Datatype.JPRB, device=device)
    CC_VARS.P0GLBSTOPRE2        =       sum((PSTOALL[FD_Index].view(-1).tolist()))

    #   [Case 1] When the water storage exceeds the maximum river channel capacity, Flooding Occurs.
    DD_P_M_ID_ = (PSTOALL > CM_NMLIST.D2RIVSTOMAX[FD_Index, 0]).nonzero(as_tuple=True)[0]
    DD_N_M_ID_ = (PSTOALL <= CM_NMLIST.D2RIVSTOMAX[FD_Index, 0]).nonzero(as_tuple=True)[0]
    DD_P_M_ID = FD_Index[DD_P_M_ID_]
    DD_N_M_ID = FD_Index[DD_N_M_ID_]
    if torch.any(DD_P_M_ID):
        DSTO_fil[DD_P_M_ID] = CM_NMLIST.D2RIVSTOMAX[DD_P_M_ID, 0]
        DWTH_fil[DD_P_M_ID] = CM_NMLIST.D2RIVWTH[DD_P_M_ID, 0]
        # This reflects the influence of terrain slope — the flatter the terrain, the more easily the floodwater spreads.
        DWTH_inc[DD_P_M_ID] = CM_NMLIST.D2GRAREA[DD_P_M_ID, 0] / CM_NMLIST.D2RIVLEN[DD_P_M_ID, 0] * CM_NMLIST.DFRCINC
        #   The excess water is allocated to the floodplain
        for ISEQ in DD_P_M_ID:
            I = 0
            #   D2FLDSTOMAX(...) defines the maximum storage capacity at each hierarchical level, representing floodplain
            #   expansion across riverbanks with a fixed width increment per layer as follows
            while PSTOALL[ISEQ] > CM_NMLIST.D2FLDSTOMAX[ISEQ, 0, I] and I <= CC_NMLIST.NLFP:
                DSTO_fil[ISEQ] = CM_NMLIST.D2FLDSTOMAX[ISEQ, 0, I]
                DWTH_fil[ISEQ] = DWTH_fil[ISEQ].item() + DWTH_inc[ISEQ].item()
                DDPH_fil[ISEQ] = DDPH_fil[ISEQ].item() + CM_NMLIST.D2FLDGRD[ISEQ, 0, I].item() * DWTH_inc[ISEQ].item()
                I += 1
                #   If none of the floodplain levels can accommodate the excess water, overflow occurs
                if I > CC_NMLIST.NLFP - 1:
                    break
            if I > CC_NMLIST.NLFP - 1:
                DSTO_add = DSTOALL[ISEQ].item() - DSTO_fil[ISEQ].item()
                DWTH_add[ISEQ] = torch.tensor(0, dtype=Datatype.JPRB, device=device)
                CC_VARS.D2FLDDPH[ISEQ, 0] = DDPH_fil[ISEQ].item() + DSTO_add / DWTH_fil[ISEQ].item() / \
                                            CM_NMLIST.D2RIVLEN[ISEQ, 0].item()
            else:
                DSTO_add = DSTOALL[ISEQ].item() - DSTO_fil[ISEQ].item()
                DWTH_add[ISEQ] = (-DWTH_fil[ISEQ].item() +
                                  np.sqrt(
                                      DWTH_fil[ISEQ].item() ** 2 + 2 * DSTO_add / CM_NMLIST.D2RIVLEN[ISEQ, 0].item() /
                                      CM_NMLIST.D2FLDGRD[ISEQ, 0, I].item())
                                  )
                CC_VARS.D2FLDDPH[ISEQ, 0] = DDPH_fil[ISEQ].item() + CM_NMLIST.D2FLDGRD[ISEQ, 0, I].item() * DWTH_add[
                    ISEQ].item()

        #   Update river water depth and storage
        #   P2RIVSTO represents the sum of the full river channel storage and the floodplain water volume converted
        #   into an equivalent river storage."
        CC_VARS.P2RIVSTO[DD_P_M_ID, 0] = (CM_NMLIST.D2RIVSTOMAX[DD_P_M_ID, 0] +
                                          CM_NMLIST.D2RIVLEN[DD_P_M_ID, 0] * CM_NMLIST.D2RIVWTH[DD_P_M_ID, 0] *
                                          CC_VARS.D2FLDDPH[DD_P_M_ID, 0])

        CC_VARS.P2RIVSTO[DD_P_M_ID, 0] = torch.minimum(CC_VARS.P2RIVSTO[DD_P_M_ID, 0], PSTOALL[DD_P_M_ID])

        CC_VARS.D2RIVDPH[DD_P_M_ID, 0] = (
                    CC_VARS.P2RIVSTO[DD_P_M_ID, 0] / CM_NMLIST.D2RIVLEN[DD_P_M_ID, 0] / CM_NMLIST.D2RIVWTH[
                DD_P_M_ID, 0])

        CC_VARS.P2FLDSTO[DD_P_M_ID, 0] = PSTOALL[DD_P_M_ID] - CC_VARS.P2RIVSTO[DD_P_M_ID, 0]

        CC_VARS.P2FLDSTO[DD_P_M_ID, 0] = torch.maximum(CC_VARS.P2FLDSTO[DD_P_M_ID, 0],
                                                       torch.tensor(0, dtype=Datatype.JPRD, device=device))
        #   Computation of floodplain area and its proportion
        CC_VARS.D2FLDFRC[DD_P_M_ID, 0] = (
                    (-CM_NMLIST.D2RIVWTH[DD_P_M_ID, 0] + DWTH_fil[DD_P_M_ID] + DWTH_add[DD_P_M_ID]) /
                    (DWTH_inc[DD_P_M_ID] * CC_NMLIST.NLFP))
        CC_VARS.D2FLDFRC[DD_P_M_ID, 0] = torch.clamp(CC_VARS.D2FLDFRC[DD_P_M_ID, 0], min=0, max=1)
        CC_VARS.D2FLDARE[DD_P_M_ID, 0] = (CM_NMLIST.D2GRAREA[DD_P_M_ID, 0] * CC_VARS.D2FLDFRC[DD_P_M_ID, 0])


    CC_VARS.P2RIVSTO[DD_N_M_ID, 0] = PSTOALL[DD_N_M_ID]
    CC_VARS.D2RIVDPH[DD_N_M_ID, 0] = DSTOALL[DD_N_M_ID] / CM_NMLIST.D2RIVLEN[DD_N_M_ID, 0] / CM_NMLIST.D2RIVWTH[
        DD_N_M_ID, 0]
    CC_VARS.D2RIVDPH[DD_N_M_ID, 0] = torch.maximum(CC_VARS.D2RIVDPH[DD_N_M_ID, 0],
                                                   torch.tensor(0, dtype=Datatype.JPRB, device=device))
    CC_VARS.P2FLDSTO[DD_N_M_ID, 0] = torch.tensor(0, dtype=Datatype.JPRD, device=device)
    CC_VARS.D2FLDDPH[DD_N_M_ID, 0] = torch.tensor(0, dtype=Datatype.JPRB, device=device)
    CC_VARS.D2FLDFRC[DD_N_M_ID, 0] = torch.tensor(0, dtype=Datatype.JPRB, device=device)
    CC_VARS.D2FLDARE[DD_N_M_ID, 0] = torch.tensor(0, dtype=Datatype.JPRB, device=device)

    CC_VARS.D2SFCELV[FD_Index, 0] = CM_NMLIST.D2RIVELV[FD_Index, 0] + CC_VARS.D2RIVDPH[FD_Index, 0]
    CC_VARS.D2STORGE[FD_Index, 0] = (CC_VARS.P2RIVSTO[FD_Index, 0] + CC_VARS.P2FLDSTO[FD_Index, 0]).to(Datatype.JPRB)
    # CC_VARS.P0GLBSTOPRE2                    =           DSTOALL.raw().sum()
    CC_VARS.P0GLBSTOPRE2 = sum(DSTOALL.tolist())
    # CC_VARS.P0GLBSTONEW2                    =           P0GLBSTONEW2_.sum()
    CC_VARS.P0GLBSTONEW2 = sum(
        ((CC_VARS.P2RIVSTO[FD_Index, 0].view(-1)) + (CC_VARS.P2FLDSTO[FD_Index, 0].view(-1))).tolist())
    # CC_VARS.P0GLBRIVSTO                     =           CC_VARS.P2RIVSTO.raw().sum()
    CC_VARS.P0GLBRIVSTO = sum(CC_VARS.P2RIVSTO.view(-1).tolist())
    # CC_VARS.P0GLBFLDSTO                     =           CC_VARS.P2FLDSTO.raw().sum()
    CC_VARS.P0GLBFLDSTO = sum(CC_VARS.P2FLDSTO.view(-1).tolist())
    # CC_VARS.P0GLBFLDARE                     =           CC_VARS.D2FLDARE.raw().sum()
    CC_VARS.P0GLBFLDARE = sum(CC_VARS.D2FLDARE.view(-1).tolist())

    # print(f' ====  CMF_CALC_FLDSTG_DEF====')
    # print(f'P0GLBSTOPRE2:   {CC_VARS.P0GLBSTOPRE2:,.15f}')
    # print(f'P0GLBSTONEW2:   {CC_VARS.P0GLBSTONEW2:,.15f}')
    # print(f'P0GLBRIVSTO:    {CC_VARS.P0GLBRIVSTO:,.15f}')
    # print(f'P0GLBFLDSTO:    {CC_VARS.P0GLBFLDSTO:,.15f}')
    # print(f'P0GLBFLDARE:    {CC_VARS.P0GLBFLDARE:,.15f}')

    return CM_NMLIST, CC_VARS