#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on  April  24  08:42 2025
@Author: Qingliang Li: liqingliang@ccsfu.edu.cn （Email）
@Co-author1: Zhongwang Wei:  weizhw6@mail.sysu.edu.cn（Email）
@Co-author2: Kaixuan Cai:  caikx22@mails.jlu.edu.cn（Email）
@purpose:  subroutine for bifurcation channel flow (python)
Licensed under the Apache License, Version 2.0.

* CONTAINS:
! -- CMF_CALC_PTHOUT
"""
import  os

import numpy as np
import torch
from fortran_tensor_3D import Ftensor_3D
from fortran_tensor_2D import Ftensor_2D
from fortran_tensor_1D import Ftensor_1D

os.environ['PYTHONWARNINGS']='ignore::FutureWarning'
os.environ['PYTHONWARNINGS']='ignore::RuntimeWarning'


def CMF_CALC_PTHOUT(CC_NMLIST, CM_NMLIST, CC_VARS , device, Datatype):
    # ------------------------------------------------------------------------------------------------------------------
    CC_VARS.D2SFCELV_PRE[:,:]       =           CM_NMLIST.D2RIVELV[:,:]   +   CC_VARS.D2RIVDPH_PRE[:,:]

    CC_VARS.D1PTHFLW[:,:]           =           torch.tensor(0,dtype=Datatype.JPRB,device=device)

    RC_Index                        =           torch.arange(1, CM_NMLIST.NPTHOUT + 1, device=device)
    ISEQP                           =           CM_NMLIST.PTH_UPST[RC_Index]
    JSEQP                           =           CM_NMLIST.PTH_DOWN[RC_Index]
    # !! Avoid calculation outside of domain
    # !! I2MASK is for 1: kinemacit 2: dam  no bifurcation
    ID_M                             =            ( (ISEQP > 0)                     &       (JSEQP > 0) &
                                                  (CM_NMLIST.I2MASK[ISEQP, 1] <= 0) &       (CM_NMLIST.I2MASK[JSEQP, 1] <= 0)).nonzero(as_tuple=True)[0]
    DSLP                            =             (CC_VARS.D2SFCELV[ISEQP[ID_M], 1] - CC_VARS.D2SFCELV[JSEQP[ID_M], 1]) / CM_NMLIST.PTH_DST[RC_Index[ID_M]]

    DP_min                          =              torch.tensor(0.005, dtype=Datatype.JPRB, device=device)
    DP_max                          =             -torch.tensor(0.005, dtype=Datatype.JPRB, device=device)
    DSLP                            =              torch.maximum(DP_max, torch.minimum(DP_min,DSLP))    #!! v390 stabilization

    for ILEV in range(1, CM_NMLIST.NPTHLEV + 1):

        DFLW                        =               (torch.maximum(CC_VARS.D2SFCELV[ISEQP[ID_M], 1], CC_VARS.D2SFCELV[JSEQP[ID_M], 1])
                                                     - CM_NMLIST.PTH_ELV[RC_Index[ID_M], ILEV])
        DFLW                        =               torch.maximum(DFLW, torch.tensor(0, dtype=Datatype.JPRB, device=device))

        DFLW_pr                     =                (torch.maximum(CC_VARS.D2SFCELV_PRE[ISEQP[ID_M], 1], CC_VARS.D2SFCELV_PRE[JSEQP[ID_M], 1])
                                                     - CM_NMLIST.PTH_ELV[RC_Index[ID_M], ILEV])
        DFLW_pr                     =               torch.maximum(DFLW_pr , torch.tensor(0, dtype=Datatype.JPRB, device=device))

        DFLW_im                     =               (DFLW   *   DFLW_pr)   **  0.5         # !! semi implicit flow depth
        DFLW_im                     =               torch.maximum(DFLW_im, (DFLW * torch.tensor(0.01,dtype=Datatype.JPRB,device=device)) ** 0.5)

        DW_P_M_ID                   =               (DFLW_im >  1e-5).nonzero(as_tuple=True)[0]         #    !! local inertial equation, see [Bates et al., 2010, J.Hydrol.]
        DW_N_M_ID                   =               (DFLW_im <= 1e-5).nonzero(as_tuple=True)[0]

        DOUT_pr                     =               CC_VARS.D1PTHFLW_PRE[RC_Index[ID_M][DW_P_M_ID], ILEV] / CM_NMLIST.PTH_WTH[RC_Index[ID_M][DW_P_M_ID], ILEV]   #  !! outflow (t-1) [m2/s] (unit width)
        CC_VARS.D1PTHFLW[RC_Index[ID_M][DW_P_M_ID], ILEV]\
                                    =               (CM_NMLIST.PTH_WTH[RC_Index[ID_M][DW_P_M_ID], ILEV]  *
                                                     (DOUT_pr + CC_NMLIST.PGRV * CC_NMLIST.DT * DFLW_im[DW_P_M_ID] * DSLP [DW_P_M_ID]) /
                                                     (1 + CC_NMLIST.PGRV * CC_NMLIST.DT * CM_NMLIST.PTH_MAN[ILEV] ** 2 *
                                                      torch.abs(DOUT_pr) * DFLW_im[DW_P_M_ID] ** (-7/3)))

        CC_VARS.D1PTHFLW[RC_Index[ID_M][DW_N_M_ID], ILEV]   =   torch.tensor(0,dtype=Datatype.JPRB,device=device)

    CC_VARS.D1PTHFLWSUM[RC_Index[ID_M]]                     =   torch.tensor(0,dtype=Datatype.JPRB,device=device)

    for ILEV in range (1, CM_NMLIST.NPTHLEV+1):
        CC_VARS.D1PTHFLWSUM[RC_Index]                       =   CC_VARS.D1PTHFLWSUM[RC_Index] + CC_VARS.D1PTHFLW[RC_Index, ILEV]    #   !! bifurcation height layer summation


    #   !! Storage change limitter (to prevent sudden increase of upstream water level) (v423)
    RC_Index                                   =                torch.arange(1, CM_NMLIST.NPTHOUT + 1, device=device)
    ISEQP                                      =                CM_NMLIST.PTH_UPST[RC_Index]
    JSEQP                                      =                CM_NMLIST.PTH_DOWN[RC_Index]
    DW_P_M_ID                                  =                (CC_VARS.D1PTHFLWSUM[RC_Index] != 0).nonzero(as_tuple=True)[0]
    RATE                                       =                (torch.tensor(0.05,dtype=Datatype.JPRB,device=device) *
                                                                torch.minimum(CC_VARS.D2STORGE[ISEQP[DW_P_M_ID],1], CC_VARS.D2STORGE[JSEQP[DW_P_M_ID],1]) /
                                                                torch.abs(CC_VARS.D1PTHFLWSUM[RC_Index[DW_P_M_ID]] * CC_NMLIST.DT))      #   !! flow limit: 5% storage for stability
    RATE                                       =                torch.minimum(RATE, torch.tensor(1.0,dtype=Datatype.JPRB,device=device))
    CC_VARS.D1PTHFLW[RC_Index[DW_P_M_ID],1]    =                CC_VARS.D1PTHFLW[RC_Index[DW_P_M_ID],1] *  RATE
    CC_VARS.D1PTHFLWSUM[RC_Index[DW_P_M_ID]]   =                CC_VARS.D1PTHFLWSUM[RC_Index[DW_P_M_ID]] * RATE

    return CC_VARS