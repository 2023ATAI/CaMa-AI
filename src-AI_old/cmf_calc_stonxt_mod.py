#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on  April  24  08:42 2025
@Author: Qingliang Li: liqingliang@ccsfu.edu.cn （Email）
@Co-author1: Zhongwang Wei:  weizhw6@mail.sysu.edu.cn（Email）
@Co-author2: Kaixuan Cai:  caikx22@mails.jlu.edu.cn（Email）
@purpose:  calculate the storage in the next time step in FTCS diff. eq. (python)
Licensed under the Apache License, Version 2.0.

* CONTAINS:
! -- CMF_CALC_STONXT
"""
import  os
import torch


os.environ['PYTHONWARNINGS']='ignore::FutureWarning'
os.environ['PYTHONWARNINGS']='ignore::RuntimeWarning'

def CMF_CALC_STONXT(CC_NMLIST, CM_NMLIST, CC_VARS , device, Datatype):

    if CC_NMLIST.LGDWDLY:
        print("The 'CALC_GDWDLY' code in 37-th Line for cmf_calc_stonxt_mod.py is needed to improved")
        print("The 'CALC_GDWDLY' code in 38-th Line for cmf_calc_stonxt_mod.py is needed to improved")
    elif CC_NMLIST.LROSPLIT:
        # ! No ground water delay
        CC_VARS.D2GDWRTN[:CM_NMLIST.NSEQALL,0]          =       CC_VARS.D2ROFSUB[:CM_NMLIST.NSEQALL,0]
        CC_VARS.P2GDWSTO[:CM_NMLIST.NSEQALL,0]          =       torch.tensor(0,dtype=Datatype.JPRD,device=device)
    # ------------------------------------------------------------------------------------------------------------------
    NQ_Index                                            =           torch.arange(0, CM_NMLIST.NSEQALL,device=device)
    # CC_VARS.P0GLBSTOPRE                                 =           CC_VARS.P2RIVSTO.raw().sum()  +    CC_VARS.P2FLDSTO.raw().sum()
    # CC_VARS.P0GLBRIVINF                                 =           (((CC_VARS.D2RIVINF.raw()   *  CC_NMLIST.DT).sum()  +
    #                                                                 (CC_VARS.D2FLDINF.raw()   *  CC_NMLIST.DT).sum()    +
    #                                                                 (CC_VARS.D2PTHINF.raw()   *  CC_NMLIST.DT).sum()))
    # CC_VARS.P0GLBRIVOUT                                 =           (((CC_VARS.D2RIVOUT.raw() * CC_NMLIST.DT).sum()     +
    #                                                                 (CC_VARS.D2FLDOUT.raw() * CC_NMLIST.DT).sum()       +
    #                                                                 (CC_VARS.D2PTHOUT.raw() * CC_NMLIST.DT).sum()))

    CC_VARS.P0GLBSTOPRE                                 =            sum((CC_VARS.P2RIVSTO[NQ_Index,0]    +  CC_VARS.P2FLDSTO[NQ_Index,0]).view(-1).tolist())
    CC_VARS.P0GLBRIVINF                                 =           (sum(((CC_VARS.D2RIVINF[NQ_Index,0]   *  CC_NMLIST.DT).view(-1))   +
                                                                        ((CC_VARS.D2FLDINF[NQ_Index,0]    *  CC_NMLIST.DT).view(-1))).tolist())
    CC_VARS.P0GLBRIVOUT                                 =           (sum(((CC_VARS.D2RIVOUT[NQ_Index,0]   *  CC_NMLIST.DT).view(-1))      +
                                                                         ((CC_VARS.D2FLDOUT[NQ_Index,0]   *  CC_NMLIST.DT).view(-1))      +
                                                                         ((CC_VARS.D2PTHOUT[NQ_Index,0]   *  CC_NMLIST.DT).view(-1))).tolist())
                                                                    # ------------------------------------------------------------------------------------------------------------------
    CC_VARS.P2RIVSTO[NQ_Index, 0]                       =           (CC_VARS.P2RIVSTO[NQ_Index,0]       +
                                                                     CC_VARS.D2RIVINF[NQ_Index,0]       *  CC_NMLIST.DT -
                                                                     CC_VARS.D2RIVOUT[NQ_Index,0]       *  CC_NMLIST.DT)
    PT_N_M_ID                                           =           (CC_VARS.P2RIVSTO[NQ_Index,0] < 0).nonzero(as_tuple=True)[0]
    CC_VARS.P2FLDSTO[NQ_Index[PT_N_M_ID], 0]            =           CC_VARS.P2FLDSTO[NQ_Index[PT_N_M_ID], 0]      +       CC_VARS.P2RIVSTO[NQ_Index[PT_N_M_ID], 0]
    CC_VARS.P2RIVSTO[NQ_Index[PT_N_M_ID], 0]            =           torch.tensor(0,dtype=Datatype.JPRD,device=device)

    CC_VARS.P2FLDSTO[NQ_Index, 0]                       =          (CC_VARS.P2FLDSTO[NQ_Index,0]        +
                                                                    CC_VARS.D2FLDINF[NQ_Index,0]        *  CC_NMLIST.DT -
                                                                    CC_VARS.D2FLDOUT[NQ_Index,0]        *  CC_NMLIST.DT -
                                                                    CC_VARS.D2PTHOUT[NQ_Index,0]        *  CC_NMLIST.DT )
    PL_N_M_ID                                           =          (CC_VARS.P2FLDSTO[NQ_Index,0] < 0).nonzero(as_tuple=True)[0]
    CC_VARS.P2RIVSTO[NQ_Index[PL_N_M_ID], 0]            =           torch.maximum(CC_VARS.P2RIVSTO[NQ_Index[PL_N_M_ID], 0] + CC_VARS.P2FLDSTO[NQ_Index[PL_N_M_ID], 0],
                                                                    torch.tensor(0,dtype=Datatype.JPRD,device=device))
    CC_VARS.P2FLDSTO[NQ_Index[PL_N_M_ID], 0]            =           torch.tensor(0,dtype=Datatype.JPRD,device=device)

    # ------------------------------------------------------------------------------------------------------------------
    # CC_VARS.P0GLBSTONXT                                 =           CC_VARS.P2RIVSTO.raw().sum() +  CC_VARS.P2FLDSTO.raw().sum()
    CC_VARS.P0GLBSTONXT                                 =            sum(CC_VARS.P2RIVSTO[NQ_Index,0].view(-1).tolist())  +  sum(CC_VARS.P2FLDSTO[NQ_Index,0].view(-1).tolist())
    CC_VARS.D2OUTFLW[NQ_Index, 0]                       =            CC_VARS.D2RIVOUT[NQ_Index, 0] + CC_VARS.D2FLDOUT[NQ_Index, 0]
    # ------------------------------------------------------------------------------------------------------------------
    #     !! bug before v4.2 (pthout shoudl not be added)
    # CC_VARS.D2OUTFLW[NQ_Index, 1]                       =           (CC_VARS.D2RIVOUT[NQ_Index, 1]  +  CC_VARS.D2FLDOUT[NQ_Index, 1]  +
    #                                                                  CC_VARS.D2PTHOUT[NQ_Index, 1])
    DRIVROF                                             =           ((CC_VARS.D2RUNOFF[NQ_Index, 0] + CC_VARS.D2GDWRTN[NQ_Index, 0]) *
                                                                     (torch.tensor(1,dtype=Datatype.JPRB,device=device) - CC_VARS.D2FLDFRC[NQ_Index, 0]) *
                                                                     CC_NMLIST.DT)
    DFLDROF                                             =           ((CC_VARS.D2RUNOFF[NQ_Index, 0] + CC_VARS.D2GDWRTN[NQ_Index, 0]) *
                                                                                                                                (CC_VARS.D2FLDFRC[NQ_Index, 0]) *
                                                                     CC_NMLIST.DT)
    CC_VARS.P2RIVSTO[NQ_Index, 0]                       =            CC_VARS.P2RIVSTO[NQ_Index, 0]   +   DRIVROF
    CC_VARS.P2FLDSTO[NQ_Index, 0]                       =            CC_VARS.P2FLDSTO[NQ_Index, 0]   +   DFLDROF

    if  CC_NMLIST.LWEVAP:
        #   !! Find out amount of water to be extracted from flooplain reservoir
        #   !! Assuming "potential water evaporation", multiplied by flood area fraction#
        #   !! Limited by total amount of flooplain storage
        print("The 'flux' code in 75-th Line for cmf_calc_stonxt_mod.py is needed to improved")
        print("The 'flux' code in 76-th Line for cmf_calc_stonxt_mod.py is needed to improved")

    CC_VARS.D2STORGE[NQ_Index, 0]                       =             CC_VARS.P2RIVSTO[NQ_Index, 0]  +  CC_VARS.P2FLDSTO[NQ_Index, 0]


    # CC_VARS.P0GLBSTONEW                                 =             CC_VARS.D2STORGE.raw().sum()
    CC_VARS.P0GLBSTONEW                                 =             sum(CC_VARS.D2STORGE[NQ_Index,0].view(-1).tolist())
    print(f'====  CMF_CALC_STONXT====')
    print(f'P0GLBSTOPRE:    {CC_VARS.P0GLBSTOPRE:,.15f}')
    print(f'P0GLBRIVINF:    {CC_VARS.P0GLBRIVINF:,.15f}')
    print(f'P0GLBRIVOUT:    {CC_VARS.P0GLBRIVOUT:,.15f}')
    print(f'P0GLBSTONXT:    {CC_VARS.P0GLBSTONXT:,.15f}')
    print(f'P0GLBSTONEW:    {CC_VARS.P0GLBSTONEW:,.15f}')
    return  CC_VARS