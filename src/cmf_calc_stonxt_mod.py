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


def _scalar_tensor(value, dtype, device):
    if torch.is_tensor(value):
        return value.to(dtype=dtype, device=device)
    return torch.tensor(value, dtype=dtype, device=device)

def CMF_CALC_STONXT(CC_NMLIST, CM_NMLIST, CC_VARS , device, Datatype):
    storage_dtype = CC_VARS.P2RIVSTO.raw().dtype
    diag_dtype = CC_VARS.D2OUTFLW.raw().dtype
    dt_storage = _scalar_tensor(CC_NMLIST.DT, storage_dtype, device)
    zero_storage = torch.zeros((), dtype=storage_dtype, device=device)

    if CC_NMLIST.LGDWDLY:
        raise RuntimeError("LGDWDLY is not supported in the formal CaMa-PyTorch v1.0 CPU/CUDA release.")
    elif CC_NMLIST.LROSPLIT:
        # ! No ground water delay
        CC_VARS.D2GDWRTN[:CM_NMLIST.NSEQALL+1,1]        =       CC_VARS.D2ROFSUB[:CM_NMLIST.NSEQALL+1,1].to(dtype=CC_VARS.D2GDWRTN.raw().dtype)
        CC_VARS.P2GDWSTO[:CM_NMLIST.NSEQALL+1,1]        =       zero_storage
    # ------------------------------------------------------------------------------------------------------------------
    NQ_Index                                            =           torch.arange(1, CM_NMLIST.NSEQALL + 1, dtype=torch.long, device=device)
    # CC_VARS.P0GLBSTOPRE                                 =           CC_VARS.P2RIVSTO.raw().sum()  +    CC_VARS.P2FLDSTO.raw().sum()
    # CC_VARS.P0GLBRIVINF                                 =           (((CC_VARS.D2RIVINF.raw()   *  CC_NMLIST.DT).sum()  +
    #                                                                 (CC_VARS.D2FLDINF.raw()   *  CC_NMLIST.DT).sum()    +
    #                                                                 (CC_VARS.D2PTHINF.raw()   *  CC_NMLIST.DT).sum()))
    # CC_VARS.P0GLBRIVOUT                                 =           (((CC_VARS.D2RIVOUT.raw() * CC_NMLIST.DT).sum()     +
    #                                                                 (CC_VARS.D2FLDOUT.raw() * CC_NMLIST.DT).sum()       +
    #                                                                 (CC_VARS.D2PTHOUT.raw() * CC_NMLIST.DT).sum()))

    rivsto_old                                          =           CC_VARS.P2RIVSTO[NQ_Index, 1].to(dtype=storage_dtype).clone()
    fldsto_old                                          =           CC_VARS.P2FLDSTO[NQ_Index, 1].to(dtype=storage_dtype).clone()
    rivinf                                              =           CC_VARS.D2RIVINF[NQ_Index, 1].to(dtype=storage_dtype)
    fldinf                                              =           CC_VARS.D2FLDINF[NQ_Index, 1].to(dtype=storage_dtype)
    rivout                                              =           CC_VARS.D2RIVOUT[NQ_Index, 1].to(dtype=storage_dtype)
    fldout                                              =           CC_VARS.D2FLDOUT[NQ_Index, 1].to(dtype=storage_dtype)
    pthout                                              =           CC_VARS.D2PTHOUT[NQ_Index, 1].to(dtype=storage_dtype)
    runoff                                              =           CC_VARS.D2RUNOFF[NQ_Index, 1].to(dtype=storage_dtype)
    gdwrtn                                              =           CC_VARS.D2GDWRTN[NQ_Index, 1].to(dtype=storage_dtype)
    fldfrc                                              =           CC_VARS.D2FLDFRC[NQ_Index, 1].to(dtype=storage_dtype)

    CC_VARS.P0GLBSTOPRE                                 =           torch.sum(rivsto_old + fldsto_old)
    CC_VARS.P0GLBRIVINF                                 =           torch.sum((rivinf + fldinf) * dt_storage)
    CC_VARS.P0GLBRIVOUT                                 =           torch.sum((rivout + fldout + pthout) * dt_storage)
                                                                    # ------------------------------------------------------------------------------------------------------------------
    rivsto_after_flow                                   =           rivsto_old + rivinf * dt_storage - rivout * dt_storage
    rivsto_deficit                                      =           torch.minimum(rivsto_after_flow, zero_storage)
    rivsto_after_transfer                               =           torch.maximum(rivsto_after_flow, zero_storage)
    fldsto_after_riv_transfer                           =           fldsto_old + rivsto_deficit

    fldsto_after_flow                                   =           (fldsto_after_riv_transfer +
                                                                     fldinf * dt_storage -
                                                                     fldout * dt_storage -
                                                                     pthout * dt_storage)
    fldsto_deficit                                      =           torch.minimum(fldsto_after_flow, zero_storage)
    rivsto_after_exchange                               =           torch.maximum(rivsto_after_transfer + fldsto_deficit, zero_storage)
    fldsto_after_exchange                               =           torch.maximum(fldsto_after_flow, zero_storage)

    # ------------------------------------------------------------------------------------------------------------------
    # CC_VARS.P0GLBSTONXT                                 =           CC_VARS.P2RIVSTO.raw().sum() +  CC_VARS.P2FLDSTO.raw().sum()
    CC_VARS.P0GLBSTONXT                                 =            torch.sum(rivsto_after_exchange + fldsto_after_exchange)
    CC_VARS.D2OUTFLW[NQ_Index, 1]                       =            (rivout + fldout).to(dtype=diag_dtype)
    # ------------------------------------------------------------------------------------------------------------------
    #     !! bug before v4.2 (pthout shoudl not be added)
    # CC_VARS.D2OUTFLW[NQ_Index, 1]                       =           (CC_VARS.D2RIVOUT[NQ_Index, 1]  +  CC_VARS.D2FLDOUT[NQ_Index, 1]  +
    #                                                                  CC_VARS.D2PTHOUT[NQ_Index, 1])
    DRIVROF                                             =           ((runoff + gdwrtn) *
                                                                     (torch.ones((), dtype=storage_dtype, device=device) - fldfrc) *
                                                                     dt_storage)
    DFLDROF                                             =           ((runoff + gdwrtn) *
                                                                                                                                fldfrc *
                                                                     dt_storage)
    rivsto_new                                          =            rivsto_after_exchange + DRIVROF
    fldsto_new                                          =            fldsto_after_exchange + DFLDROF
    CC_VARS.P2RIVSTO[NQ_Index, 1]                       =            rivsto_new
    CC_VARS.P2FLDSTO[NQ_Index, 1]                       =            fldsto_new

    if  CC_NMLIST.LWEVAP:
        #   !! Find out amount of water to be extracted from flooplain reservoir
        #   !! Assuming "potential water evaporation", multiplied by flood area fraction#
        #   !! Limited by total amount of flooplain storage
        raise RuntimeError("LWEVAP is not supported in the formal CaMa-PyTorch v1.0 CPU/CUDA release.")

    CC_VARS.D2STORGE[NQ_Index, 1]                       =             (rivsto_new + fldsto_new).to(dtype=CC_VARS.D2STORGE.raw().dtype)


    # CC_VARS.P0GLBSTONEW                                 =             CC_VARS.D2STORGE.raw().sum()
    CC_VARS.P0GLBSTONEW                                 =             torch.sum(rivsto_new + fldsto_new)
    return  CC_VARS
