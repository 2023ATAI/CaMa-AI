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


def _get_pthout_cuda_cache(CC_NMLIST, CM_NMLIST, CC_VARS, device):
    device_obj = torch.device(device)
    if device_obj.type != "cuda":
        raise RuntimeError("CMF_CALC_PTHOUT_CUDA requires a CUDA device.")

    npth = int(CM_NMLIST.NPTHOUT)
    nlev = int(CM_NMLIST.NPTHLEV)
    nseq = int(CM_NMLIST.NSEQALL)
    device_index = device_obj.index
    if device_index is None:
        device_index = torch.cuda.current_device()

    key = (id(CM_NMLIST), device_index, npth, nlev, nseq)
    cache = getattr(CC_NMLIST, "_PTHOUT_CUDA_CACHE", None)
    if cache is not None and cache.get("key") == key:
        return cache

    path0 = torch.arange(npth, dtype=torch.long, device=device)
    level0 = torch.arange(nlev, dtype=torch.long, device=device)
    iseqp1 = CM_NMLIST.PTH_UPST.raw()[:npth].to(device=device, dtype=torch.long)
    jseqp1 = CM_NMLIST.PTH_DOWN.raw()[:npth].to(device=device, dtype=torch.long)
    in_domain = (iseqp1 > 0) & (jseqp1 > 0) & (iseqp1 <= nseq) & (jseqp1 <= nseq)

    valid_mask = torch.zeros(npth, dtype=torch.bool, device=device)
    cand = in_domain.nonzero(as_tuple=True)[0]
    if cand.numel() > 0:
        iseqp0_cand = iseqp1[cand] - 1
        jseqp0_cand = jseqp1[cand] - 1
        mask_raw = CM_NMLIST.I2MASK.raw()
        valid_mask[cand] = (mask_raw[iseqp0_cand, 0] <= 0) & (mask_raw[jseqp0_cand, 0] <= 0)

    valid_path0 = path0[valid_mask]
    iseqp0 = iseqp1[valid_path0] - 1
    jseqp0 = jseqp1[valid_path0] - 1

    cache = {
        "key": key,
        "path0": path0,
        "level0": level0,
        "valid_path_mask": valid_mask,
        "valid_path0": valid_path0,
        "iseqp0": iseqp0,
        "jseqp0": jseqp0,
        "pth_dst": CM_NMLIST.PTH_DST.raw()[:npth][valid_path0].to(device=device),
        "pth_elv": CM_NMLIST.PTH_ELV.raw()[:npth, :nlev][valid_path0, :].to(device=device),
        "pth_wth": CM_NMLIST.PTH_WTH.raw()[:npth, :nlev][valid_path0, :].to(device=device),
        "pth_man": CM_NMLIST.PTH_MAN.raw()[:nlev].to(device=device),
        "level_valid_mask": (CM_NMLIST.PTH_WTH.raw()[:npth, :nlev][valid_path0, :].to(device=device) > 0),
    }
    CC_NMLIST._PTHOUT_CUDA_CACHE = cache
    return cache


def CMF_CALC_PTHOUT_CUDA(CC_NMLIST, CM_NMLIST, CC_VARS, device, Datatype):
    if torch.device(device).type != "cuda":
        raise RuntimeError("CMF_CALC_PTHOUT_CUDA requires a CUDA device.")
    if (not getattr(CC_NMLIST, "LPTHOUT", False) or
            int(CM_NMLIST.NPTHOUT) <= 0 or
            int(CM_NMLIST.NPTHLEV) <= 0):
        return CC_VARS

    npth = int(CM_NMLIST.NPTHOUT)
    nlev = int(CM_NMLIST.NPTHLEV)
    cache = _get_pthout_cuda_cache(CC_NMLIST, CM_NMLIST, CC_VARS, device)

    sfcelv_pre = CC_VARS.D2SFCELV_PRE.raw()
    sfcelv_pre[:, :] = CM_NMLIST.D2RIVELV.raw()[:, :] + CC_VARS.D2RIVDPH_PRE.raw()[:, :]

    pthflw = CC_VARS.D1PTHFLW.raw()
    pthflw[:, :] = pthflw.new_tensor(0.0)
    pthflw_active = pthflw[:npth, :nlev]

    valid_path0 = cache["valid_path0"]
    if valid_path0.numel() > 0:
        iseqp0 = cache["iseqp0"]
        jseqp0 = cache["jseqp0"]
        pth_dst = cache["pth_dst"]
        pth_elv = cache["pth_elv"]
        pth_wth = cache["pth_wth"]
        pth_man = cache["pth_man"]

        sfcelv = CC_VARS.D2SFCELV.raw()[:, 0]
        sfcelv_pre_1d = sfcelv_pre[:, 0]
        dslp = (sfcelv[iseqp0] - sfcelv[jseqp0]) / pth_dst
        dslp = torch.maximum(
            dslp.new_tensor(-0.005),
            torch.minimum(dslp.new_tensor(0.005), dslp),
        )

        max_sfc = torch.maximum(sfcelv[iseqp0], sfcelv[jseqp0])[:, None]
        max_sfc_pre = torch.maximum(sfcelv_pre_1d[iseqp0], sfcelv_pre_1d[jseqp0])[:, None]
        zero = pthflw.new_tensor(0.0)
        dflw = torch.maximum(max_sfc - pth_elv, zero)
        dflw_pr = torch.maximum(max_sfc_pre - pth_elv, zero)
        dflw_im = torch.sqrt(dflw * dflw_pr)
        dflw_im = torch.maximum(dflw_im, torch.sqrt(dflw * dflw.new_tensor(0.01)))
        active = dflw_im > 1.0e-5

        safe_wth = torch.where(active, pth_wth, torch.ones_like(pth_wth))
        safe_dflw_im = torch.where(active, dflw_im, torch.ones_like(dflw_im))
        pthflw_pre_active = CC_VARS.D1PTHFLW_PRE.raw()[:npth, :nlev]
        dout_pr = pthflw_pre_active[valid_path0, :] / safe_wth
        numerator = (
            dout_pr +
            CC_NMLIST.PGRV * CC_NMLIST.DT * safe_dflw_im * dslp[:, None]
        )
        denominator = (
            1 +
            CC_NMLIST.PGRV * CC_NMLIST.DT * (pth_man[None, :] ** 2) *
            torch.abs(dout_pr) * safe_dflw_im ** (-7 / 3)
        )
        flow = pth_wth * numerator / denominator
        pthflw_active[valid_path0, :] = torch.where(active, flow, zero)

    pthsum = CC_VARS.D1PTHFLWSUM.raw()
    pthsum_active = pthsum[:npth]
    valid_mask = cache["valid_path_mask"]
    # Fortran semantics: reset the total flow for every PTH path before
    # summing all active layers. Invalid paths remain exactly zero because
    # D1PTHFLW was reset above.
    pthsum_active.zero_()
    pthsum_active[:] = torch.sum(pthflw_active, dim=1)

    nonzero_valid = (pthsum_active != 0) & valid_mask
    limit_path0 = cache["path0"][nonzero_valid]
    if limit_path0.numel() > 0:
        valid_pos = valid_mask.nonzero(as_tuple=True)[0]
        valid_nonzero = pthsum_active[valid_pos] != 0
        up0 = cache["iseqp0"][valid_nonzero]
        down0 = cache["jseqp0"][valid_nonzero]
        storge = CC_VARS.D2STORGE.raw()[:, 0]
        rate = (
            pthsum.new_tensor(0.05) *
            torch.minimum(storge[up0], storge[down0]) /
            torch.abs(pthsum[limit_path0] * CC_NMLIST.DT)
        )
        rate = torch.minimum(rate, pthsum.new_tensor(1.0))
        # The Fortran limiter scales every PTH elevation layer, not only
        # the first layer. Keep D1PTHFLW and D1PTHFLWSUM internally consistent.
        pthflw[limit_path0, :nlev] = (
            pthflw[limit_path0, :nlev] * rate[:, None]
        )
        pthsum[limit_path0] = pthsum[limit_path0] * rate

    return CC_VARS


def CMF_CALC_PTHOUT_CPU(CC_NMLIST, CM_NMLIST, CC_VARS, device, Datatype):
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

    # Fortran semantics: clear every PTH path before summing all layers.
    # Clearing only ID_M can leave stale values on invalid paths.
    CC_VARS.D1PTHFLWSUM[RC_Index]                           =   torch.tensor(0,dtype=Datatype.JPRB,device=device)

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

    for ILEV in range(1, CM_NMLIST.NPTHLEV + 1):
        CC_VARS.D1PTHFLW[RC_Index[DW_P_M_ID], ILEV] = (
            CC_VARS.D1PTHFLW[RC_Index[DW_P_M_ID], ILEV] * RATE
        )
    CC_VARS.D1PTHFLWSUM[RC_Index[DW_P_M_ID]]   =                CC_VARS.D1PTHFLWSUM[RC_Index[DW_P_M_ID]] * RATE

    return CC_VARS


def CMF_CALC_PTHOUT(CC_NMLIST, CM_NMLIST, CC_VARS, device, Datatype):
    backend = torch.device(device).type
    if backend == "cpu":
        return CMF_CALC_PTHOUT_CPU(CC_NMLIST, CM_NMLIST, CC_VARS, device, Datatype)
    if backend == "cuda":
        return CMF_CALC_PTHOUT_CUDA(CC_NMLIST, CM_NMLIST, CC_VARS, device, Datatype)
    raise RuntimeError(f"Unsupported CaMa-PyTorch PTHOUT backend: {backend!r}.")
