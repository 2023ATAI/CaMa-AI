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


def CMF_CALC_OUTFLW_CPU(CC_NMLIST, CM_NMLIST, CC_VARS, device, Datatype):
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
        raise RuntimeError("LSLOPEMOUTH is not supported in the formal CaMa-PyTorch v1.0 CPU/CUDA release.")


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
            raise RuntimeError("LSLOPEMOUTH is not supported in the formal CaMa-PyTorch v1.0 CPU/CUDA release.")

        DFLW                            =           CC_VARS.D2SFCELV[RMF_Index, 1] - CM_NMLIST.D2ELEVTN[RMF_Index, 1]
        DARE                            =           (torch.maximum
                                                     (DFSTO * CM_NMLIST.D2RIVLEN[RMF_Index, 1] ** (-1) -
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


    RC_Index                            =           torch.arange(1, CM_NMLIST.NSEQRIV + 1, device=device)
    #   !! Storage change limiter to prevent sudden increase of upstream water level during backward flow (v4.23)
    DOUT                                =           torch.maximum((-CC_VARS.D2RIVOUT[RC_Index, 1] - CC_VARS.D2FLDOUT[RC_Index, 1]) * CC_NMLIST.DT,
                                                                  torch.tensor(1e-10, dtype=Datatype.JPRB, device=device))
    RATE                                =           torch.minimum(torch.tensor(0.05, dtype=Datatype.JPRB, device=device) * CC_VARS.D2STORGE[RC_Index, 1] / DOUT,
                                                                  torch.tensor(1, dtype=Datatype.JPRB, device=device))
    CC_VARS.D2RIVOUT[RC_Index, 1]      =           CC_VARS.D2RIVOUT[RC_Index, 1] * RATE
    CC_VARS.D2FLDOUT[RC_Index, 1]      =           CC_VARS.D2FLDOUT[RC_Index, 1] * RATE

    return CC_VARS


def _get_outflw_cuda_cache(CC_NMLIST, CM_NMLIST, CC_VARS, device):
    device_obj = torch.device(device)
    if device_obj.type != "cuda":
        raise RuntimeError("CMF_CALC_OUTFLW_CUDA requires a CUDA device.")

    nseq = int(CM_NMLIST.NSEQALL)
    nriv = int(CM_NMLIST.NSEQRIV)
    device_index = device_obj.index
    if device_index is None:
        device_index = torch.cuda.current_device()

    key = (
        id(CM_NMLIST),
        device_index,
        nriv,
        nseq,
        bool(getattr(CC_NMLIST, "LFLDOUT", False)),
        bool(getattr(CC_NMLIST, "LSLOPEMOUTH", False)),
    )
    cache = getattr(CC_NMLIST, "_OUTFLW_CUDA_CACHE", None)
    if cache is not None and cache.get("key") == key:
        return cache

    rc0 = torch.arange(nriv, dtype=torch.long, device=device)
    rmf0 = torch.arange(nriv, nseq, dtype=torch.long, device=device)
    next0 = CM_NMLIST.I1NEXT.raw()[:nriv].to(device=device, dtype=torch.long) - 1
    valid_next_mask = (next0 >= 0) & (next0 < nseq)

    cache = {
        "key": key,
        "nseq": nseq,
        "nriv": nriv,
        "rc0": rc0,
        "rmf0": rmf0,
        "next0": next0,
        "valid_next_mask": valid_next_mask,
        "all_next_valid": bool(torch.all(valid_next_mask)),
        "rivelv": CM_NMLIST.D2RIVELV.raw()[:nseq, 0],
        "rivwth": CM_NMLIST.D2RIVWTH.raw()[:nseq, 0],
        "rivhgt": CM_NMLIST.D2RIVHGT.raw()[:nseq, 0],
        "rivman": CM_NMLIST.D2RIVMAN.raw()[:nseq, 0],
        "rivlen": CM_NMLIST.D2RIVLEN.raw()[:nseq, 0],
        "elevtn": CM_NMLIST.D2ELEVTN.raw()[:nseq, 0],
        "nxtdst": CM_NMLIST.D2NXTDST.raw()[:nseq, 0],
    }
    CC_NMLIST._OUTFLW_CUDA_CACHE = cache
    return cache


def CMF_CALC_OUTFLW_CUDA(CC_NMLIST, CM_NMLIST, CC_VARS, device, Datatype):
    if torch.device(device).type != "cuda":
        raise RuntimeError("CMF_CALC_OUTFLW_CUDA requires a CUDA device.")

    cache = _get_outflw_cuda_cache(CC_NMLIST, CM_NMLIST, CC_VARS, device)
    if not cache["all_next_valid"]:
        raise RuntimeError(
            "CUDA OUTFLW requires valid downstream sequence IDs for all river cells; "
            "invalid topology was detected."
        )

    nseq = cache["nseq"]
    nriv = cache["nriv"]
    rc0 = cache["rc0"]
    rmf0 = cache["rmf0"]
    next0 = cache["next0"]

    rivelv = cache["rivelv"]
    rivwth = cache["rivwth"]
    rivhgt = cache["rivhgt"]
    rivman = cache["rivman"]
    rivlen = cache["rivlen"]
    elevtn = cache["elevtn"]
    nxtdst = cache["nxtdst"]

    sfcelv = CC_VARS.D2SFCELV.raw()[:nseq, 0]
    sfcelv_pre = CC_VARS.D2SFCELV_PRE.raw()[:nseq, 0]
    rivdph = CC_VARS.D2RIVDPH.raw()[:nseq, 0]
    rivdph_pre = CC_VARS.D2RIVDPH_PRE.raw()[:nseq, 0]
    flddph = CC_VARS.D2FLDDPH.raw()[:nseq, 0]
    flddph_pre = CC_VARS.D2FLDDPH_PRE.raw()[:nseq, 0]
    rivout = CC_VARS.D2RIVOUT.raw()[:nseq, 0]
    fldout = CC_VARS.D2FLDOUT.raw()[:nseq, 0]
    rivout_pre = CC_VARS.D2RIVOUT_PRE.raw()[:nseq, 0]
    fldout_pre = CC_VARS.D2FLDOUT_PRE.raw()[:nseq, 0]
    rivvel = CC_VARS.D2RIVVEL.raw()[:nseq, 0]
    dwnelv = CM_NMLIST.D2DWNELV.raw()[:nseq, 0]
    dwnelv_pre = CC_VARS.D2DWNELV_PRE.raw()[:nseq, 0]
    fldsto = CC_VARS.P2FLDSTO.raw()[:nseq, 0]
    fldsto_pre = CC_VARS.D2FLDSTO_PRE.raw()[:nseq, 0]
    storge = CC_VARS.D2STORGE.raw()[:nseq, 0]

    zero = torch.zeros((), dtype=Datatype.JPRB, device=device)
    one = torch.ones((), dtype=Datatype.JPRB, device=device)
    eps_area = torch.tensor(1e-10, dtype=Datatype.JPRB, device=device)
    eps_depth = torch.tensor(1e-6, dtype=Datatype.JPRB, device=device)
    eps_mask = torch.tensor(1e-5, dtype=Datatype.JPRB, device=device)
    slope_lim = torch.tensor(0.005, dtype=Datatype.JPRB, device=device)
    storage_frac = torch.tensor(0.05, dtype=Datatype.JPRB, device=device)

    sfcelv[:] = rivelv + rivdph
    sfcelv_pre[:] = rivelv + rivdph_pre
    flddph_pre[:] = torch.maximum(rivdph_pre - rivhgt, zero)

    if nriv > 0:
        dwnelv[rc0] = sfcelv[next0]
        dwnelv_pre[rc0] = sfcelv_pre[next0]

        dsfc = torch.maximum(sfcelv[rc0], dwnelv[rc0])
        dslp = (sfcelv[rc0] - dwnelv[rc0]) * nxtdst[rc0] ** (-1)
        dflw = dsfc - rivelv[rc0]
        dare = torch.maximum(rivwth[rc0] * dflw, eps_area)
        dsfc_pr = torch.maximum(sfcelv_pre[rc0], dwnelv_pre[rc0])
        dflw_pr = dsfc_pr - rivelv[rc0]
        dflw_im = torch.maximum((dflw * dflw_pr) ** 0.5, eps_depth)
        dout_pr = rivout_pre[rc0] * rivwth[rc0] ** (-1)
        dout = (
            rivwth[rc0] * (dout_pr + CC_NMLIST.PGRV * CC_NMLIST.DT * dflw_im * dslp) *
            (1 + CC_NMLIST.PGRV * CC_NMLIST.DT * rivman[rc0] ** 2 * torch.abs(dout_pr) *
             dflw_im ** (-7/3)) ** (-1)
        )
        dvel = rivout[rc0] * dare ** (-1)
        mask = (dflw_im > eps_mask) & (dare > eps_mask)
        rivout[rc0] = torch.where(mask, dout, zero)
        rivvel[rc0] = torch.where(mask, dvel, zero)

        if CC_NMLIST.LFLDOUT:
            dfsto = fldsto[rc0]
            dsfc = torch.maximum(sfcelv[rc0], dwnelv[rc0])
            dslp = (sfcelv[rc0] - dwnelv[rc0]) * nxtdst[rc0] ** (-1)
            dslp = torch.maximum(-slope_lim, torch.minimum(slope_lim, dslp))
            dflw = torch.maximum(dsfc - elevtn[rc0], zero)
            dare = dfsto * rivlen[rc0] ** (-1)
            dare = torch.maximum(dare - flddph[rc0] * rivwth[rc0], zero)
            dsfc_pr = torch.maximum(sfcelv_pre[rc0], dwnelv_pre[rc0])
            dflw_pr = dsfc_pr - elevtn[rc0]
            dflw_im = torch.maximum(torch.maximum((dflw * dflw_pr), zero) ** 0.5, eps_depth)
            dare_pr = fldsto_pre[rc0] * rivlen[rc0] ** (-1)
            dare_pr = torch.maximum(dare_pr - flddph_pre[rc0] * rivwth[rc0], eps_depth)
            dare_im = torch.maximum((dare * dare_pr) ** 0.5, eps_depth)
            dout_pr = fldout_pre[rc0]
            dout = (
                (dout_pr + CC_NMLIST.PGRV * CC_NMLIST.DT * dare_im * dslp) *
                (1 + CC_NMLIST.PGRV * CC_NMLIST.DT * CC_NMLIST.PMANFLD ** 2 *
                 torch.abs(dout_pr) * dflw_im ** (-4 / 3) * dare_im ** (-1)) ** (-1)
            )
            mask = (dflw_im > eps_mask) & (dare > eps_mask)
            fld_candidate = torch.where(mask, dout, zero)
            direction_mask = fld_candidate * rivout[rc0] > 0
            fldout[rc0] = torch.where(direction_mask, fld_candidate, zero)

    if rmf0.numel() > 0:
        dslp = (sfcelv[rmf0] - dwnelv[rmf0]) * CC_NMLIST.PDSTMTH ** (-1)
        if CC_NMLIST.LSLOPEMOUTH:
            raise RuntimeError("LSLOPEMOUTH is not supported in the formal CaMa-PyTorch v1.0 CPU/CUDA release.")

        dflw = rivdph[rmf0]
        dare = rivwth[rmf0] * dflw
        dare = torch.maximum(dare, eps_area)
        dflw_pr = rivdph_pre[rmf0]
        dflw_im = torch.maximum((dflw * dflw_pr) ** 0.5, eps_depth)
        dout_pr = rivout_pre[rmf0] * rivwth[rmf0] ** (-1)
        dout = (
            rivwth[rmf0] * (dout_pr + CC_NMLIST.PGRV * CC_NMLIST.DT * dflw_im * dslp) *
            (1 + CC_NMLIST.PGRV * CC_NMLIST.DT * rivman[rmf0] ** 2 * torch.abs(dout_pr) *
             dflw_im ** (-7/3)) ** (-1)
        )
        dvel = rivout[rmf0] * dare ** (-1)
        mask = (dflw_im > eps_mask) & (dare > eps_mask)
        rivout[rmf0] = torch.where(mask, dout, zero)
        rivvel[rmf0] = torch.where(mask, dvel, zero)

        if CC_NMLIST.LFLDOUT:
            dfsto = fldsto[rmf0]
            dslp = (sfcelv[rmf0] - dwnelv[rmf0]) * CC_NMLIST.PDSTMTH ** (-1)
            dslp = torch.maximum(-slope_lim, torch.minimum(slope_lim, dslp))
            if CC_NMLIST.LSLOPEMOUTH:
                raise RuntimeError("LSLOPEMOUTH is not supported in the formal CaMa-PyTorch v1.0 CPU/CUDA release.")

            dflw = sfcelv[rmf0] - elevtn[rmf0]
            dare = torch.maximum(
                dfsto * rivlen[rmf0] ** (-1) - flddph[rmf0] * rivwth[rmf0],
                zero,
            )
            dflw_pr = sfcelv_pre[rmf0] - elevtn[rmf0]
            dflw_im = torch.maximum(torch.maximum(dflw * dflw_pr, -zero) ** 0.5, eps_depth)
            dare_pr = torch.maximum(
                fldsto_pre[rmf0] * rivlen[rmf0] ** (-1) -
                flddph_pre[rmf0] * rivwth[rmf0],
                eps_depth,
            )
            dare_im = torch.maximum((dare * dare_pr) ** 0.5, eps_depth)
            dout_pr = fldout_pre[rmf0]
            dout = (
                (dout_pr + CC_NMLIST.PGRV * CC_NMLIST.DT * dare_im * dslp) *
                (1 + CC_NMLIST.PGRV * CC_NMLIST.DT * CC_NMLIST.PMANFLD ** 2 *
                 torch.abs(dout_pr) * dflw_im ** (-4 / 3) * dare_im ** (-1)) ** (-1)
            )
            mask = (dflw_im > eps_mask) & (dare > eps_mask)
            fld_candidate = torch.where(mask, dout, zero)
            direction_mask = fld_candidate * rivout[rmf0] > 0
            fldout[rmf0] = torch.where(direction_mask, fld_candidate, zero)

    if nriv > 0:
        dout = torch.maximum((-rivout[rc0] - fldout[rc0]) * CC_NMLIST.DT, eps_area)
        rate = torch.minimum(storage_frac * storge[rc0] / dout, one)
        rivout[rc0] = rivout[rc0] * rate
        fldout[rc0] = fldout[rc0] * rate

    return CC_VARS


def CMF_CALC_OUTFLW(CC_NMLIST, CM_NMLIST, CC_VARS, device, Datatype):
    backend = torch.device(device).type
    if backend == "cpu":
        return CMF_CALC_OUTFLW_CPU(CC_NMLIST, CM_NMLIST, CC_VARS, device, Datatype)
    if backend == "cuda":
        return CMF_CALC_OUTFLW_CUDA(CC_NMLIST, CM_NMLIST, CC_VARS, device, Datatype)
    raise RuntimeError(f"Unsupported CaMa-PyTorch OUTFLW backend: {backend!r}.")

def CMF_CALC_INFLOW_CPU(CC_NMLIST, CM_NMLIST, CC_VARS, device, Datatype):
    """
    Formal CPU inflow backend.

    This keeps the formal conservation and pathway formulas, but works on
    0-based raw tensor views to reduce Ftensor_2D wrapper/index-shift overhead.
    Normal river inflow still uses torch.index_add_ to stay close to the
    Fortran accumulation semantics.
    """
    if torch.device(device).type != "cpu":
        raise RuntimeError("CMF_CALC_INFLOW_CPU requires a CPU device.")

    nseq = int(CM_NMLIST.NSEQALL)
    nriv = int(CM_NMLIST.NSEQRIV)
    zero_b = torch.zeros((), dtype=Datatype.JPRB, device=device)
    one_b = torch.ones((), dtype=Datatype.JPRB, device=device)

    rivout = CC_VARS.D2RIVOUT.raw()[:nseq, 0]
    fldout = CC_VARS.D2FLDOUT.raw()[:nseq, 0]
    rivsto = CC_VARS.P2RIVSTO.raw()[:nseq, 0]
    fldsto = CC_VARS.P2FLDSTO.raw()[:nseq, 0]
    rivinf = CC_VARS.D2RIVINF.raw()[:nseq, 0]
    fldinf = CC_VARS.D2FLDINF.raw()[:nseq, 0]
    pthout = CC_VARS.D2PTHOUT.raw()[:nseq, 0]

    p2rivinf = torch.zeros(nseq, dtype=Datatype.JPRD, device=device)
    p2fldinf = torch.zeros(nseq, dtype=Datatype.JPRD, device=device)
    p2pthout = torch.zeros(nseq, dtype=Datatype.JPRD, device=device)
    p2stoout = torch.zeros(nseq, dtype=Datatype.JPRD, device=device)
    d2rate = torch.ones(nseq, dtype=Datatype.JPRB, device=device)

    if nriv > 0:
        rc0 = torch.arange(nriv, device=device)
        next0 = CM_NMLIST.I1NEXT.raw()[:nriv].to(device=device, dtype=torch.long) - 1

        out_r1 = torch.maximum(rivout[:nriv], zero_b)
        out_r2 = torch.maximum(-rivout[:nriv], zero_b)
        out_f1 = torch.maximum(fldout[:nriv], zero_b)
        out_f2 = torch.maximum(-fldout[:nriv], zero_b)
        diup = (out_r1 + out_f1) * CC_NMLIST.DT
        didw = (out_r2 + out_f2) * CC_NMLIST.DT

        p2stoout.index_add_(0, rc0, diup.to(dtype=Datatype.JPRD))
        p2stoout.index_add_(0, next0, didw.to(dtype=Datatype.JPRD))

    if nseq > nriv:
        mouth = slice(nriv, nseq)
        out_r1 = torch.maximum(rivout[mouth], zero_b)
        out_f1 = torch.maximum(fldout[mouth], zero_b)
        p2stoout[mouth] += ((out_r1 + out_f1) * CC_NMLIST.DT).to(dtype=Datatype.JPRD)

    path_idx = None
    iseqp0 = None
    jseqp0 = None
    if CC_NMLIST.LPTHOUT:
        npth = int(CM_NMLIST.NPTHOUT)
        if npth > 0:
            iseqp1_all = CM_NMLIST.PTH_UPST.raw()[:npth].to(device=device, dtype=torch.long)
            jseqp1_all = CM_NMLIST.PTH_DOWN.raw()[:npth].to(device=device, dtype=torch.long)
            in_domain = (
                (iseqp1_all > 0) & (jseqp1_all > 0) &
                (iseqp1_all <= nseq) & (jseqp1_all <= nseq)
            )
            valid = torch.zeros(npth, dtype=torch.bool, device=device)
            if torch.any(in_domain):
                cand = torch.where(in_domain)[0]
                cand_iseqp0 = iseqp1_all[cand] - 1
                cand_jseqp0 = jseqp1_all[cand] - 1
                mask_raw = CM_NMLIST.I2MASK.raw()
                valid[cand] = (mask_raw[cand_iseqp0, 0] <= 0) & (mask_raw[cand_jseqp0, 0] <= 0)

            if torch.any(valid):
                path_idx = torch.where(valid)[0]
                iseqp0 = iseqp1_all[path_idx] - 1
                jseqp0 = jseqp1_all[path_idx] - 1
                path_sum = CC_VARS.D1PTHFLWSUM.raw()[:npth]
                path_sum_valid = path_sum[path_idx]
                p2stoout.index_add_(0, iseqp0, (torch.maximum(path_sum_valid, zero_b) * CC_NMLIST.DT).to(dtype=Datatype.JPRD))
                p2stoout.index_add_(0, jseqp0, (torch.maximum(-path_sum_valid, zero_b) * CC_NMLIST.DT).to(dtype=Datatype.JPRD))

    active = p2stoout > 1.0e-8
    if torch.any(active):
        d2rate[active] = torch.minimum(
            (rivsto[active] + fldsto[active]) / p2stoout[active].to(dtype=Datatype.JPRB),
            one_b,
        )

    if nriv > 0:
        pos = rivout[:nriv] >= 0
        rate_src = torch.where(pos, d2rate[:nriv], d2rate[next0])
        rivout[:nriv] *= rate_src
        fldout[:nriv] *= rate_src

        p2rivinf.index_add_(0, next0, rivout[:nriv].to(dtype=Datatype.JPRD))
        p2fldinf.index_add_(0, next0, fldout[:nriv].to(dtype=Datatype.JPRD))

    if nseq > nriv:
        rivout[nriv:nseq] *= d2rate[nriv:nseq]
        fldout[nriv:nseq] *= d2rate[nriv:nseq]

    if path_idx is not None:
        npth = int(CM_NMLIST.NPTHOUT)
        nlev = int(CM_NMLIST.NPTHLEV)
        d1pth = CC_VARS.D1PTHFLW.raw()[:npth, :nlev]
        path_sum = CC_VARS.D1PTHFLWSUM.raw()[:npth]
        rate_up = d2rate[iseqp0]
        rate_down = d2rate[jseqp0]

        for ilev in range(nlev):
            flow = d1pth[path_idx, ilev]
            d1pth[path_idx, ilev] = torch.where(flow >= 0, flow * rate_up, flow * rate_down)

        flow_sum = path_sum[path_idx]
        path_sum[path_idx] = torch.where(flow_sum >= 0, flow_sum * rate_up, flow_sum * rate_down)
        path_sum_valid = path_sum[path_idx]
        p2pthout.index_add_(0, iseqp0, path_sum_valid.to(dtype=Datatype.JPRD))
        p2pthout.index_add_(0, jseqp0, -path_sum_valid.to(dtype=Datatype.JPRD))

    rivinf[:] = p2rivinf.to(dtype=rivinf.dtype)
    fldinf[:] = p2fldinf.to(dtype=fldinf.dtype)
    pthout[:] = p2pthout.to(dtype=pthout.dtype)

    return CC_VARS


def _get_inflow_cuda_cache(CC_NMLIST, CM_NMLIST, CC_VARS, device):
    device_obj = torch.device(device)
    if device_obj.type != "cuda":
        raise RuntimeError("CMF_CALC_INFLOW_CUDA requires a CUDA device.")

    nseq = int(CM_NMLIST.NSEQALL)
    nriv = int(CM_NMLIST.NSEQRIV)
    npth = int(getattr(CM_NMLIST, "NPTHOUT", 0))
    nlev = int(getattr(CM_NMLIST, "NPTHLEV", 0))
    device_index = device_obj.index
    if device_index is None:
        device_index = torch.cuda.current_device()

    key = (id(CM_NMLIST), device_index, nseq, nriv, npth, nlev, bool(getattr(CC_NMLIST, "LPTHOUT", False)))
    cache = getattr(CC_NMLIST, "_INFLOW_CUDA_CACHE", None)
    if cache is not None and cache.get("key") == key:
        return cache

    empty_long = torch.empty(0, dtype=torch.long, device=device)
    rc0 = torch.arange(nriv, dtype=torch.long, device=device)
    next0 = empty_long
    if nriv > 0:
        next0 = CM_NMLIST.I1NEXT.raw()[:nriv].to(device=device, dtype=torch.long) - 1
    mouth0 = torch.arange(nriv, nseq, dtype=torch.long, device=device)

    path_idx0 = empty_long
    iseqp0 = empty_long
    jseqp0 = empty_long
    if bool(getattr(CC_NMLIST, "LPTHOUT", False)) and npth > 0:
        iseqp1_all = CM_NMLIST.PTH_UPST.raw()[:npth].to(device=device, dtype=torch.long)
        jseqp1_all = CM_NMLIST.PTH_DOWN.raw()[:npth].to(device=device, dtype=torch.long)
        in_domain = (
            (iseqp1_all > 0) & (jseqp1_all > 0) &
            (iseqp1_all <= nseq) & (jseqp1_all <= nseq)
        )
        cand = in_domain.nonzero(as_tuple=True)[0]
        if cand.numel() > 0:
            cand_iseqp0 = iseqp1_all[cand] - 1
            cand_jseqp0 = jseqp1_all[cand] - 1
            mask_raw = CM_NMLIST.I2MASK.raw()
            valid = (mask_raw[cand_iseqp0, 0] <= 0) & (mask_raw[cand_jseqp0, 0] <= 0)
            path_idx0 = cand[valid]
            iseqp0 = iseqp1_all[path_idx0] - 1
            jseqp0 = jseqp1_all[path_idx0] - 1

    cache = {
        "key": key,
        "river_src0": rc0,
        "river_dst0": next0,
        "valid_downstream_mask": (next0 >= 0) & (next0 < nseq),
        "mouth0": mouth0,
        "path_idx0": path_idx0,
        "path_iseqp0": iseqp0,
        "path_jseqp0": jseqp0,
        "path_valid_mask": torch.ones(path_idx0.numel(), dtype=torch.bool, device=device),
    }
    CC_NMLIST._INFLOW_CUDA_CACHE = cache
    return cache


def CMF_CALC_INFLOW_CUDA(CC_NMLIST, CM_NMLIST, CC_VARS, device, Datatype):
    """
    CUDA-only inflow route using the same conservation and pathway semantics
    as the CPU backend, with static topology indices cached on GPU.
    """
    if torch.device(device).type != "cuda":
        raise RuntimeError("CMF_CALC_INFLOW_CUDA requires a CUDA device.")

    nseq = int(CM_NMLIST.NSEQALL)
    nriv = int(CM_NMLIST.NSEQRIV)
    npth = int(getattr(CM_NMLIST, "NPTHOUT", 0))
    nlev = int(getattr(CM_NMLIST, "NPTHLEV", 0))
    cache = _get_inflow_cuda_cache(CC_NMLIST, CM_NMLIST, CC_VARS, device)

    rivout = CC_VARS.D2RIVOUT.raw()[:nseq, 0]
    fldout = CC_VARS.D2FLDOUT.raw()[:nseq, 0]
    rivsto = CC_VARS.P2RIVSTO.raw()[:nseq, 0]
    fldsto = CC_VARS.P2FLDSTO.raw()[:nseq, 0]
    rivinf = CC_VARS.D2RIVINF.raw()[:nseq, 0]
    fldinf = CC_VARS.D2FLDINF.raw()[:nseq, 0]
    pthout = CC_VARS.D2PTHOUT.raw()[:nseq, 0]

    p2rivinf = torch.zeros(nseq, dtype=Datatype.JPRD, device=device)
    p2fldinf = torch.zeros(nseq, dtype=Datatype.JPRD, device=device)
    p2pthout = torch.zeros(nseq, dtype=Datatype.JPRD, device=device)
    p2stoout = torch.zeros(nseq, dtype=Datatype.JPRD, device=device)
    d2rate = torch.ones(nseq, dtype=Datatype.JPRB, device=device)
    zero_flow = rivout.new_tensor(0.0)
    one_rate = d2rate.new_tensor(1.0)

    if nriv > 0:
        rc0 = cache["river_src0"]
        next0 = cache["river_dst0"]
        valid = cache["valid_downstream_mask"]
        rc_valid = rc0[valid]
        dst_valid = next0[valid]
        out_r1 = torch.maximum(rivout[rc0], zero_flow)
        out_f1 = torch.maximum(fldout[rc0], zero_flow)
        p2stoout.index_add_(0, rc0, ((out_r1 + out_f1) * CC_NMLIST.DT).to(dtype=Datatype.JPRD))
        if dst_valid.numel() > 0:
            out_r2 = torch.maximum(-rivout[rc_valid], zero_flow)
            out_f2 = torch.maximum(-fldout[rc_valid], zero_flow)
            p2stoout.index_add_(0, dst_valid, ((out_r2 + out_f2) * CC_NMLIST.DT).to(dtype=Datatype.JPRD))

    mouth0 = cache["mouth0"]
    if mouth0.numel() > 0:
        out_r1 = torch.maximum(rivout[mouth0], zero_flow)
        out_f1 = torch.maximum(fldout[mouth0], zero_flow)
        p2stoout[mouth0] += ((out_r1 + out_f1) * CC_NMLIST.DT).to(dtype=Datatype.JPRD)

    path_idx0 = cache["path_idx0"]
    iseqp0 = cache["path_iseqp0"]
    jseqp0 = cache["path_jseqp0"]
    has_path = bool(getattr(CC_NMLIST, "LPTHOUT", False)) and path_idx0.numel() > 0
    if has_path:
        path_sum = CC_VARS.D1PTHFLWSUM.raw()[:npth]
        path_sum_valid = path_sum[path_idx0]
        p2stoout.index_add_(
            0, iseqp0,
            (torch.maximum(path_sum_valid, zero_flow) * CC_NMLIST.DT).to(dtype=Datatype.JPRD),
        )
        p2stoout.index_add_(
            0, jseqp0,
            (torch.maximum(-path_sum_valid, zero_flow) * CC_NMLIST.DT).to(dtype=Datatype.JPRD),
        )

    active = p2stoout > 1.0e-8
    denom = torch.where(active, p2stoout, torch.ones_like(p2stoout))
    rate_candidate = ((rivsto + fldsto) / denom).to(dtype=Datatype.JPRB)
    d2rate = torch.where(active, torch.minimum(rate_candidate, one_rate), d2rate)

    if nriv > 0:
        rc0 = cache["river_src0"]
        next0 = cache["river_dst0"]
        valid = cache["valid_downstream_mask"]
        rc_valid = rc0[valid]
        dst_valid = next0[valid]
        rate_src = torch.ones_like(rivout[rc0])
        positive = rivout[rc0] >= 0
        rate_src[positive] = d2rate[rc0[positive]]
        if dst_valid.numel() > 0:
            valid_pos = valid.nonzero(as_tuple=True)[0]
            positive_valid = rivout[rc_valid] >= 0
            rate_src[valid_pos[~positive_valid]] = d2rate[dst_valid[~positive_valid]]
        rivout[rc0] *= rate_src
        fldout[rc0] *= rate_src

        if dst_valid.numel() > 0:
            p2rivinf.index_add_(0, dst_valid, rivout[rc_valid].to(dtype=Datatype.JPRD))
            p2fldinf.index_add_(0, dst_valid, fldout[rc_valid].to(dtype=Datatype.JPRD))

    if mouth0.numel() > 0:
        rivout[mouth0] *= d2rate[mouth0]
        fldout[mouth0] *= d2rate[mouth0]

    if has_path:
        d1pth = CC_VARS.D1PTHFLW.raw()[:npth, :nlev]
        path_sum = CC_VARS.D1PTHFLWSUM.raw()[:npth]
        rate_up = d2rate[iseqp0]
        rate_down = d2rate[jseqp0]

        for ilev in range(nlev):
            flow = d1pth[path_idx0, ilev]
            d1pth[path_idx0, ilev] = torch.where(flow >= 0, flow * rate_up, flow * rate_down)

        flow_sum = path_sum[path_idx0]
        path_sum[path_idx0] = torch.where(flow_sum >= 0, flow_sum * rate_up, flow_sum * rate_down)
        path_sum_valid = path_sum[path_idx0]
        p2pthout.index_add_(0, iseqp0, path_sum_valid.to(dtype=Datatype.JPRD))
        p2pthout.index_add_(0, jseqp0, -path_sum_valid.to(dtype=Datatype.JPRD))

    rivinf[:] = p2rivinf.to(dtype=rivinf.dtype)
    fldinf[:] = p2fldinf.to(dtype=fldinf.dtype)
    pthout[:] = p2pthout.to(dtype=pthout.dtype)

    return CC_VARS


def CMF_CALC_INFLOW(CC_NMLIST, CM_NMLIST, CC_VARS, device, Datatype):
    backend = torch.device(device).type
    if backend == "cpu":
        return CMF_CALC_INFLOW_CPU(CC_NMLIST, CM_NMLIST, CC_VARS, device, Datatype)
    if backend == "cuda":
        return CMF_CALC_INFLOW_CUDA(CC_NMLIST, CM_NMLIST, CC_VARS, device, Datatype)
    raise RuntimeError(f"Unsupported CaMa-PyTorch INFLOW backend: {backend!r}.")
