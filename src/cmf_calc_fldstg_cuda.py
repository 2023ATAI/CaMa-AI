#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Floodplain staging for CaMa-Pytorch.

The implementation keeps the default local-inertial formula but executes the
flood/no-flood split with PyTorch tensors. This mirrors the optimization idea in
pycama-main without copying the NumPy/Numba CPU implementation.
"""
import os

import torch

os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning"
os.environ["PYTHONWARNINGS"] = "ignore::RuntimeWarning"


def _fortran_passed_flood_levels(psto_f, fldstomax_f, nlfp):
    """Replicate the default Fortran FLDSTG level scan exactly.

    Fortran v4.23 starts at level 1, advances toward NLFP, uses the exact
    comparison ``DSTOALL > P2FLDSTOMAX(..., I)``, and stops permanently at the
    first level for which that comparison is false.  The returned count is the
    number of levels passed, expressed as a zero-based PyTorch count.
    """
    passed_count = torch.zeros(psto_f.shape, dtype=torch.long, device=psto_f.device)
    can_advance = torch.ones(psto_f.shape, dtype=torch.bool, device=psto_f.device)
    for level in range(int(nlfp)):
        active = can_advance & (psto_f > fldstomax_f[:, level])
        passed_count = torch.where(active, passed_count + 1, passed_count)
        can_advance = active
    return passed_count


def CMF_CALC_FLDSTG_CUDA(CM_NMLIST, CC_NMLIST, CC_VARS, device, Datatype):
    if torch.device(device).type != "cuda":
        raise RuntimeError("CMF_CALC_FLDSTG_CUDA requires a CUDA device.")

    nseq = int(CM_NMLIST.NSEQALL)
    nlfp = int(CC_NMLIST.NLFP)

    rivsto = CC_VARS.P2RIVSTO.raw()[:nseq, 0]
    fldsto = CC_VARS.P2FLDSTO.raw()[:nseq, 0]
    rivdph = CC_VARS.D2RIVDPH.raw()[:nseq, 0]
    flddph = CC_VARS.D2FLDDPH.raw()[:nseq, 0]
    fldfrc = CC_VARS.D2FLDFRC.raw()[:nseq, 0]
    fldare = CC_VARS.D2FLDARE.raw()[:nseq, 0]
    sfcelv = CC_VARS.D2SFCELV.raw()[:nseq, 0]
    storge = CC_VARS.D2STORGE.raw()[:nseq, 0]

    grarea = CM_NMLIST.D2GRAREA.raw()[:nseq, 0]
    rivlen = CM_NMLIST.D2RIVLEN.raw()[:nseq, 0]
    rivwth = CM_NMLIST.D2RIVWTH.raw()[:nseq, 0]
    rivelv = CM_NMLIST.D2RIVELV.raw()[:nseq, 0]
    rivstomax = CM_NMLIST.D2RIVSTOMAX.raw()[:nseq, 0]
    fldstomax = CM_NMLIST.D2FLDSTOMAX.raw()[:nseq, 0, :nlfp]
    fldgrd = CM_NMLIST.D2FLDGRD.raw()[:nseq, 0, :nlfp]

    zero_b = flddph.new_tensor(0.0)
    zero_d = fldsto.new_tensor(0.0)

    pstoall = rivsto + fldsto
    dstoall = pstoall.to(dtype=Datatype.JPRB)
    CC_VARS.P0GLBSTOPRE2 = torch.sum(pstoall)

    bankfull_threshold = rivstomax.to(dtype=pstoall.dtype, device=pstoall.device)
    has_flood = pstoall > bankfull_threshold
    no_flood = ~has_flood

    rivsto[no_flood] = pstoall[no_flood]
    fldsto[no_flood] = zero_d
    rivdph_nf = dstoall[no_flood] / rivlen[no_flood] / rivwth[no_flood]
    rivdph[no_flood] = torch.maximum(rivdph_nf, zero_b)
    flddph[no_flood] = zero_b
    fldfrc[no_flood] = zero_b
    fldare[no_flood] = zero_b
    sfcelv[no_flood] = rivelv[no_flood] + rivdph[no_flood]
    storge[no_flood] = dstoall[no_flood]

    flood_idx = has_flood.nonzero(as_tuple=True)[0]
    if flood_idx.numel() > 0:
        psto_f = pstoall[flood_idx].to(dtype=Datatype.JPRB)
        rivlen_f = rivlen[flood_idx]
        rivwth_f = rivwth[flood_idx]
        rivstomax_f = rivstomax[flood_idx]
        grarea_f = grarea[flood_idx]
        rivelv_f = rivelv[flood_idx]
        fldstomax_f = fldstomax[flood_idx, :]
        fldgrd_f = fldgrd[flood_idx, :]

        dsto_fil = rivstomax_f.clone()
        dwth_fil = rivwth_f.clone()
        ddph_fil = psto_f.new_zeros(psto_f.shape)
        dwth_add = psto_f.new_zeros(psto_f.shape)
        dwth_inc = grarea_f / rivlen_f * CM_NMLIST.DFRCINC
        passed_count = psto_f.new_zeros(psto_f.shape, dtype=torch.long)
        can_advance = psto_f.new_full(psto_f.shape, True, dtype=torch.bool)


        for level in range(nlfp):
            active = can_advance & (psto_f > fldstomax_f[:, level])
            dsto_fil = torch.where(active, fldstomax_f[:, level], dsto_fil)
            dwth_fil = torch.where(active, dwth_fil + dwth_inc, dwth_fil)
            ddph_fil = torch.where(active, ddph_fil + fldgrd_f[:, level] * dwth_inc, ddph_fil)
            passed_count = torch.where(active, passed_count + 1, passed_count)
            can_advance = active

        overflow = passed_count >= nlfp
        next_level = torch.clamp(passed_count, max=nlfp - 1)
        grd_next = torch.gather(fldgrd_f, 1, next_level[:, None]).squeeze(1)
        dsto_add = psto_f - dsto_fil

        two = psto_f.new_tensor(2.0)
        sqrt_arg = dwth_fil * dwth_fil + two * dsto_add / rivlen_f / grd_next
        dwth_add_regular = -dwth_fil + torch.sqrt(sqrt_arg)
        dwth_add = torch.where(overflow, zero_b.expand_as(dwth_add), dwth_add_regular)

        flddph_f = torch.where(
            overflow,
            ddph_fil + dsto_add / dwth_fil / rivlen_f,
            ddph_fil + grd_next * dwth_add,
        )
        flddph[flood_idx] = flddph_f

        rivsto_f = rivstomax_f.to(dtype=Datatype.JPRD) + (
            rivlen_f * rivwth_f * flddph_f
        ).to(dtype=Datatype.JPRD)
        rivsto_f = torch.minimum(rivsto_f, pstoall[flood_idx])
        rivsto[flood_idx] = rivsto_f
        rivdph[flood_idx] = rivsto_f.to(dtype=Datatype.JPRB) / rivlen_f / rivwth_f

        fldsto_f = pstoall[flood_idx] - rivsto_f
        fldsto[flood_idx] = torch.maximum(fldsto_f, zero_d)

        fldfrc_f = (-rivwth_f + dwth_fil + dwth_add) / (dwth_inc * nlfp)
        fldfrc_f = torch.clamp(fldfrc_f, min=0.0, max=1.0)
        fldfrc[flood_idx] = fldfrc_f
        fldare[flood_idx] = grarea_f * fldfrc_f

        sfcelv[flood_idx] = rivelv_f + rivdph[flood_idx]
        storge[flood_idx] = (rivsto[flood_idx] + fldsto[flood_idx]).to(dtype=Datatype.JPRB)

    CC_VARS.P0GLBSTONEW2 = torch.sum((rivsto + fldsto).to(dtype=Datatype.JPRB))
    CC_VARS.P0GLBRIVSTO = torch.sum(rivsto)
    CC_VARS.P0GLBFLDSTO = torch.sum(fldsto)
    CC_VARS.P0GLBFLDARE = torch.sum(fldare)

    return CM_NMLIST, CC_VARS
