#!/usr/bin/env python3
"""Optimized CPU flood-stage backend with a Numba flooded-cell loop.

This module is the formal CPU FLDSTG backend. It preserves the no-flood closed-form
updates and the Fortran flooded per-cell/per-level while-loop order.
"""
import os

import numpy as np
import torch

try:
    import numba
except Exception:  # pragma: no cover - handled explicitly at runtime.
    numba = None


os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning"
os.environ["PYTHONWARNINGS"] = "ignore::RuntimeWarning"


if numba is not None:
    @numba.njit(cache=True, fastmath=False)
    def process_flooded_cells_numba_cpu(
        flooded_cells,
        p2rivsto,
        p2fldsto,
        d2rivdph,
        d2flddph,
        d2fldfrc,
        d2fldare,
        d2sfcelv,
        d2storge,
        d2rivelv,
        d2rivstomax,
        d2rivwth,
        d2rivlen,
        d2grarea,
        d2fldstomax,
        d2fldgrd,
        dfrcinc,
        nlfp,
    ):
        for idx in range(flooded_cells.shape[0]):
            iseq = flooded_cells[idx]
            pstoall_i = p2rivsto[iseq, 0] + p2fldsto[iseq, 0]
            dstoall_i = pstoall_i
            dsto_fil = d2rivstomax[iseq, 0]
            dwth_fil = d2rivwth[iseq, 0]
            ddph_fil = 0.0
            dwth_add = 0.0
            dwth_inc = d2grarea[iseq, 0] / d2rivlen[iseq, 0] * dfrcinc

            level = 0

            while level < nlfp and pstoall_i > d2fldstomax[iseq, 0, level]:
                dsto_fil = d2fldstomax[iseq, 0, level]
                dwth_fil = dwth_fil + dwth_inc
                ddph_fil = ddph_fil + d2fldgrd[iseq, 0, level] * dwth_inc
                level += 1
                if level >= nlfp:
                    break

            dsto_add = dstoall_i - dsto_fil
            if level >= nlfp:
                dwth_add = 0.0
                d2flddph[iseq, 0] = ddph_fil + dsto_add / dwth_fil / d2rivlen[iseq, 0]
            else:
                dwth_add = (
                    -dwth_fil
                    + np.sqrt(
                        dwth_fil * dwth_fil
                        + 2.0 * dsto_add / d2rivlen[iseq, 0] / d2fldgrd[iseq, 0, level]
                    )
                )
                d2flddph[iseq, 0] = ddph_fil + d2fldgrd[iseq, 0, level] * dwth_add

            rivsto_i = d2rivstomax[iseq, 0] + d2rivlen[iseq, 0] * d2rivwth[iseq, 0] * d2flddph[iseq, 0]
            if rivsto_i > pstoall_i:
                rivsto_i = pstoall_i
            p2rivsto[iseq, 0] = rivsto_i
            d2rivdph[iseq, 0] = rivsto_i / d2rivlen[iseq, 0] / d2rivwth[iseq, 0]

            fldsto_i = pstoall_i - rivsto_i
            if fldsto_i < 0.0:
                fldsto_i = 0.0
            p2fldsto[iseq, 0] = fldsto_i

            fldfrc_i = (-d2rivwth[iseq, 0] + dwth_fil + dwth_add) / (dwth_inc * nlfp)
            if fldfrc_i < 0.0:
                fldfrc_i = 0.0
            elif fldfrc_i > 1.0:
                fldfrc_i = 1.0
            d2fldfrc[iseq, 0] = fldfrc_i
            d2fldare[iseq, 0] = d2grarea[iseq, 0] * fldfrc_i
            d2sfcelv[iseq, 0] = d2rivelv[iseq, 0] + d2rivdph[iseq, 0]
            d2storge[iseq, 0] = p2rivsto[iseq, 0] + p2fldsto[iseq, 0]
else:
    process_flooded_cells_numba_cpu = None


def _require_contiguous_cpu_numpy(name, tensor):
    if tensor.device.type != "cpu":
        raise RuntimeError("%s must be a CPU tensor for CPU FLDSTG backend." % name)
    if not tensor.is_contiguous():
        raise RuntimeError("%s must be contiguous for CPU FLDSTG backend." % name)
    arr = tensor.detach().numpy()
    if not arr.flags["C_CONTIGUOUS"]:
        raise RuntimeError("%s numpy view must be C-contiguous for CPU FLDSTG backend." % name)
    if int(tensor.data_ptr()) != int(arr.__array_interface__["data"][0]):
        raise RuntimeError("%s did not produce a zero-copy numpy view." % name)
    return arr


def CMF_CALC_FLDSTG_CPU(CM_NMLIST, CC_NMLIST, CC_VARS, device, Datatype):
    if torch.device(device).type != "cpu":
        raise RuntimeError("CPU FLDSTG backend is CPU-only; run with device=cpu.")
    if process_flooded_cells_numba_cpu is None:
        raise RuntimeError("CPU FLDSTG backend requires numba, but numba is not available.")

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

    rivlen = CM_NMLIST.D2RIVLEN.raw()[:nseq, 0]
    rivwth = CM_NMLIST.D2RIVWTH.raw()[:nseq, 0]
    rivelv = CM_NMLIST.D2RIVELV.raw()[:nseq, 0]
    rivstomax = CM_NMLIST.D2RIVSTOMAX.raw()[:nseq, 0]

    zero_b = torch.tensor(0, dtype=Datatype.JPRB, device=device)
    zero_d = torch.tensor(0, dtype=Datatype.JPRD, device=device)

    pstoall = rivsto + fldsto
    has_flood = pstoall > rivstomax.to(dtype=pstoall.dtype)
    no_flood = ~has_flood

    if torch.any(no_flood):
        psto_nf = pstoall[no_flood]
        rivsto[no_flood] = psto_nf
        rivdph[no_flood] = torch.maximum(
            psto_nf.to(dtype=Datatype.JPRB) / rivlen[no_flood] / rivwth[no_flood],
            zero_b,
        )
        fldsto[no_flood] = zero_d
        flddph[no_flood] = zero_b
        fldfrc[no_flood] = zero_b
        fldare[no_flood] = zero_b
        sfcelv[no_flood] = rivelv[no_flood] + rivdph[no_flood]
        storge[no_flood] = psto_nf.to(dtype=Datatype.JPRB)

    flooded_cells = torch.nonzero(has_flood, as_tuple=False).view(-1).to(dtype=torch.int64).contiguous()
    if flooded_cells.numel() > 0:
        arrays = {
            "flooded_cells": _require_contiguous_cpu_numpy("flooded_cells", flooded_cells),
            "p2rivsto": _require_contiguous_cpu_numpy("P2RIVSTO", CC_VARS.P2RIVSTO.raw()),
            "p2fldsto": _require_contiguous_cpu_numpy("P2FLDSTO", CC_VARS.P2FLDSTO.raw()),
            "d2rivdph": _require_contiguous_cpu_numpy("D2RIVDPH", CC_VARS.D2RIVDPH.raw()),
            "d2flddph": _require_contiguous_cpu_numpy("D2FLDDPH", CC_VARS.D2FLDDPH.raw()),
            "d2fldfrc": _require_contiguous_cpu_numpy("D2FLDFRC", CC_VARS.D2FLDFRC.raw()),
            "d2fldare": _require_contiguous_cpu_numpy("D2FLDARE", CC_VARS.D2FLDARE.raw()),
            "d2sfcelv": _require_contiguous_cpu_numpy("D2SFCELV", CC_VARS.D2SFCELV.raw()),
            "d2storge": _require_contiguous_cpu_numpy("D2STORGE", CC_VARS.D2STORGE.raw()),
            "d2rivelv": _require_contiguous_cpu_numpy("D2RIVELV", CM_NMLIST.D2RIVELV.raw()),
            "d2rivstomax": _require_contiguous_cpu_numpy("D2RIVSTOMAX", CM_NMLIST.D2RIVSTOMAX.raw()),
            "d2rivwth": _require_contiguous_cpu_numpy("D2RIVWTH", CM_NMLIST.D2RIVWTH.raw()),
            "d2rivlen": _require_contiguous_cpu_numpy("D2RIVLEN", CM_NMLIST.D2RIVLEN.raw()),
            "d2grarea": _require_contiguous_cpu_numpy("D2GRAREA", CM_NMLIST.D2GRAREA.raw()),
            "d2fldstomax": _require_contiguous_cpu_numpy("D2FLDSTOMAX", CM_NMLIST.D2FLDSTOMAX.raw()),
            "d2fldgrd": _require_contiguous_cpu_numpy("D2FLDGRD", CM_NMLIST.D2FLDGRD.raw()),
        }
        process_flooded_cells_numba_cpu(
            arrays["flooded_cells"],
            arrays["p2rivsto"],
            arrays["p2fldsto"],
            arrays["d2rivdph"],
            arrays["d2flddph"],
            arrays["d2fldfrc"],
            arrays["d2fldare"],
            arrays["d2sfcelv"],
            arrays["d2storge"],
            arrays["d2rivelv"],
            arrays["d2rivstomax"],
            arrays["d2rivwth"],
            arrays["d2rivlen"],
            arrays["d2grarea"],
            arrays["d2fldstomax"],
            arrays["d2fldgrd"],
            float(CM_NMLIST.DFRCINC),
            nlfp,
        )

    CC_VARS.P0GLBSTOPRE2 = torch.sum(pstoall.to(dtype=Datatype.JPRB))
    CC_VARS.P0GLBSTONEW2 = torch.sum((rivsto + fldsto).to(dtype=Datatype.JPRB))
    CC_VARS.P0GLBRIVSTO = torch.sum(CC_VARS.P2RIVSTO.raw())
    CC_VARS.P0GLBFLDSTO = torch.sum(CC_VARS.P2FLDSTO.raw())
    CC_VARS.P0GLBFLDARE = torch.sum(CC_VARS.D2FLDARE.raw())

    return CM_NMLIST, CC_VARS
