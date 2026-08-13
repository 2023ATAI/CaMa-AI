import torch


def build_upstream_matrix_torch(I1NEXT, NSEQALL, NSEQRIV, device):
    """
    Build a padded upstream-neighbor matrix on device.

    Inputs use the current CaMa-Pytorch convention: I1NEXT stores 1-based
    downstream sequence ids. Outputs are 0-based tensor indices for direct
    gather/scatter-free reads in hot inflow paths.
    """
    nseqall = int(NSEQALL)
    nseqriv = int(NSEQRIV)
    next0 = I1NEXT.raw()[:nseqriv].to(device=device, dtype=torch.long) - 1
    src0 = torch.arange(nseqriv, dtype=torch.long, device=device)
    valid = (next0 >= 0) & (next0 < nseqall)
    next0 = next0[valid]
    src0 = src0[valid]

    upn = torch.zeros(nseqall, dtype=torch.long, device=device)
    if next0.numel() > 0:
        upn.scatter_add_(0, next0, torch.ones_like(next0, dtype=torch.long))
    upnmax = int(torch.max(upn).item()) if upn.numel() > 0 else 0
    upnmax = max(upnmax, 1)

    upst = torch.full((nseqall, upnmax), -1, dtype=torch.long, device=device)
    if next0.numel() > 0:
        order = torch.argsort(next0, stable=True)
        sorted_dst = next0[order]
        sorted_src = src0[order]
        edge_pos = torch.arange(sorted_dst.numel(), dtype=torch.long, device=device)
        segment_start = torch.cumsum(upn, dim=0) - upn
        slot = edge_pos - segment_start[sorted_dst]
        upst[sorted_dst, slot] = sorted_src

    upstream_mask = upst >= 0
    return upst, upn, upstream_mask
