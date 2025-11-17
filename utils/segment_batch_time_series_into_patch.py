import math

import torch
from typing import Dict, Tuple, Optional

from torch import Tensor


@torch.no_grad()
def segment_time_series_into_patches(
    observed_data: torch.Tensor,
    observed_tp: torch.Tensor,
    observed_mask: torch.Tensor,
    history_len: float,      # length of the history window on the same scale as observed_tp
    patch_size: float,
    stride: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Segment irregular time series into sliding-window patches on the normalized time axis.

    Inputs
    ------
    observed_data : (B, L, D)
        Observed values per time step and variable (already padded in time).
    observed_tp : (B, L)
        Normalized timestamps aligned with `observed_data`.
    observed_mask : (B, L, D)
        Binary mask indicating actual observations (1) vs. missing/padded (0).
    history_len : float
        Length of the history window on the normalized time axis.
    patch_size : float
        Window width on the normalized time axis.
    stride : float
        Step size for the sliding window.

    Returns
    -------
    observed_data_patched : (B, M_max, Lp_max, D)
        Padded observed values for each patch.
    observed_tp_patched : (B, M_max, Lp_max, D)
        Padded timestamps for each patch.
    observed_mask_patched : (B, M_max, Lp_max, D)
        Padded binary mask for each patch.
    """
    assert patch_size > 0 and stride > 0, "patch_size and stride must be positive."

    B, L, D = observed_data.shape
    device = observed_data.device
    dtype = observed_data.dtype

    # Valid time rows: at least one variable observed at this time step
    valid_row = (observed_mask.sum(dim=-1) > 0)  # (B, L)

    # Fixed number of patches decided solely by (history_len, patch_size, stride)
    if history_len <= patch_size:
        M = 1
    else:
        M = int(math.floor((history_len - patch_size) / stride) + 1)
        M = max(M, 1)

    # ---- Pass 1: compute Lp_max across all (B, M, D) within [0, history_len) ----
    Lp_max = 0
    for m in range(M):
        st = m * stride
        ed_eff = min(st + patch_size, history_len)  # clamp to history_len

        for b in range(B):
            in_win = (observed_tp[b] >= st) & (observed_tp[b] < ed_eff)
            in_win = in_win & valid_row[b] & (observed_tp[b] >= 0.0) & (observed_tp[b] < history_len)

            for d in range(D):
                pos = in_win & (observed_mask[b, :, d] > 0.5)
                Ld  = int(pos.sum().item())
                if Ld > Lp_max:
                    Lp_max = Ld

    if Lp_max == 0:
        Lp_max = 1  # corner case: no observations at all

    # ---- Pass 2: allocate and fill ----
    observed_tp_patched   = torch.zeros((B, M, Lp_max, D), dtype=dtype, device=device)
    observed_data_patched = torch.zeros((B, M, Lp_max, D), dtype=dtype, device=device)
    observed_mask_patched = torch.zeros((B, M, Lp_max, D), dtype=dtype, device=device)

    for m in range(M):
        st = m * stride
        ed_eff = min(st + patch_size, history_len)

        for b in range(B):
            in_win = (observed_tp[b] >= st) & (observed_tp[b] < ed_eff)
            in_win = in_win & valid_row[b] & (observed_tp[b] >= 0.0) & (observed_tp[b] < history_len)

            for d in range(D):
                pos = in_win & (observed_mask[b, :, d] > 0.5)
                idx = torch.nonzero(pos, as_tuple=False).squeeze(-1)
                Ld  = min(int(idx.numel()), Lp_max)
                if Ld > 0:
                    observed_tp_patched[b, m, :Ld, d]   = observed_tp[b].index_select(0, idx[:Ld])
                    observed_data_patched[b, m, :Ld, d] = observed_data[b, :, d].index_select(0, idx[:Ld])
                    observed_mask_patched[b, m, :Ld, d] = 1.0

    return observed_data_patched, observed_tp_patched, observed_mask_patched