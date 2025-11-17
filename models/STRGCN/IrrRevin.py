
import torch
import torch.nn as nn
import torch.nn.functional as F

class IrrRevIN(nn.Module):
    """
    Reversible Instance Normalization (Min–Max, 0-1) for long-form irregular series.

    """
    def __init__(self, num_vars: int, eps: float = 1e-6, learn_affine: bool = False):
        super().__init__()
        self.num_vars = num_vars
        self.eps = eps
        self.learn_affine = learn_affine
        if learn_affine:

            self.gamma = nn.Parameter(torch.ones(1, num_vars))
            self.beta  = nn.Parameter(torch.zeros(1, num_vars))
        else:
            self.register_parameter("gamma", None)
            self.register_parameter("beta",  None)

    @torch.no_grad()
    def _per_var_minmax(self, x, var_idx, valid_mask):

        B, L = x.shape
        V = self.num_vars

        valid = valid_mask.bool()  # (B, L)
        counts = torch.zeros(B, V, device=x.device, dtype=torch.long)
        counts.scatter_add_(1, var_idx, valid.long())
        vmin = torch.full((B, V), float('inf'),  device=x.device)
        vmax = torch.full((B, V), float('-inf'), device=x.device)


        x_min = torch.where(valid, x, torch.full_like(x, float('inf')))
        vmin.scatter_reduce_(1, var_idx, x_min, reduce='amin', include_self=True)
        x_max = torch.where(valid, x, torch.full_like(x, float('-inf')))
        vmax.scatter_reduce_(1, var_idx, x_max, reduce='amax', include_self=True)
        batch_min = x_min.amin(dim=1, keepdim=True)  # (B,1)
        batch_max = x_max.amax(dim=1, keepdim=True)  # (B,1)
        batch_min = torch.where(torch.isfinite(batch_min), batch_min, torch.zeros_like(batch_min))
        batch_max = torch.where(torch.isfinite(batch_max), batch_max, torch.ones_like(batch_max))

        no_obs = (counts == 0)  # (B,V)
        vmin = torch.where(no_obs, batch_min, vmin)
        vmax = torch.where(no_obs, batch_max, vmax)

        vmax = torch.maximum(vmax, vmin + self.eps)
        return vmin, vmax

    def _gather_stats(self, stats, var_idx):
        B, V = stats.shape
        if var_idx.dim() == 1:
            var_idx = var_idx.unsqueeze(0).expand(B, -1)
        return torch.gather(stats, dim=1, index=var_idx)

    def normalize(self, x, var_idx, pad_mask=None, pred_mask=None):
        valid = (pad_mask.bool()) & (~pred_mask.bool())  
        vmin, vmax = self._per_var_minmax(x, var_idx, valid)

        if self.learn_affine:
            vmin = vmin - self.beta
            vmax = (vmax - self.beta) / (self.gamma + self.eps)

        xmin = self._gather_stats(vmin, var_idx)  # (B,L)
        xmax = self._gather_stats(vmax, var_idx)  # (B,L)
        denom = (xmax - xmin).clamp_min(self.eps)

        x_norm = (x - xmin) / denom
        return x_norm, {"vmin": vmin, "vmax": vmax}

    def denormalize(self, y, var_idx, stats):
        
        vmin, vmax = stats["vmin"], stats["vmax"]   # (B,V)
        if self.learn_affine:
            vmin = vmin + self.beta
            vmax = vmax * (self.gamma + self.eps) + self.beta

        xmin = self._gather_stats(vmin, var_idx).unsqueeze(-1)  # (B,L,1)
        xmax = self._gather_stats(vmax, var_idx).unsqueeze(-1)  # (B,L,1)
        return y * (xmax - xmin) + xmin
