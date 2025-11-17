# gp_models.py
import math
import torch
import gpytorch
from gpytorch.kernels import MaternKernel, RQKernel, ScaleKernel


class TimeVaryingMean(gpytorch.means.Mean):
    """
    Time-varying mean: supports two simple basis representations: "fourier" or "spline".
    """
    def __init__(self, num_feats: int = 16, kind: str = "fourier"):
        super().__init__()
        self.kind = kind
        self.register_parameter("bias", torch.nn.Parameter(torch.zeros(1)))
        if kind == "fourier":
            # Fixed frequencies; learn linear weights: phi(t) = [sin(2π f t), cos(2π f t)]_f
            freqs = torch.linspace(1.0, num_feats // 2, num_feats // 2)
            self.register_buffer("freqs", freqs)  # non-trainable buffer
            self.weight = torch.nn.Parameter(torch.zeros(num_feats))
        elif kind == "spline":
            knots = torch.linspace(0.0, 1.0, steps=num_feats)
            self.register_buffer("knots", knots)
            self.weight = torch.nn.Parameter(torch.zeros(num_feats))
        else:
            raise ValueError(f"Unknown mean kind: {kind}")

    def forward(self, x):
        t = x.view(-1, 1)  # (N,1)
        if self.kind == "fourier":
            phi = []
            for f in self.freqs:
                phi.append(torch.sin(2 * math.pi * f * t))
                phi.append(torch.cos(2 * math.pi * f * t))
            Phi = torch.cat(phi, dim=1)  # (N, num_feats)
        else:  # 'spline': piecewise-linear triangular basis
            centers = self.knots.view(1, -1)  # (1,K)
            Phi = torch.relu(1.0 - torch.abs(t - centers) * (len(self.knots) - 1))
        return (Phi @ self.weight.view(-1, 1)).view(-1) + self.bias


class AmplitudeNet(torch.nn.Module):
    """ Input-dependent amplitude s(t) > 0; softplus ensures positivity. """
    def __init__(self, hidden: int = 16):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(1, hidden), torch.nn.Tanh(),
            torch.nn.Linear(hidden, 1)
        )

    def forward(self, x):
        return torch.nn.functional.softplus(self.net(x))  # (N,1) positive


class ModulatedKernel(gpytorch.kernels.Kernel):
    r"""
    Non-stationary kernel: k(t,t') = s(t) s(t') k0(t,t'),
    where k0 is a stationary base kernel (e.g., Matérn + RQ) and s(t) is given by AmplitudeNet.
    """
    is_stationary = False

    def __init__(self, base_kernel: gpytorch.kernels.Kernel, amp_net: AmplitudeNet):
        super().__init__(ard_num_dims=1)
        self.base = base_kernel
        self.amp = amp_net

    def forward(self, x1, x2, **params):
        K0 = self.base(x1, x2, **params)       # (N,M)
        s1 = self.amp(x1).view(-1)             # (N,)
        s2 = self.amp(x2).view(-1)             # (M,)
        return torch.outer(s1, s2) * K0
    
    


class ExactGP1D_TimeVarying(gpytorch.models.ExactGP):
    """
    Final 1D GP: time-varying mean + [AmplitudeNet modulation] × (Matérn + RQ), wrapped by ScaleKernel.
    """
    def __init__(
        self,
        train_x,
        train_y,
        likelihood,
        mean_kind: str = "fourier",
        mean_feats: int = 16,
        amp_hidden: int = 16,
    ):
        super().__init__(train_x, train_y, likelihood)
    # time-varying mean
        self.mean_module = TimeVaryingMean(num_feats=mean_feats, kind=mean_kind)
    # stationary base kernel
        base = MaternKernel(nu=2.5) + RQKernel()
    # amplitude modulation + global scale
        self.modulated = ModulatedKernel(base_kernel=base, amp_net=AmplitudeNet(hidden=amp_hidden))
        self.covar_module = ScaleKernel(self.modulated)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
