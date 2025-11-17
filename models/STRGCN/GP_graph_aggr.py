from torch import nn
import torch, os, math, copy, random
import gpytorch
from gpytorch.priors import GammaPrior, LogNormalPrior
from gpytorch.constraints import GreaterThan, Interval
from gpytorch.kernels import RQKernel
import numpy as np
import matplotlib.pyplot as plt
from .GP_models import ExactGP1D_TimeVarying as _ExactGP1D


class GaussianBasedGraphAggregationMaskGenerator(nn.Module):
    """
        Training: For each variable id (vid) we learn a GP with shared prior hyperparameters.
            In every iteration we loop over samples containing that variable, treat (t_s, y_s) for that
            variable as its training set, compute the marginal log-likelihood, accumulate losses and then
            perform one optimizer step (samples independent; hypers shared per variable).
        Inference: Within a batch of irregular time points, group points by variable and sort by time; we
            sequentially add observations via GP fantasy updates, measuring the decrease in the sum of
            posterior standard deviations as information gain. We then partition each variable's timeline
            into hyper_num_nodes groups of near-equal total information, yielding a hard mask [B, Lin, Lout].
        Conventions:
            - train_dataset[i] -> (L_i, 4) tensor: [t_norm, value, var_id, pred_mask]; t/value are normalized.
            - vars_idx (forward input) are contiguous indices 0..V-1 (aligned with DataLoader long_collate_fn).
    """

    def __init__(self,
                 train_dataset,
                 hyper_num_nodes,
                 pretrain_gp_models_saved_path=None,
                 gp_train_iters=100,
                 gp_lr=0.05,
                 gp_sample_ratio=0.25,
                 allow_dynamic_hypernode_allocation=False,
                 normalize_across_vars=False,
                 **kwargs):
        super().__init__()
        self.hyper_num_nodes = hyper_num_nodes
        self.device = kwargs.get("device", "cuda")

        # train_dataset can be None in eval/test mode
        self.train_dataset = train_dataset

        # Decide default cache path only when we have a dataset name
        if pretrain_gp_models_saved_path is None:
            if self.train_dataset is None:
                raise ValueError(
                    "[GP] pretrain_gp_models_saved_path must be provided when train_dataset is None."
                )
            _default_path = f"./cache/cached_gp_models_{self.train_dataset._make_cache_name()}_iter{gp_train_iters}_lr{gp_lr}_bs{gp_sample_ratio}.pt"
            self.pretrain_gp_models_saved_path = _default_path
        else:
            self.pretrain_gp_models_saved_path = pretrain_gp_models_saved_path

        self.gp_train_iters = int(gp_train_iters)
        self.gp_lr = float(gp_lr)
        self.gp_sample_ratio = float(gp_sample_ratio)

        self.allow_dynamic_hypernode_allocation = allow_dynamic_hypernode_allocation
        self.normalize_across_vars = normalize_across_vars

        self.num_vars = None
        self.state_dicts_per_var = None
        self.var2col = None
        self.col2var = None

        # (1) Train-mode: scan & pretrain; Eval-mode: load-only
        if self.train_dataset is not None:
            indices_by_var = self.get_trian_data(self.train_dataset)
            self.pretrain_gp_model(indices_by_var)
            self.load_cached_flag = False
        else:
            # strictly load pretrained state in eval/test
            if not (self.pretrain_gp_models_saved_path and os.path.exists(self.pretrain_gp_models_saved_path)):
                raise FileNotFoundError(
                    f"[GP] pretrained file not found: {self.pretrain_gp_models_saved_path}"
                )
            self._load_pretrained_model(self.pretrain_gp_models_saved_path)
            self.load_cached_flag = True

        # (2) Build per-var prototypes for inference utilities
        self.instantiate_gp_model_cache()
        
        if not self.load_cached_flag:
            base, _ = os.path.splitext(self.pretrain_gp_models_saved_path)
            figs_dir = base + "_figs"
            self.plot_prior_per_variable(save_dir=figs_dir, num_points=512)
            print(f"[GP] prior figures saved to: {figs_dir}")

    def _estimate_empirical_priors(self, indices_by_var=None):
        """
        Estimate empirical priors for each variable vid across the whole training set:
            - noise_mean        : Rough observation noise variance (half the median squared adjacent difference).
            - outscale_mean     : Latent function variance (data variance - noise variance; clipped positive).
            - lengthscale_mean  : Median positive time interval multiplied by a gentle smoothing factor.
            - rq_alpha_mean     : Alpha of the RQ kernel (mild default; can be tuned/learned later).
        Returns:
            dict[int -> dict] keyed by vid (0..V-1).
        """
        priors_all = {}
        all_pairs_per_var = {}  # vid -> list[(t,y)]
        dt_list_per_var = {}  # vid -> list[Δt>0]
        dy_list_per_var = {}  # vid -> list[Δy]

        for idx in range(len(self.train_dataset)):
            smp = self.train_dataset[idx]  # (L,4) = [t_norm, value, var_id, pred_mask]
            if isinstance(smp, (tuple, list)):
                smp = smp[0]
            elif isinstance(smp, dict):
                smp = smp["sample_long"]
            if smp.numel() == 0:
                continue
            t = smp[:, 0].detach().cpu().numpy()
            y = smp[:, 1].detach().cpu().numpy()
            v = smp[:, 2].detach().cpu().numpy().astype(int)

            for vid in np.unique(v):
                m = (v == vid)
                if not np.any(m):
                    continue
                t_v = t[m]
                y_v = y[m]
                if t_v.size == 0:
                    continue

                all_pairs_per_var.setdefault(vid, []).extend(list(zip(t_v, y_v)))

                ord_idx = np.argsort(t_v)
                t_sorted = t_v[ord_idx]
                y_sorted = y_v[ord_idx]
                if t_sorted.size >= 2:
                    dt = np.diff(t_sorted)
                    dy = np.diff(y_sorted)
                    dt_list_per_var.setdefault(vid, []).extend(list(dt[dt > 0]))
                    if dy.size > 0:
                        dy_list_per_var.setdefault(vid, []).extend(list(dy))

        for vid in range(int(self.num_vars or 0)):
            pairs = all_pairs_per_var.get(vid, [])
            if len(pairs) < 3:
                priors_all[vid] = dict(
                    noise_mean=1e-3,
                    outscale_mean=1e-2,
                    lengthscale_mean=0.1,
                    rq_alpha_mean=1.0
                )
                continue

            arr = np.asarray(pairs, dtype=np.float64)
            y_all = arr[:, 1]
            var_y = float(np.var(y_all)) if y_all.size > 1 else 1e-3

            dy_all = np.asarray(dy_list_per_var.get(vid, []), dtype=np.float64)
            if dy_all.size >= 2:
                noise_mean = float(np.median(dy_all ** 2) / 2.0)
                noise_mean = max(noise_mean, 1e-5)
            else:
                noise_mean = min(0.1 * var_y, 1e-3)

            outscale_mean = max(var_y - noise_mean, 1e-6)

            dt_all = np.asarray(dt_list_per_var.get(vid, []), dtype=np.float64)
            if dt_all.size == 0:
                med_dt = 0.05
            else:
                med_dt = float(np.median(dt_all))
                med_dt = max(med_dt, 1e-3)
            lengthscale_mean = float(np.clip(5.0 * med_dt, 0.02, 0.5))

            priors_all[vid] = dict(
                noise_mean=noise_mean,
                outscale_mean=outscale_mean,
                lengthscale_mean=lengthscale_mean,
                rq_alpha_mean=1.0,  # mild default
            )

        return priors_all

    def instantiate_gp_model_cache(self):
        """
        Build an in-memory prototype model per variable (eval mode).
        During inference we deepcopy the prototype and use set_train_data / fantasy updates instead
        of repeatedly loading state_dict.
        """
        assert self.state_dicts_per_var is not None and self.num_vars is not None
        self._gp_model_protos = []  # list[_ExactGP1D]

        for vid in range(self.num_vars):
            like = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
            # Placeholder training pair (overwritten with real data at inference)
            dummy_x = torch.zeros(1, 1, dtype=torch.float32, device=self.device)
            dummy_y = torch.zeros(1, dtype=torch.float32, device=self.device)
            m = _ExactGP1D(dummy_x, dummy_y, like).to(self.device)
            m.load_state_dict(self.state_dicts_per_var[vid], strict=False)
            m.eval()
            m.likelihood.eval()
            self._gp_model_protos.append(m)

    def largest_remainder_alloc(self, weights: torch.Tensor, total: int):
        """
        Allocate 'total' integer slots across len(weights) categories proportionally to weights:
            - First take floor(total * w_i / sum_w) for base allocation.
            - Distribute remaining slots to largest fractional parts.
        Returns: list[int] of length K.
        """
        K = weights.numel()
        if total <= 0 or K == 0:
            return [0] * int(K)
        sum_w = float(weights.sum().item())
        if sum_w <= 0.0:
            # All-zero weights: fall back to uniform allocation; distribute remainder from the front
            base = [total // K] * K
            for i in range(total % K):
                base[i] += 1
            return base

        raw = weights * (total / sum_w)
        base = torch.floor(raw).to(torch.int64)
        remain = int(total - int(base.sum().item()))
        if remain > 0:
            frac = raw - base.to(raw.dtype)
            order = torch.argsort(frac, descending=True)
            for i in range(remain):
                base[int(order[i].item())] += 1
        return [int(x) for x in base.tolist()]

        # ========================== Data summary: variable -> sample indices ==========================
    def get_trian_data(self, train_dataset):

        if hasattr(train_dataset, "var2col") and isinstance(
            train_dataset.var2col, dict
        ):
            self.var2col = {int(k): int(v) for k, v in train_dataset.var2col.items()}
        else:
            raw_ids = set()
            for i in range(len(train_dataset)):
                sample = train_dataset[i]
                if sample.numel() == 0:
                    continue
                raw_ids.update(int(v.item()) for v in sample[:, 2].long().unique())
            raw_ids = sorted(list(raw_ids))
            self.var2col = {raw: j for j, raw in enumerate(raw_ids)}
        self.col2var = {col: raw for raw, col in self.var2col.items()}
        self.num_vars = int(len(self.var2col))


        indices_by_var = {c: [] for c in range(self.num_vars)}
        for i in range(len(train_dataset)):
            sample = train_dataset[i]
            if isinstance(sample, (tuple, list)):  # new tuple/list case
                sample = sample[0]
            elif isinstance(sample, dict):
                sample = sample["sample_long"]
            if sample.numel() == 0:
                continue
            raw_ids_in_sample = [int(v.item()) for v in sample[:, 2].long().unique()]
            for rid in raw_ids_in_sample:
                c = self.var2col.get(rid, None)
                if c is not None:
                    indices_by_var[c].append(i)
        return indices_by_var

    # ======================== Pretraining (sample-independent + shared hypers) =========================
    def pretrain_gp_model(self, indices_by_var):
        """
        If cache exists: load directly; otherwise for each variable vid perform joint training by
            summing per-sample marginal log-likelihoods:
                total_loss(vid) = sum_s -log p(y_s | t_s; θ_vid)
        After training, store only each variable's state_dict (shared prior hyperparameters).
        """
        if self.pretrain_gp_models_saved_path is not None and os.path.exists(
                self.pretrain_gp_models_saved_path
        ):
            self._load_pretrained_model(self.pretrain_gp_models_saved_path)
            return

        total_samples = len(self.train_dataset)
        sample_batch_size = int(total_samples * self.gp_sample_ratio)

        assert self.num_vars is not None and self.num_vars > 0
        self.state_dicts_per_var = []

        priors_all = self._estimate_empirical_priors(indices_by_var)
        for vid in range(self.num_vars):
            sample_indices = indices_by_var.get(vid, [])
            total_points = 0
            for idx in sample_indices:
                smp = self.train_dataset[idx]
                if isinstance(smp, (tuple, list)):
                    smp = smp[0]
                elif isinstance(smp, dict):
                    smp = smp["sample_long"]
                col_of = lambda rid: self.var2col[int(rid)]
                col_tensor = torch.tensor(
                    [col_of(int(v.item())) for v in smp[:, 2].long()],
                    dtype=torch.long,
                    device=smp.device,
                )
                total_points += int((col_tensor == vid).sum().item())

            # Initialize placeholder training pair
            if len(sample_indices) == 0 or total_points == 0:
                print(f"[GP][var={vid}] skipped: no data.")
                tx0 = torch.zeros(1, 1, dtype=torch.float32, device=self.device)
                ty0 = torch.zeros(1, dtype=torch.float32, device=self.device)
            else:
                init_idx = sample_indices[0]
                smp = self.train_dataset[init_idx]  # (L,4)
                if isinstance(smp, (tuple, list)):
                    smp = smp[0]
                elif isinstance(smp, dict):
                    smp = smp["sample_long"]
                col_of = lambda rid: self.var2col[int(rid)]
                col_tensor = torch.tensor(
                    [col_of(int(v.item())) for v in smp[:, 2].long()],
                    dtype=torch.long,
                    device=smp.device,
                )
                mask0 = (col_tensor == vid)
                if mask0.any():
                    tx0 = smp[mask0, 0].reshape(-1, 1).to(torch.float32).to(self.device)
                    ty0 = smp[mask0, 1].reshape(-1).to(torch.float32).to(self.device)
                else:
                    tx0 = torch.zeros(1, 1, dtype=torch.float32, device=self.device)
                    ty0 = torch.zeros(1, dtype=torch.float32, device=self.device)

            pv = priors_all.get(vid, dict(noise_mean=1e-3, outscale_mean=1e-2, lengthscale_mean=0.1, rq_alpha_mean=1.0))

            # === likelihood with prior & constraint ===
            noise_constraint = GreaterThan(1e-5)  # Prevent noise collapse
            likelihood = gpytorch.likelihoods.GaussianLikelihood(
                noise_constraint=noise_constraint
            ).to(self.device)
            # Gamma(k=2, θ=mean/2) 
            likelihood.register_prior("noise_prior", GammaPrior(2.0, pv["noise_mean"] / 2.0), "noise")

            # === model & kernel priors ===
            model = _ExactGP1D(tx0, ty0, likelihood).to(self.device)

            # outputscale prior
            model.covar_module.register_prior(
                "outputscale_prior", GammaPrior(2.0, pv["outscale_mean"] / 2.0), "outputscale"
            )

            # lengthscale priors + constraints on additive sub-kernels
            length_prior = LogNormalPrior(np.log(pv["lengthscale_mean"]), 0.5)
            length_constraint = Interval(0.02, 2.0)

            base_k = model.covar_module.base_kernel
            add_k = getattr(base_k, "base", None)
            if add_k is not None and hasattr(add_k, "kernels"):
                for k in add_k.kernels:
                    try:
                        k.register_prior("lengthscale_prior", length_prior, "lengthscale")
                        k.lengthscale_constraint = length_constraint
                    except Exception:
                        pass
                # Give RQ alpha a mild prior
                try:
                    for k in add_k.kernels:
                        if isinstance(k, RQKernel):
                            k.register_prior("alpha_prior", LogNormalPrior(np.log(pv["rq_alpha_mean"]), 0.5), "alpha")
                            break
                except Exception:
                    pass

            # === Training ===
            model.train()
            likelihood.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=self.gp_lr)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

            print(
                f"[GP][var={vid}] start training: iters={self.gp_train_iters}, samples={len(sample_indices)}, total_points≈{total_points}")
            running_loss = None

            all_idx = sample_indices
            for it in range(self.gp_train_iters):
                if sample_batch_size is None:
                    batch_idx = all_idx
                else:
                    if len(all_idx) <= int(sample_batch_size or 0):
                        batch_idx = all_idx
                    else:
                        batch_idx = random.sample(all_idx, k=int(sample_batch_size))

                optimizer.zero_grad()
                total_loss = 0.0
                batch_points = 0

                for idx in batch_idx:
                    smp = self.train_dataset[idx]  # (L,4)
                    if isinstance(smp, (tuple, list)):
                        smp = smp[0]
                    elif isinstance(smp, dict):
                        smp = smp["sample_long"]
                    col_of = lambda rid: self.var2col[int(rid)]  # raw var id -> column id
                    col_tensor = torch.tensor(
                        [col_of(int(v.item())) for v in smp[:, 2].long()],
                        dtype=torch.long,
                        device=smp.device,
                    )
                    mask = (col_tensor == vid)
                    if not mask.any():
                        continue

                    Xs = smp[mask, 0].reshape(-1, 1).to(torch.float32).to(self.device)
                    ys = smp[mask, 1].reshape(-1).to(torch.float32).to(self.device)
                    batch_points += int(Xs.shape[0])

                    with gpytorch.settings.cholesky_jitter(1e-6):
                        model.set_train_data(inputs=Xs, targets=ys, strict=False)
                        out = model(Xs)
                        loss_s = -mll(out, ys) / max(ys.numel(), 1)
                        total_loss = (total_loss + loss_s) if not isinstance(total_loss, float) else loss_s

                if isinstance(total_loss, float) and total_loss == 0.0:
                    print(f"[GP][var={vid}][iter={it + 1}/{self.gp_train_iters}] skipped (no points in batch).")
                    continue

                total_loss.backward()
                optimizer.step()

                cur_loss = float(total_loss.item())
                running_loss = cur_loss if running_loss is None else (0.9 * running_loss + 0.1 * cur_loss)

                h = self._read_hypers(model)
                print(
                    f"[GP][var={vid}][iter={it + 1}/{self.gp_train_iters}] "
                    f"loss={cur_loss:.4f} (ema={running_loss:.4f}), "
                    f"batch_pts={batch_points}, "
                    f"noise={h['noise']:.2e}, outscale={h['outscale']:.3g}, "
                    f"Matérnℓ={h['matern_ls']:.3g}, RQℓ={h['rq_ls']:.3g}, RQα={h['rq_alpha']:.3g}"
                )

            model.eval()
            likelihood.eval()
            self.state_dicts_per_var.append(copy.deepcopy(model.state_dict()))
            print(f"[GP][var={vid}] done.")

        if self.pretrain_gp_models_saved_path is not None:
            self._save_pretrained_model(self.pretrain_gp_models_saved_path)
            report_path = os.path.splitext(self.pretrain_gp_models_saved_path)[0] + "_report.txt"
            try:
                self.save_final_report(report_path, indices_by_var)
                print(f"[GP] final report saved to: {report_path}")
            except Exception as e:
                print(f"[GP] final report failed: {e}")

    def _save_pretrained_model(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {
            "num_vars": self.num_vars,
            "state_dicts": self.state_dicts_per_var,
            "var2col": self.var2col,
            "col2var": self.col2var,
            "gp_iters": self.gp_train_iters,
            "gp_lr": self.gp_lr,
            "gp_sample_ratio": self.gp_sample_ratio,
            "version": 2,
        }
        torch.save(payload, path)

    def _load_pretrained_model(self, path):
        payload = torch.load(path, map_location=self.device)
        self.num_vars = int(payload["num_vars"])
        self.state_dicts_per_var = payload["state_dicts"]
        self.var2col = payload.get("var2col", None)
        self.col2var = payload.get("col2var", None)
        self.gp_train_iters = int(payload.get("gp_iters", self.gp_train_iters))
        self.gp_lr = float(payload.get("gp_lr", self.gp_lr))
        self.gp_sample_ratio = payload.get(
            "gp_sample_ratio", self.gp_sample_ratio
        )

    def save_final_report(self, report_path: str, indices_by_var: dict | None = None):
        """
        Write final training results to text: global config, per-variable data counts, learned key hyperparameters.
        indices_by_var: optional; if provided we also count samples/points per variable.
        """
        lines = []
        lines.append("== Gaussian Process Pretraining Report ==\n")
        lines.append(f"num_vars: {self.num_vars}\n")
        lines.append(f"gp_train_iters: {self.gp_train_iters}\n")
        lines.append(f"gp_lr: {self.gp_lr}\n")
        lines.append(f"gp_sample_ratio: {self.gp_sample_ratio}\n")
        lines.append(f"device: {self.device}\n\n")

        for vid in range(self.num_vars):
            # Build a placeholder model to read hypers
            like = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
            dummy_x = torch.zeros(1, 1, device=self.device)
            dummy_y = torch.zeros(1, device=self.device)
            model = _ExactGP1D(dummy_x, dummy_y, like).to(self.device)
            model.load_state_dict(self.state_dicts_per_var[vid], strict=False)
            model.eval()
            like.eval()

            h = self._read_hypers(model)
            outputscale = h["outscale"]
            matern_ls = h["matern_ls"]
            rq_ls = h["rq_ls"]
            rq_alpha = h["rq_alpha"]
            noise = h["noise"]

            num_samples = None
            num_points = None
            if indices_by_var is not None:
                num_samples = len(indices_by_var.get(vid, []))
                # Approximate point count (quick scan; acceptable overhead)
                pts = 0
                for idx in indices_by_var.get(vid, []):
                    smp = self.train_dataset[idx]
                    if isinstance(smp, (tuple, list)):
                        smp = smp[0]
                    elif isinstance(smp, dict):
                        smp = smp["sample_long"]
                    col_of = lambda rid: self.var2col[int(rid)]
                    col_tensor = torch.tensor(
                        [col_of(int(v.item())) for v in smp[:, 2].long()],
                        dtype=torch.long,
                        device=smp.device,
                    )
                    pts += int((col_tensor == vid).sum().item())
                num_points = pts

            lines.append(f"[var {vid}] ")
            if num_samples is not None:
                lines.append(f"samples={num_samples}, ")
            if num_points is not None:
                lines.append(f"points≈{num_points}, ")
            lines.append(
                f"noise={noise:.3e}, outscale={outputscale:.6g}, Matérnℓ={matern_ls:.6g}, RQℓ={rq_ls:.6g}, RQα={rq_alpha:.6g}\n"
            )

        os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            f.writelines(lines)

        try:
            base, _ = os.path.splitext(report_path)
            figs_dir = base + "_figs"
            self.plot_prior_per_variable(save_dir=figs_dir, num_points=512)
            print(f"[GP] prior figures saved to: {figs_dir}")
        except Exception as e:
            print(f"[GP] prior figure export failed: {e}")

    @torch.no_grad()
    def plot_prior_per_variable(self, save_dir: str, num_points: int = 512):
        """
                Plot prior distribution curves per variable:
                    - Prior mean of f(t) with ± 2σ_f band
                    - Prior mean of y(t) = f(t)+ε with ± 2σ_y band where σ_y^2 = σ_f^2 + σ_n^2
        """
        os.makedirs(save_dir, exist_ok=True)
        grid = torch.linspace(0.0, 1.0, steps=int(num_points), device=self.device).view(
            -1, 1
        )

        for vid in range(self.num_vars):
            like = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
            dummy_x = torch.zeros(1, 1, device=self.device)
            dummy_y = torch.zeros(1, device=self.device)
            model = _ExactGP1D(dummy_x, dummy_y, like).to(self.device)
            model.load_state_dict(self.state_dicts_per_var[vid], strict=False)
            model.eval()
            like.eval()

            # Prior mean and variance of latent function
            prior_mean = model.mean_module(grid).detach().cpu().view(-1).numpy()
            K = model.covar_module(grid, grid).evaluate()  # [N,N]
            prior_var_f = K.diag().clamp_min(0).detach().cpu().numpy()
            std_f = np.sqrt(prior_var_f)

            # Observation variance including noise
            noise = float(model.likelihood.noise.item())
            prior_var_y = prior_var_f + noise
            std_y = np.sqrt(prior_var_y)

            t = grid.detach().cpu().view(-1).numpy()
            mu = prior_mean

            plt.figure(figsize=(10, 4))
            # Prior of f(t)
            plt.fill_between(
                t, mu - 2 * std_f, mu + 2 * std_f, alpha=0.25, label="f(t) prior ± 2σ_f"
            )
            # Prior of y(t)
            plt.fill_between(
                t,
                mu - 2 * std_y,
                mu + 2 * std_y,
                alpha=0.15,
                label="y(t)=f+ε prior ± 2σ_y",
                color="orange",
            )
            plt.plot(t, mu, lw=1.5, label="prior mean")
            plt.title(f"Variable {vid} prior")
            plt.xlabel("normalized time")
            plt.ylabel("value")
            plt.legend()
            out_path = os.path.join(save_dir, f"var_{vid}_prior.png")
            plt.tight_layout()
            plt.savefig(out_path, dpi=200)
            plt.close()

    def _read_hypers(self, model):
        """
        Unified reader for several hyperparameters printed in logs/reports; auto-adapts to
        ScaleKernel(ModulatedKernel(Additive(Matern, RQ))).
        """
        with torch.no_grad():
            outscale = float(model.covar_module.outputscale.item())
            noise = float(model.likelihood.noise.item())

            matern_ls = float("nan")
            rq_ls = float("nan")
            rq_alpha = float("nan")

            base_k = model.covar_module.base_kernel  # -> ModulatedKernel
            try:
                # ModulatedKernel(base=AdditiveKernel([...]))
                add_k = getattr(base_k, "base", None)
                if add_k is not None and hasattr(add_k, "kernels"):
                    # Convention: index 0 = Matern, 1 = RQ
                    mat_k = add_k.kernels[0]
                    rq_k = add_k.kernels[1]
                    matern_ls = float(mat_k.lengthscale.mean().item())
                    rq_ls = float(rq_k.lengthscale.mean().item())
                    rq_alpha = float(rq_k.alpha.item())
            except Exception:
                # If structure changes in future, fall back to printing outscale/noise only.
                pass

        return dict(
            outscale=outscale,
            noise=noise,
            matern_ls=matern_ls,
            rq_ls=rq_ls,
            rq_alpha=rq_alpha,
        )

        # ================ Information gain: fantasy updates (group by var, sort by time) ================
    @torch.no_grad()
    def compute_information_gain(self, vars_idx, tt, pred_mask):
        """
        vars_idx:  [B, Lin] variable column indices 0..V-1
        tt:        [B, Lin] normalized timestamps
        pred_mask: [B, Lin] 1 indicates an unobserved prediction point; 0 observed point

        Returns:
          info_gain: [B, Lin]
            - Observed point: reduction in sum of posterior variances when adding that point.
            - Prediction point: posterior variance after all observed points have been added.
        """
        assert self.state_dicts_per_var is not None, "GP pretraining not ready"
        assert (
            hasattr(self, "_gp_model_protos") and self._gp_model_protos is not None
        ), "Model prototypes not initialized"
        B, Lin = vars_idx.shape
        device = self.device

        info_gain = torch.zeros(B, Lin, device=device, dtype=torch.float32)

        for b in range(B):
            vars_b = vars_idx[b].to(torch.long)
            tt_b = tt[b].to(torch.float32)
            pm_b = pred_mask[b].to(torch.bool)  # True = prediction point, False = observed point

            # Process each variable independently for sample b
            for vid_t in torch.unique(vars_b):
                vid = int(vid_t.item())
                if vid < 0 or (self.num_vars is not None and vid >= self.num_vars):
                    continue

                # All positions for this variable within sample b
                idxs_all = torch.nonzero(vars_b == vid, as_tuple=False).squeeze(-1)
                if idxs_all.numel() == 0:
                    continue

                # Timestamps of those positions ([Nv,1]), sorted ascending
                t_all = tt_b[idxs_all].reshape(-1, 1)
                order_all = torch.argsort(t_all[:, 0])
                idxs_sorted_all = idxs_all[order_all]
                X_all = t_all[order_all].to(device)  # [Nv,1]

                # Split into prediction vs observed within local index space
                is_pred_local = pm_b[idxs_all][order_all]  # [Nv] bool，True=pred
                is_obs_local = ~is_pred_local

                Nv = int(X_all.shape[0])
                if Nv == 0:
                    continue
                # === Build latent prior K and initialize Sigma ===
                proto = self._gp_model_protos[vid]
                with gpytorch.settings.cholesky_jitter(1e-6):
                    K = proto.covar_module(X_all, X_all).evaluate()  # [Nv, Nv]
                K = (K + K.T) * 0.5
                K.diagonal().clamp_min_(1e-12)

                Sigma = K.clone()

                # Decide which noise to use for information calculation: learned vs near-zero
                use_learned = True  # Setting False may cause instability
                sigma_eff2 = (
                    float(proto.likelihood.noise.item()) if use_learned else 1e-10
                )

                # === No observed points: prediction point information ===
                if not is_obs_local.any():
                    if is_pred_local.any():
                        pred_global_idxs = idxs_sorted_all[is_pred_local]
                        info_gain[b, pred_global_idxs] = Sigma.diag()[is_pred_local]
                    continue

                # === With observed points: rank-1 updates in time order, record per-point info gain ===
                obs_pos_sorted = torch.nonzero(is_obs_local, as_tuple=False).squeeze(
                    -1
                )  # [M]
                for pj in obs_pos_sorted.tolist():
                    col = Sigma[:, pj]  # [Nv]
                    v = float(Sigma[pj, pj].item() + sigma_eff2)  # scalar variance + noise
                    delta = float(col.dot(col) / v)  # information gain contributed by this observed point
                    gidx = int(idxs_sorted_all[pj].item())
                    info_gain[b, gidx] = max(0.0, delta)

                    # rank-1: Sigma <- Sigma - col col^T / v
                    Sigma -= torch.outer(col, col) / v
                    Sigma = (Sigma + Sigma.T) * 0.5
                    Sigma.diagonal().clamp_min_(0.0)

                # === After incorporating all observed points: assign variance to prediction points ===
                if is_pred_local.any():
                    pred_loc = torch.nonzero(is_pred_local, as_tuple=False).squeeze(-1)
                    pred_g = idxs_sorted_all[pred_loc]
                    info_gain[b, pred_g] = Sigma.diag()[pred_loc]

        return info_gain

    # ============================ Aggregation mask computation ============================
    @torch.no_grad()
    def compute_aggregation_mask(
        self,
        info_gain: torch.Tensor,  # [B, N]
        pad_mask: torch.Tensor,  # [B, N], 0 = padding
        vars_idx: torch.Tensor,  # [B, N], each in 0..V-1
        tt: torch.Tensor,  # [B, N], normalized time
    ) -> torch.Tensor:
        """
        Return mask: [B, L, N] (L = self.hyper_num_nodes)
        Logic:
            - fixed   : L divisible by V; each variable gets L/V channels; partition its timeline into
                        equal-information segments for those channels.
            - dynamic : L not required to be divisible by V; allocate channels proportional to total
                        info mass per variable (largest remainder); then equal-information segmentation.
        """
        device = info_gain.device
        dtype = info_gain.dtype
        B, N = info_gain.shape
        hyper_num_nodes = int(self.hyper_num_nodes)
        num_vars = int(self.num_vars)
        eps = (1e-12)

    # Boolean padding mask: 0 denotes padding
        pad_bool = pad_mask == 0

        mask = torch.zeros(B, hyper_num_nodes, N, device=device, dtype=dtype)

        if self.normalize_across_vars:
            info_gain = info_gain.clone()
            for batch_idx in range(B):
                for v in range(num_vars):
                    valid = (~pad_bool[batch_idx]) & (vars_idx[batch_idx] == v)
                    if not valid.any():
                        continue
                    s = float(info_gain[batch_idx, valid].mean().item())
                    if s > eps:
                        info_gain[batch_idx, valid] /= s


        if not self.allow_dynamic_hypernode_allocation:
            assert hyper_num_nodes % num_vars == 0, f"[fixed] hyper_num_nodes={hyper_num_nodes} must divide V={num_vars}"
            per_var = [hyper_num_nodes // num_vars] * num_vars  # equal channels per variable

        for batch_idx in range(B):
            # Valid (non-padded) positions for this sample
            valid_b = ~pad_bool[batch_idx]

            # Total info per variable (for dynamic allocation)
            S_per_var = torch.zeros(num_vars, device=device, dtype=dtype)
            for v in range(num_vars):
                idx_v = valid_b & (vars_idx[batch_idx] == v)
                if idx_v.any():
                    S_per_var[v] = info_gain[batch_idx, idx_v].sum()

            # Decide channels per variable for this sample
            if not self.allow_dynamic_hypernode_allocation:
                per_var_b = per_var
            else:
                # dynamic allocation
                alloc = self.largest_remainder_alloc(S_per_var, hyper_num_nodes)  # list[int], sum==L
                per_var_b = alloc


            base = 0
            var_channel_ranges = []
            for v in range(num_vars):
                n_ch = int(per_var_b[v])
                var_channel_ranges.append((base, base + n_ch))
                base += n_ch

            assert base == hyper_num_nodes

            # For each variable: partition its timeline into n_ch segments of (approximately) equal info mass.
            for v in range(num_vars):
                start_ch, end_ch = var_channel_ranges[v]
                n_ch = end_ch - start_ch
                if n_ch <= 0:
                    continue  # Variable received no channels

                idx_v = valid_b & (vars_idx[batch_idx] == v)
                if not idx_v.any():
                    continue  # Variable has no valid points in this sample

                ig_v = info_gain[batch_idx, idx_v]
                pos_global = torch.nonzero(idx_v, as_tuple=False).squeeze(-1)
                ig_sorted = ig_v
                Nv = int(ig_sorted.numel())

                # Total information mass for this variable in this sample
                S = float(ig_sorted.sum().item())

                if Nv == 0:
                    continue

                # === Segmentation strategy: equal-information segments along time ===
                segments = []  # list of (a, b) over pos_global (left-inclusive, right-exclusive)
                if S <= eps:
                    # Degenerate: near-zero info -> fallback to equal-count partitioning
                    chunk = int(math.ceil(Nv / n_ch))
                    for j in range(n_ch):
                        a = j * chunk
                        b = min(Nv, (j + 1) * chunk)
                        if a >= b:
                            break
                        segments.append((a, b))
                else:
                    # Target cumulative threshold: k/n_ch * S
                    cumsum = torch.cumsum(ig_sorted, dim=0)  # monotonic increasing
                    a = 0
                    for j in range(1, n_ch + 1):
                        target = j * (S / n_ch)
                        # find first position with cumsum >= target
                        b = int(
                            torch.searchsorted(
                                cumsum, torch.tensor(target, device=device), right=True
                            ).item()
                        )
                        b = max(b, a + 1)  # ensure non-empty
                        b = min(b, Nv)  # stay within bounds
                        segments.append((a, b))
                        a = b
                        if a >= Nv:  # stop if early exhaustion
                            break
                    # If cumulative rounding leaves remainder, append final segment
                    if len(segments) < n_ch and a < Nv:
                        segments.append((a, Nv))

                # Segment count may be < n_ch (few points or near-zero info); use first len(segments) channels.
                for j, (a, b_idx) in enumerate(segments):
                    if j >= n_ch:
                        break
                    ch = start_ch + j
                    idxs = pos_global[a:b_idx]  # slice positions for this segment
                    mask[batch_idx, ch, idxs] = 1.0  # set 1; others remain 0

        return mask

    @torch.no_grad()
    def compute_aggregation_assignment(
            self,
            info_gain: torch.Tensor,  # [B, N]
            pad_mask: torch.Tensor,  # [B, N], 0 = padding
            vars_idx: torch.Tensor,  # [B, N], each in 0..V-1
            tt: torch.Tensor,  # [B, N], normalized time
    ) -> torch.Tensor:
        """
        Compute the *hard* assignment from input positions (N) to hyper-nodes (Lout).
        Returns:
            assign: [B, N] with values in {0..Lout-1}, and -1 for padding positions.
        The logic mirrors `compute_aggregation_mask`, but writes integer channel ids
        instead of a one-hot [B, Lout, N] tensor, to save memory and allow reuse.
        """
        device = info_gain.device
        dtype = info_gain.dtype
        B, N = info_gain.shape
        Lout = int(self.hyper_num_nodes)
        V = int(self.num_vars)
        eps = 1e-12

        pad_bool = pad_mask == 0
        assign = torch.full((B, N), -1, dtype=torch.int64, device=device)  # -1 for padding

        if self.normalize_across_vars:
            info_gain = info_gain.clone()
            for b in range(B):
                for v in range(V):
                    valid = (~pad_bool[b]) & (vars_idx[b] == v)
                    if not valid.any():
                        continue
                    m = float(info_gain[b, valid].mean().item())
                    if m > eps:
                        info_gain[b, valid] /= m

        if not self.allow_dynamic_hypernode_allocation:
            assert Lout % V == 0, f"[fixed] hyper_num_nodes={Lout} must be divisible by V={V}"
            per_var_fixed = [Lout // V] * V

        for b in range(B):
            valid_b = ~pad_bool[b]
            if not valid_b.any():
                continue

            Sv = torch.zeros(V, device=device, dtype=dtype)
            for v in range(V):
                idx_v = valid_b & (vars_idx[b] == v)
                if idx_v.any():
                    Sv[v] = info_gain[b, idx_v].sum()

            if not self.allow_dynamic_hypernode_allocation:
                per_var = per_var_fixed
            else:
                per_var = self.largest_remainder_alloc(Sv, Lout)  # list[int], sum==Lout
                
            base = 0
            var_ranges = []
            for v in range(V):
                n_ch = int(per_var[v])
                var_ranges.append((base, base + n_ch))
                base += n_ch
            assert base == Lout, "[internal] channel partition mismatch"

            for v in range(V):
                start_ch, end_ch = var_ranges[v]
                n_ch = end_ch - start_ch
                if n_ch <= 0:
                    continue

                idx_v = valid_b & (vars_idx[b] == v)
                if not idx_v.any():
                    continue

                pos_global = torch.nonzero(idx_v, as_tuple=False).squeeze(-1)  # indices in [0..N-1]
                ig_v = info_gain[b, pos_global]
                Nv = int(ig_v.numel())
                if Nv == 0:
                    continue

                S = float(ig_v.sum().item())
                segments = []
                if S <= eps:
                    chunk = int(math.ceil(Nv / n_ch))
                    for j in range(n_ch):
                        a = j * chunk
                        b_i = min(Nv, (j + 1) * chunk)
                        if a >= b_i:
                            break
                        segments.append((a, b_i))
                else:
                    cumsum = torch.cumsum(ig_v, dim=0)
                    a = 0
                    for j in range(1, n_ch + 1):
                        target = j * (S / n_ch)
                        b_i = int(torch.searchsorted(cumsum, torch.tensor(target, device=device), right=True).item())
                        b_i = max(b_i, a + 1)
                        b_i = min(b_i, Nv)
                        segments.append((a, b_i))
                        a = b_i
                        if a >= Nv:
                            break
                    if len(segments) < n_ch and a < Nv:
                        segments.append((a, Nv))

                # write segment -> channel id
                for j, (a, b_i) in enumerate(segments):
                    if j >= n_ch:
                        break
                    ch = start_ch + j
                    idxs = pos_global[a:b_i]
                    assign[b, idxs] = ch

        return assign

    @torch.no_grad()
    def precompute_assign_for_dataset(self, dataset) -> list[torch.Tensor]:
        """
        Offline precomputation over *all* samples in a dataset.
        Each dataset[i] must be a (L,4) tensor: [t_norm, value, var_id, pred_mask].
        Returns:
            list of 1-D LongTensors (per-sample assignment), cpu(), dtype=int32.
        """
        out = []
        for i in range(len(dataset)):
            smp = dataset[i]
            if smp.numel() == 0:
                out.append(torch.empty(0, dtype=torch.int32))
                continue

            # Build (tt, pred_mask, vars_idx) on device; map raw var_id -> col via self.var2col
            t = smp[:, 0].view(1, -1).to(self.device)
            pred = smp[:, 3].view(1, -1).to(self.device)
            raw_vid = smp[:, 2].long().tolist()
            col_vid = [self.var2col.get(int(v), 0) for v in raw_vid]
            v = torch.tensor(col_vid, dtype=torch.long, device=self.device).view(1, -1)
            pad = torch.ones_like(pred, device=self.device)

            info = self.compute_information_gain(v, t, pred)  # [1, L]
            assign = self.compute_aggregation_assignment(
                info_gain=info, pad_mask=pad, vars_idx=v, tt=t
            ).squeeze(0)  # [L]
            out.append(assign.to("cpu", non_blocking=True).to(torch.int32))
        return out

    @torch.no_grad()
    def compute_assign_for_sample(self, sample: torch.Tensor) -> torch.Tensor:
        """
        On-demand precomputation for a *single* sample (valid/test lazy path).
        sample: (L,4) = [t_norm, value, var_id, pred_mask]
        Returns:
            1-D LongTensor assignment, cpu(), dtype=int32.
        """
        if sample.numel() == 0:
            return torch.empty(0, dtype=torch.int32)

        t = sample[:, 0].view(1, -1).to(self.device)
        pred = sample[:, 3].view(1, -1).to(self.device)
        raw_vid = sample[:, 2].long().tolist()
        col_vid = [self.var2col.get(int(v), 0) for v in raw_vid]
        v = torch.tensor(col_vid, dtype=torch.long, device=self.device).view(1, -1)
        pad = torch.ones_like(pred, device=self.device)

        info = self.compute_information_gain(v, t, pred)
        assign = self.compute_aggregation_assignment(
            info_gain=info, pad_mask=pad, vars_idx=v, tt=t
        ).squeeze(0)
        return assign.to("cpu", non_blocking=True).to(torch.int32)

    def forward(
        self,
        batch_timestamp,
        batch_var_idx,
        batch_pred_mask,
        batch_pad_mask,
    ):
        raise NotImplementedError(
            "This module no longer computes masks in the forward pass. "
            "Use `precompute_assign_for_dataset(...)` (train) or "
            "`compute_assign_for_sample(...)` (valid/test) instead."
        )
