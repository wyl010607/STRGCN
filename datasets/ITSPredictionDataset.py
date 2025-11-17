import re
import os
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from utils.parse_duration_to_seconds import parse_duration_to_seconds

from models import GaussianBasedGraphAggregationMaskGenerator

class ITSPredictionDataset(Dataset):
    """
    Dataset for irregular multivariate time-series forecasting.

    Parameters
    ----------
    data : np.ndarray, shape (N, 4)
        Long-form array with columns: (ts_id, var_id, timestamp, value).
    history_len : int | str
        Number of time units regarded as the history window within each sample.
        Can be an integer (seconds) or a pandas-compatible duration string (e.g., "12h").
    forecast_len : int | str
        Number of time units regarded as the forecast window within each sample.
        Can be an integer (seconds) or a pandas-compatible duration string.
    time_end : int | str | None, default=None
        Right-open cutoff per `ts_id`. Rows with `timestamp > min_timestamp + time_end`
        for that `ts_id` are discarded. If None, no per-series cutoff is applied.
    use_time_chunk : bool, default=False
        If True, split each `ts_id` into chunks of length `history_len + forecast_len`.
        If False, treat the whole series as a single sample when possible.
    reset_tp_after_chunk : bool, default=True
        If True, timestamps are normalized relative to the chunk start (local normalization).
        If False, timestamps are normalized globally (relative to the series start / global max).
    chunk_stride : int | str | None, default=None
        Step size for chunking. If None, defaults to `forecast_len` (sliding window).
    normize_time_method : str, default="relative"
        Method for normalizing timestamps:
            - "relative": normalize timestamps by history + forecast length
            - "global": normalize timestamps by the global maximum timestamp
    output_style : {"long", "matrix"}, default="long"
        Output style for the collate function:
            - "long": concatenated per-variable sequences (current behavior).
            - "matrix": padded time-by-variable matrices aligned with IrrMasking outputs.
    """

    def __init__(
        self,
        data: np.ndarray,
        type: str,  # "train", "valid", or "test"
        history_len: int | str,
        forecast_len: int | str,
        time_end: int | str | None = None,
        use_time_chunk: bool = False,
        reset_tp_after_chunk: bool = True,
        chunk_stride: int | str | None = None,
        normize_time_method: str = "relative",
        output_style: str = "long",
        use_cache: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.global_args = kwargs.get("global_args")
        self.data_name = self.global_args.get("dataset_name", "unknown")
        self.device =self.global_args.get("device", "cuda")
        self.type = type  # "train", "valid", or "test"
        if self.type == "train":
            self.global_args["train_dataset"] = self
        self.history_len_sec = parse_duration_to_seconds(history_len)
        self.forecast_len_sec = parse_duration_to_seconds(forecast_len)
        self.time_end = (
            None if time_end is None else parse_duration_to_seconds(time_end)
        )
        self.use_time_chunk = bool(use_time_chunk)
        self.reset_tp_after_chunk = bool(reset_tp_after_chunk)
        self.chunk_stride = (
            None if chunk_stride is None else parse_duration_to_seconds(chunk_stride)
        )
        self.normize_time_method = normize_time_method.lower()  # "relative" or "global"
        assert self.normize_time_method in ["relative", "global"]

        self.output_style = output_style.lower()  # "long" or "matrix"

        # Whether to enable precomputed aggregation mask
        self.enable_agg_mask = bool(kwargs.get("enable_agg_mask", False))
        self.num_hyper_nodes_per_var = int(
            kwargs.get("num_hyper_nodes_per_var", self.global_args.get("num_hyper_nodes_per_var", 12) or 12)
        )
        self.global_args["num_hyper_nodes_per_var"] = self.num_hyper_nodes_per_var
        # GP/aggregator hyperparams
        self.agg_gp_train_iters = int(kwargs.get("agg_gp_train_iters", 100))
        self.agg_gp_lr = float(kwargs.get("agg_gp_lr", 0.05))
        self.agg_sample_ratio = float(kwargs.get("agg_sample_ratio", 0.25))
        self.agg_dynamic = bool(kwargs.get("agg_dynamic", False))
        self.agg_norm_across_vars = bool(kwargs.get("agg_norm_across_vars", False))

        self._config = {
            "data_name": self.data_name,
            "history_len": history_len,
            "forecast_len": forecast_len,
            "time_end": time_end,
            "use_time_chunk": use_time_chunk,
            "reset_tp_after_chunk": reset_tp_after_chunk,
            "chunk_stride": chunk_stride,
            "normize_time_method": normize_time_method,
            "output_style": output_style,
            "type": type,
            "enable_agg_mask": bool(self.enable_agg_mask),
            "num_hyper_nodes_per_var": int(self.num_hyper_nodes_per_var),
            "agg_gp_train_iters": int(self.agg_gp_train_iters),
            "agg_gp_lr": float(self.agg_gp_lr),
            "agg_sample_ratio": float(self.agg_sample_ratio),
            "agg_dynamic": bool(self.agg_dynamic),
            "agg_norm_across_vars": bool(self.agg_norm_across_vars),
        }
        self.cache_dir = "./cache"
        self.cache_path = os.path.join(self.cache_dir, f"{self._make_cache_name()}.pt")

        loaded_from_cache = False
        if use_cache:
            loaded_from_cache = self.load_from_cache()
            if loaded_from_cache:
                return
        if not loaded_from_cache:
            df = pd.DataFrame(
                data, columns=["ts_id", "var_id", "timestamp", "value"]
            ).dropna()
            df[["ts_id", "var_id"]] = df[["ts_id", "var_id"]].astype(int)

            if self.time_end is not None:
                filtered = []
                for ts_id, g in df.groupby("ts_id", group_keys=False):
                    mask = g["timestamp"] <= g["timestamp"].min() + self.time_end
                    filtered.append(g.loc[mask])
                df = pd.concat(filtered, ignore_index=True)

            df["timestamp"] = df.groupby("ts_id")["timestamp"].transform(
                lambda s: s - s.min()
            )  # start each ts_id from 0

            self._global_time_max = float(
                self.time_end if self.time_end is not None else df["timestamp"].max()
            )
            if self.normize_time_method == "global":
                # Normalize timestamps globally
                self.time_norm_scale = (
                    self.time_end if self.time_end is not None else self._global_time_max
                )
            else:  # self.normize_time_method == "relative":
                self.time_norm_scale = self.history_len_sec + self.forecast_len_sec

            if not np.isfinite(self._global_time_max) or self._global_time_max <= 0:
                self._global_time_max = 1.0  # avoid division by zero
            df.sort_values(["ts_id", "timestamp"], inplace=True, ignore_index=True)

            self._all_var_ids = np.sort(df["var_id"].astype(int).unique()).tolist()
            self.var2col = {
                int(v): i for i, v in enumerate(self._all_var_ids)
            }  # 0..enc_in-1

            # split into samples
            self._samples = []
            for ts_id, g in df.groupby("ts_id", sort=False):
                self._samples.extend(self._split_one_series(g.reset_index(drop=True)))

            (
                self.max_observed_tp_len,
                self.max_target_forecast_tp_len,
            ) = self._compute_max_tp_lens()

            self.global_args["max_observed_tp_len"] = (
                self.max_observed_tp_len
                if self.max_observed_tp_len > self.global_args.get("max_observed_tp_len", 0)
                else self.global_args["max_observed_tp_len"]
            )
            self.global_args["max_target_forecast_tp_len"] = (
                self.max_target_forecast_tp_len
                if self.max_target_forecast_tp_len
                > self.global_args.get("max_target_forecast_tp_len", 0)
                else self.global_args["max_target_forecast_tp_len"]
            )
            self.global_args["num_of_vals"] = len(self._all_var_ids)
            self.global_args["time_norm_scale"] = self.time_norm_scale
            self.global_args["history_len_sec"] = self.history_len_sec
            self.global_args["forecast_len_sec"] = self.forecast_len_sec

            self._assign_list = None

        self.hyper_num_nodes = self.num_hyper_nodes_per_var * len(self._all_var_ids)
        # Prepare path of GP hyper cache: train vs eval/test
        if "train_dataset" in self.global_args and self.global_args["train_dataset"] is not None:
            tr_ds = self.global_args["train_dataset"]
            gp_cache_from_train = (
                f"./cache/{tr_ds._make_gp_cache_name()}.pt"
            )
        else:
            gp_cache_from_train = (
                f"./cache/{self._make_gp_cache_name()}.pt"
            )

        self._aggregator = None
        if self.enable_agg_mask:
            # Create aggregator: for train, pretrain if needed; for eval/test, only load from train cache
            if self.type == "train":
                self._aggregator = GaussianBasedGraphAggregationMaskGenerator(
                    train_dataset=self,
                    hyper_num_nodes=self.hyper_num_nodes,
                    pretrain_gp_models_saved_path=gp_cache_from_train,
                    gp_train_iters=self.agg_gp_train_iters,
                    gp_lr=self.agg_gp_lr,
                    gp_sample_ratio=self.agg_sample_ratio,
                    allow_dynamic_hypernode_allocation=self.agg_dynamic,
                    normalize_across_vars=self.agg_norm_across_vars,
                    device=self.device,
                )
            else:
                self._aggregator = GaussianBasedGraphAggregationMaskGenerator(
                    train_dataset=None,
                    hyper_num_nodes=self.hyper_num_nodes,
                    pretrain_gp_models_saved_path=gp_cache_from_train,
                    gp_train_iters=self.agg_gp_train_iters,
                    gp_lr=self.agg_gp_lr,
                    gp_sample_ratio=self.agg_sample_ratio,
                    allow_dynamic_hypernode_allocation=self.agg_dynamic,
                    normalize_across_vars=self.agg_norm_across_vars,
                    device=self.device,
                )

            if (not loaded_from_cache) or (getattr(self, "_assign_list", None) is None):
                self._assign_list = []
                for smp in self._samples:
                    if smp.numel() == 0:
                        self._assign_list.append(torch.empty(0, dtype=torch.int32))
                        continue
                    # Build (vars_idx, tt, pred_mask) per sample, consistent with aggregator's mapping
                    t = smp[:, 0].view(1, -1).to(self.device)
                    pred = smp[:, 3].view(1, -1).to(self.device)
                    raw_vid = smp[:, 2].long().tolist()
                    col_vid = [self._aggregator.var2col.get(int(v), 0) for v in raw_vid]
                    v = torch.tensor(col_vid, dtype=torch.long, device=self.device).view(1, -1)
                    pad = torch.ones_like(pred, device=self.device)
                    # Information gain
                    info = self._aggregator.compute_information_gain(v, t, pred)  # [1, L]
                    assign = self._aggregator.compute_aggregation_assignment(
                        info_gain=info, pad_mask=pad, vars_idx=v, tt=t
                    ).squeeze(0)
                    self._assign_list.append(assign.to("cpu", non_blocking=True).to(torch.int32))

        if use_cache:
            self.save_cache()

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx: int):
        # Keep backward compatibility: if aggregation is disabled, return tensor only.
        if not self.enable_agg_mask or (getattr(self, "_assign_list", None) is None):
            return self._samples[idx]
        # Train: assign is precomputed; Valid/Test: may be None (lazy compute later)
        return self._samples[idx], self._assign_list[idx]

    def collate_fn(self, batch: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        if self.output_style == "matrix":
            return self.matrix_collate_fn(batch)
        elif self.output_style == "long":
            return self.long_collate_fn(batch)
        else:
            raise ValueError(f"Unknown output style: {self.output_style}")

    def long_collate_fn_old(self, batch: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        vars_num = len(self._all_var_ids)
        per_var_ts, per_var_val, per_var_mask = (
            [[] for _ in range(vars_num)],
            [[] for _ in range(vars_num)],
            [[] for _ in range(vars_num)],
        )

        # Accept both sample-only and (sample, assign); ignore assign here.
        batch_only_samples = [b[0] if (isinstance(b, (tuple, list)) and len(b) == 2) else b for b in batch]

        for sample in batch_only_samples:  # sample shape (L, 4)
            for v_idx, v_id in enumerate(self._all_var_ids):
                rows = sample[sample[:, 2] == float(v_id)]
                if rows.numel() == 0:
                    rows = torch.empty((0, 4), dtype=sample.dtype, device=sample.device)
                per_var_ts[v_idx].append(rows[:, 0])
                per_var_val[v_idx].append(rows[:, 1])
                per_var_mask[v_idx].append(rows[:, 3])

        pad_ts, pad_val, pad_mask = [], [], []
        for v_idx in range(vars_num):
            pad_ts.append(
                pad_sequence(per_var_ts[v_idx], batch_first=True, padding_value=-1.0)
            )
            pad_val.append(
                pad_sequence(per_var_val[v_idx], batch_first=True, padding_value=0.0)
            )
            pad_mask.append(
                pad_sequence(per_var_mask[v_idx], batch_first=True, padding_value=0.0)
            )

        batch_timestamp = torch.cat(pad_ts, dim=1)  # (B, total_L)
        batch_value = torch.cat(pad_val, dim=1)  # (B, total_L)
        batch_pred_mask = torch.cat(pad_mask, dim=1)  # (B, total_L)
        batch_pad_mask = (batch_timestamp >= 0).float()  # (0 = missing, 1 = present)

        batch_var_id = torch.cat(
            [
                torch.full(
                    (pad_ts[v_idx].shape[1],),
                    self._all_var_ids[v_idx],
                    dtype=torch.long,
                    device=batch_timestamp.device,
                )
                for v_idx in range(vars_num)
            ]
        )

        B, total_L = batch_timestamp.shape
        batch_time_id = torch.full_like(batch_timestamp, -1, dtype=torch.long)
        for b in range(B):
            ts_valid = batch_timestamp[b][batch_pad_mask[b] == 1]
            if ts_valid.numel() == 0:
                continue
            uniq_ts, inverse = torch.unique(ts_valid, sorted=True, return_inverse=True)
            idx = (batch_pad_mask[b] == 1).nonzero(as_tuple=False).squeeze(-1)
            batch_time_id[b, idx] = inverse

        out = {
            "batch_timestamp": batch_timestamp,
            "batch_value": batch_value,
            "batch_var_idx": batch_var_id,
            "batch_pred_mask": batch_pred_mask,
            "batch_pad_mask": batch_pad_mask,
            "batch_time_id": batch_time_id,
        }

        return out

    def long_collate_fn(self, batch: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Each sample is a (L_i, 4) tensor: [t_norm, value, var_id, pred_mask]
        No variable alignment is performed. Outputs are flattened per-sample and
        padded to the batch maximum length N_obs_max. Optionally returns batch_agg_mask.
        """
        qscale = 1e9  # quantization to suppress float jitter before unique

        # Accept both sample-only and (sample, assign)
        pairs = []
        for item in batch:
            if isinstance(item, (tuple, list)) and len(item) == 2:
                pairs.append(item)
            else:
                pairs.append((item, None))

        ts_list, val_list, vid_list = [], [], []
        pm_list, pad_list, tid_list = [], [], []

        for sample, _assign in pairs:
            t_norm = sample[:, 0]
            x_val = sample[:, 1]
            raw_var_idx = sample[:, 2].long()
            var_idx = torch.tensor(
                [self.var2col[int(v.item())] for v in raw_var_idx],
                dtype=torch.long,
                device=raw_var_idx.device,
            )
            pred_msk = sample[:, 3]  # 0/1

            if t_norm.numel() == 0:
                ts_list.append(torch.zeros(0, dtype=torch.float32))
                val_list.append(torch.zeros(0, dtype=torch.float32))
                vid_list.append(torch.zeros(0, dtype=torch.long))
                pm_list.append(torch.zeros(0, dtype=torch.float32))
                pad_list.append(torch.zeros(0, dtype=torch.float32))
                tid_list.append(torch.zeros(0, dtype=torch.long))
                continue

            t_q = torch.round(t_norm * qscale) / qscale
            _, inv = torch.unique(t_q, sorted=True, return_inverse=True)

            pad_mask = torch.ones_like(x_val, dtype=torch.float32)

            ts_list.append(t_norm.to(torch.float32))
            val_list.append(x_val.to(torch.float32))
            vid_list.append(var_idx.to(torch.long))
            pm_list.append(pred_msk.to(torch.float32))
            pad_list.append(pad_mask)
            tid_list.append(inv.to(torch.long))

        batch_timestamp = pad_sequence(ts_list, batch_first=True, padding_value=0.0)
        batch_value = pad_sequence(val_list, batch_first=True, padding_value=0.0)
        batch_var_idx = pad_sequence(vid_list, batch_first=True, padding_value=0)
        batch_pred_mask = pad_sequence(pm_list, batch_first=True, padding_value=0.0)
        batch_pad_mask = pad_sequence(pad_list, batch_first=True, padding_value=0.0)
        batch_time_id = pad_sequence(tid_list, batch_first=True, padding_value=-1)

        out = {
            "batch_timestamp": batch_timestamp,  # (B, N_obs_max)
            "batch_value": batch_value,          # (B, N_obs_max)
            "batch_var_idx": batch_var_idx,      # (B, N_obs_max)
            "batch_pred_mask": batch_pred_mask,  # (B, N_obs_max)
            "batch_pad_mask": batch_pad_mask,    # (B, N_obs_max)
            "batch_time_id": batch_time_id,      # (B, N_obs_max)
        }

        if self.enable_agg_mask:
            B, Nmax = batch_pad_mask.shape
            Lout = int(self.hyper_num_nodes)
            batch_agg_mask = torch.zeros(B, Lout, Nmax, dtype=torch.float32, device=batch_pad_mask.device)

            for b, (sample, assign) in enumerate(pairs):
                # Number of valid observations in this padded row
                Li = int((batch_pad_mask[b] > 0).sum().item())
                if Li == 0:
                    continue
                # Convert to one-hot over first Li positions
                assign = assign[:Li].to(torch.int64)
                valid = assign >= 0
                if valid.any():
                    rows = assign[valid]
                    cols = torch.arange(Li, device=batch_agg_mask.device)[valid]
                    batch_agg_mask[b].index_put_((rows, cols), torch.ones_like(rows, dtype=torch.float32), accumulate=False)

            out["batch_agg_mask"] = batch_agg_mask  # (B, Lout, Nmax)

        return out

    @torch.no_grad()
    def matrix_collate_fn(self, batch: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        num_vars = len(self._all_var_ids)
        device = batch[0].device if isinstance(batch[0], torch.Tensor) else batch[0][0].device
        dtype = batch[0].dtype if isinstance(batch[0], torch.Tensor) else batch[0][0].dtype

        # Accept both sample-only and (sample, assign); ignore assign for matrix mode
        only_samples = [b[0] if (isinstance(b, (tuple, list)) and len(b) == 2) else b for b in batch]

        obs_t_list, obs_v_list, obs_m_list = [], [], []
        fut_t_list, fut_v_list, fut_m_list = [], [], []

        qscale = 1e9

        for sample in only_samples:
            ts = sample[:, 0]
            val = sample[:, 1]
            vid = sample[:, 2].long()
            pmk = sample[:, 3]

            hist_mask = pmk == 0
            fut_mask = pmk == 1

            # ---- Historical ----
            ts_h, val_h, vid_h = ts[hist_mask], val[hist_mask], vid[hist_mask]
            if ts_h.numel() == 0:
                obs_t_list.append(ts_h)
                obs_v_list.append(torch.zeros(0, num_vars, dtype=dtype, device=device))
                obs_m_list.append(torch.zeros(0, num_vars, dtype=dtype, device=device))
            else:
                tsq = torch.round(ts_h * qscale) / qscale
                t_u, inv = torch.unique(tsq, sorted=True, return_inverse=True)
                Lh = t_u.shape[0]
                vals = torch.zeros(Lh, num_vars, dtype=dtype, device=device)
                mask = torch.zeros(Lh, num_vars, dtype=dtype, device=device)
                col = torch.tensor([self.var2col[int(v.item())] for v in vid_h], dtype=torch.long, device=device)
                mask.index_put_((inv, col), torch.ones_like(val_h), accumulate=False)
                vals.index_put_((inv, col), val_h, accumulate=False)
                obs_t_list.append(t_u)
                obs_v_list.append(vals)
                obs_m_list.append(mask)

            # ---- Future ----
            ts_f, val_f, vid_f = ts[fut_mask], val[fut_mask], vid[fut_mask]
            if ts_f.numel() == 0:
                fut_t_list.append(ts_f)
                fut_v_list.append(torch.zeros(0, num_vars, dtype=dtype, device=device))
                fut_m_list.append(torch.zeros(0, num_vars, dtype=dtype, device=device))
            else:
                tsq = torch.round(ts_f * qscale) / qscale
                t_u, inv = torch.unique(tsq, sorted=True, return_inverse=True)
                Lf = t_u.shape[0]
                vals = torch.zeros(Lf, num_vars, dtype=dtype, device=device)
                mask = torch.zeros(Lf, num_vars, dtype=dtype, device=device)
                col = torch.tensor([self.var2col[int(v.item())] for v in vid_f], dtype=torch.long, device=device)
                mask.index_put_((inv, col), torch.ones_like(val_f), accumulate=False)
                vals.index_put_((inv, col), val_f, accumulate=False)
                fut_t_list.append(t_u)
                fut_v_list.append(vals)
                fut_m_list.append(mask)

        observed_tp = pad_sequence(obs_t_list, batch_first=True)
        observed_data = pad_sequence(obs_v_list, batch_first=True)
        observed_mask = pad_sequence(obs_m_list, batch_first=True)
        target_forecast_tp = pad_sequence(fut_t_list, batch_first=True)
        target_forecast_data = pad_sequence(fut_v_list, batch_first=True)
        target_forecast_mask = pad_sequence(fut_m_list, batch_first=True)

        return {
            "observed_data": observed_data,
            "observed_tp": observed_tp,
            "observed_mask": observed_mask,
            "target_forecast_data": target_forecast_data,
            "target_forecast_tp": target_forecast_tp,
            "target_forecast_mask": target_forecast_mask,
        }

    # -------------------- unchanged --------------------
    def _split_one_series(self, g: pd.DataFrame) -> list[torch.Tensor]:
        step = self.history_len_sec + self.forecast_len_sec
        t_end = float(g["timestamp"].max())

        if self.use_time_chunk:
            stride = self.chunk_stride if (self.chunk_stride is not None) else self.forecast_len_sec
            start_times = np.arange(0, t_end - self.forecast_len_sec + 1, stride)
        else:
            if (t_end) < self.history_len_sec:
                return []
            start_times = [0]

        samples = []
        for st in start_times:
            et = st + step
            df_chunk = g[(g["timestamp"] >= st) & (g["timestamp"] < et)].copy()
            if df_chunk.empty:
                continue

            df_chunk["pred_mask"] = (df_chunk["timestamp"] >= st + self.history_len_sec).astype(float)
            if df_chunk["pred_mask"].sum() == 0 or df_chunk["pred_mask"].sum() == len(df_chunk):
                continue

            if self.reset_tp_after_chunk:
                rel_ts = df_chunk["timestamp"] - st
            else:
                rel_ts = df_chunk["timestamp"]
            norm_ts = rel_ts / self.time_norm_scale
            df_chunk["timestamp"] = norm_ts

            tensor_sample = torch.tensor(
                df_chunk[["timestamp", "value", "var_id", "pred_mask"]].values,
                dtype=torch.float32,
            )
            samples.append(tensor_sample)

        return samples

    @torch.no_grad()
    def _compute_max_tp_lens(self):
        qscale = 1e9
        max_hist, max_forecast = 0, 0

        for sample in self._samples:
            if sample.numel() == 0:
                continue

            t_all = sample[:, 0]
            m_all = sample[:, 3]

            t_hist = t_all[m_all == 0]
            if t_hist.numel() > 0:
                tsq = torch.round(t_hist * qscale) / qscale
                Lh = int(torch.unique(tsq, sorted=True).shape[0])
                max_hist = max(max_hist, Lh)

            t_fore = t_all[m_all == 1]
            if t_fore.numel() > 0:
                tsq = torch.round(t_fore * qscale) / qscale
                Lf = int(torch.unique(tsq, sorted=True).shape[0])
                max_forecast = max(max_forecast, Lf)

        return max_hist, max_forecast

    def _make_cache_name(self) -> str:
        parts = [
            re.sub(r"\W+", "", str(self.data_name)),
            f"hist{str(self._config['history_len'])}",
            f"fore{str(self._config['forecast_len'])}",
            f"end{str(self._config['time_end'])}",
            f"chunk{str(self._config['use_time_chunk'])}",
            f"stride{str(self._config['chunk_stride'])}",
            f"reset{str(self._config['reset_tp_after_chunk'])}",
            f"norm-{str(self._config['normize_time_method'])}",
            f"style-{str(self._config['output_style'])}",
            f"type-{str(self._config.get('type'))}",
            # NEW: add agg config to the cache name to avoid mismatches
            f"agg{int(bool(self._config['enable_agg_mask']))}",
            f"Lout{int(self._config['num_hyper_nodes_per_var'])}",
            f"it{int(self._config['agg_gp_train_iters'])}",
            f"lr{float(self._config['agg_gp_lr'])}",
            f"bs{float(self._config['agg_sample_ratio'])}",
            f"dyn{int(bool(self._config['agg_dynamic']))}",
            f"normv{int(bool(self._config['agg_norm_across_vars']))}",
        ]
        return "its_pred_ds__" + "__".join(map(str, parts))

    def _make_gp_cache_name(self) -> str:
        parts = [
            re.sub(r"\W+", "", str(self.data_name)),
            f"hist{str(self._config['history_len'])}",
            f"fore{str(self._config['forecast_len'])}",
            f"end{str(self._config['time_end'])}",
            f"chunk{str(self._config['use_time_chunk'])}",
            f"stride{str(self._config['chunk_stride'])}",
            f"reset{str(self._config['reset_tp_after_chunk'])}",
            f"norm-{str(self._config['normize_time_method'])}",
            f"it{int(self._config['agg_gp_train_iters'])}",
            f"lr{float(self._config['agg_gp_lr'])}",
            f"bs{float(self._config['agg_sample_ratio'])}",
            f"dyn{int(bool(self._config['agg_dynamic']))}",
            f"normv{int(bool(self._config['agg_norm_across_vars']))}",
        ]
        return "cached_gp_models__" + "__".join(map(str, parts))

    def load_from_cache(self) -> bool:
        if self.cache_path is None or not os.path.exists(self.cache_path):
            return False
        try:
            payload = torch.load(self.cache_path, map_location="cpu")
        except Exception:
            return False
        if not isinstance(payload, dict) or payload.get("config") != self._config:
            return False

        self._samples = payload["samples"]
        self._all_var_ids = list(payload["all_var_ids"])
        self.var2col = dict(payload["var2col"])
        self.time_norm_scale = float(payload["time_norm_scale"])
        self._global_time_max = float(payload["global_time_max"])
        self.max_observed_tp_len = int(payload["max_observed_tp_len"])
        self.max_target_forecast_tp_len = int(payload["max_target_forecast_tp_len"])
        self._assign_list = payload.get("assign_list", None)
        self.global_args["max_observed_tp_len"] = max(
            self.max_observed_tp_len, self.global_args.get("max_observed_tp_len", 0)
        )
        self.global_args["max_target_forecast_tp_len"] = max(
            self.max_target_forecast_tp_len,
            self.global_args.get("max_target_forecast_tp_len", 0),
        )
        self.global_args["num_of_vals"] = len(self._all_var_ids)
        self.hyper_num_nodes = self.num_hyper_nodes_per_var * len(self._all_var_ids)
        self.global_args["time_norm_scale"] = self.time_norm_scale
        self.global_args["history_len_sec"] = float(self._config["history_len"])
        self.global_args["forecast_len_sec"] = float(self._config["forecast_len"])
        return True

    def save_cache(self) -> None:
        os.makedirs(self.cache_dir, exist_ok=True)
        payload = {
            "config": self._config,
            "samples": [t.detach().cpu() for t in self._samples],
            "all_var_ids": self._all_var_ids,
            "var2col": self.var2col,
            "time_norm_scale": float(self.time_norm_scale),
            "global_time_max": float(self._global_time_max),
            "max_observed_tp_len": int(self.max_observed_tp_len),
            "max_target_forecast_tp_len": int(self.max_target_forecast_tp_len),
        }
        if getattr(self, "_assign_list", None) is not None:
            payload["assign_list"] = [
                (a.detach().cpu() if isinstance(a, torch.Tensor) else None)
                for a in self._assign_list
            ]
        torch.save(payload, self.cache_path)
