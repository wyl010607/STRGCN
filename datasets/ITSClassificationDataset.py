# its_classification_dataset.py
import os
import re
import numpy as np
import pandas as pd
from sympy import N
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from models import GaussianBasedGraphAggregationMaskGenerator


class ITSClassificationDataset(Dataset):
    """
    Irregular multivariate time-series dataset for CLASSIFICATION


    """

    def __init__(
        self,
        data: dict,
        type: str,                         # "train" | "valid" | "test"
        output_style: str = "long",
        time_scale: float = 2880.0,
        drop_empty_series: bool = True,
        use_cache: bool = True,
        enable_agg_mask: bool = False,
        num_hyper_nodes_per_var: int = 12,
        agg_gp_train_iters: int = 100,
        agg_gp_lr: float = 0.05,
        agg_sample_ratio: float = 0.25,
        agg_dynamic: bool = False,
        agg_norm_across_vars: bool = False,
        global_args: dict | None = None,
        *args,
        **kwargs,
    ):
        super().__init__()
        assert type in {"train", "valid", "test"}
        self.split = type
        self.output_style = output_style.lower()
        assert self.output_style in {"long", "matrix"}
        self.time_scale = float(time_scale)
        self.drop_empty_series = bool(drop_empty_series)

        self.global_args = global_args
        self.data_name = self.global_args.get("dataset_name", "p12")
        self.device = self.global_args.get("device", "cuda")

        self.use_cache = bool(use_cache)
        self.enable_agg_mask = bool(enable_agg_mask)
        self.num_hyper_nodes_per_var = int(num_hyper_nodes_per_var)
        self.agg_gp_train_iters = int(agg_gp_train_iters)
        self.agg_gp_lr = float(agg_gp_lr)
        self.agg_sample_ratio = float(agg_sample_ratio)
        self.agg_dynamic = bool(agg_dynamic)
        self.agg_norm_across_vars = bool(agg_norm_across_vars)

        dynamic_key = {"train": "train_data", "valid": "valid_data", "test": "test_data"}[type]
        tsids_key   = {"train": "train_ts_ids", "valid": "valid_ts_ids", "test": "test_ts_ids"}[type]
        static_key  = {"train": "train_static_data", "valid": "valid_static_data", "test": "test_static_data"}[type]
        label_key   = {"train": "train_label", "valid": "valid_label", "test": "test_label"}[type]

        x_long = np.asarray(data[dynamic_key], dtype=np.float64)
        self.split_ts_ids = np.asarray(data[tsids_key], dtype=np.int64) if (data[tsids_key] is not None) else np.unique(x_long[:,0])
        self.split_static = np.asarray(data.get(static_key, np.zeros((0, 0))), dtype=np.float64)  # (N_split, D)
        self.split_label = np.asarray(data.get(label_key, np.zeros((0,), dtype=np.int64)), dtype=np.int64)  # (N_split,)
        assert len(self.split_ts_ids) == len(self.split_static) == len(self.split_label), \
            f"Split arrays length mismatch: ts={len(self.split_ts_ids)}, static={len(self.split_static)}, label={len(self.split_label)}"
        df = pd.DataFrame(x_long, columns=["ts_id", "var_id", "timestamp", "value"]).dropna()
        df[["ts_id", "var_id"]] = df[["ts_id", "var_id"]].astype(int)

        df["timestamp"] = df.groupby("ts_id")["timestamp"].transform(lambda s: s - s.min())
        df["timestamp"] = df["timestamp"] / self.time_scale

        self._all_var_ids = np.sort(df["var_id"].astype(int).unique()).tolist()
        self.num_vars = len(self._all_var_ids)
        self.var2col = {int(v): i for i, v in enumerate(self._all_var_ids)}  # var_id -> 0..V-1
        self.hyper_num_nodes = self.num_hyper_nodes_per_var * len(self._all_var_ids)

        self.global_args["num_of_vals"] = int(len(self._all_var_ids))
        self.global_args["num_hyper_nodes_per_var"] = int(self.num_hyper_nodes_per_var)
        self.global_args["static_in_dim"] = int(self.split_static.shape[1]) if self.split_static.size > 0 else 0
        self.global_args.setdefault("task_name", "classification")
        self.global_args.setdefault("dataset_name", self.data_name)

        self._samples, self._statics, self._labels, self._ts_ids = [], [], [], []
        groups = dict(tuple(df.groupby("ts_id", sort=False)))
        for i, ts in enumerate(self.split_ts_ids.tolist()):
            g = groups.get(int(ts), None)
            if g is None or g.empty:
                if self.drop_empty_series:
                    continue
                g = pd.DataFrame(columns=["ts_id", "var_id", "timestamp", "value"])

            g = g.sort_values(["timestamp", "var_id"], kind="mergesort", ignore_index=True)
            if g.shape[0] == 0 and self.drop_empty_series:
                continue

            # (L,3) -> [t_norm_hipatch, value, var_id]
            sample = torch.tensor(g[["timestamp", "value", "var_id"]].values, dtype=torch.float32)

            static_vec = (
                torch.tensor(self.split_static[i], dtype=torch.float32)
                if self.split_static.size > 0 else torch.zeros(0, dtype=torch.float32)
            )
            label = int(self.split_label[i])

            self._samples.append(sample)
            self._statics.append(static_vec)
            self._labels.append(label)
            self._ts_ids.append(int(ts))

        self._config = {
            "data_name": str(self.data_name),
            "split": str(self.split),
            "output_style": str(self.output_style),
            "time_scale": float(self.time_scale),
            "num_vars": int(len(self._all_var_ids)),
            "static_dim": int(self.split_static.shape[1]) if self.split_static.size > 0 else 0,
            "n_samples": int(len(self._samples)),
            # aggregation
            "enable_agg_mask": bool(self.enable_agg_mask),
            "num_hyper_nodes_per_var": int(self.num_hyper_nodes_per_var),
            "agg_gp_train_iters": int(self.agg_gp_train_iters),
            "agg_gp_lr": float(self.agg_gp_lr),
            "agg_sample_ratio": float(self.agg_sample_ratio),
            "agg_dynamic": bool(self.agg_dynamic),
            "agg_norm_across_vars": bool(self.agg_norm_across_vars),
        }
        self.cache_dir = "./cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_path = os.path.join(self.cache_dir, self._make_cache_name())

        if self.use_cache:
            loaded = self.load_from_cache()
            if loaded:
                V = len(self._all_var_ids)
                self.global_args["num_of_vals"] = int(V)
                nhpv = getattr(self, "num_hyper_nodes_per_var", None)
                if nhpv is None:
                    nhpv = int(round(self.hyper_num_nodes / max(1, V)))
                    self.num_hyper_nodes_per_var = nhpv
                self.global_args["num_hyper_nodes_per_var"] = int(self.num_hyper_nodes_per_var)
                self.global_args["static_in_dim"] = int(self.split_static.shape[1]) if self.split_static.size > 0 else 0
                self.global_args.setdefault("task_name", "classification")
                self.global_args.setdefault("dataset_name", self.data_name)
                return

        self._assign_list = None
        if self.enable_agg_mask:
            if self.split == "train":
                self.global_args["cls_train_dataset"] = self
                gp_cache_from_train = os.path.join(self.cache_dir, self._make_gp_cache_name())
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
                tr = self.global_args.get("cls_train_dataset", None)
                gp_cache_from_train = os.path.join(self.cache_dir, tr._make_gp_cache_name()) if tr else os.path.join(self.cache_dir, self._make_gp_cache_name())
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

            self._assign_list = []
            for smp in self._samples:
                if smp.numel() == 0:
                    self._assign_list.append(torch.empty(0, dtype=torch.int32))
                    continue
                t = smp[:, 0].view(1, -1).to(self.device)            # [1, L]
                pred = torch.ones_like(t, device=self.device)        # [1, L]
                raw_vid = smp[:, 2].long().tolist()
                col_vid = [
                    (self._aggregator.var2col.get(int(v), 0) if hasattr(self._aggregator, "var2col") else self.var2col.get(int(v), 0))
                    for v in raw_vid
                ]
                v = torch.tensor(col_vid, dtype=torch.long, device=self.device).view(1, -1)
                pad = torch.ones_like(pred, device=self.device)
                info = self._aggregator.compute_information_gain(v, t, pred)  # [1, L]
                assign = self._aggregator.compute_aggregation_assignment(
                    info_gain=info, pad_mask=pad, vars_idx=v, tt=t
                ).squeeze(0)  # (L,)
                self._assign_list.append(assign.to("cpu", non_blocking=True).to(torch.int32))


        if self.use_cache:
            self.save_cache()

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx: int):
        item = {
            "sample_long": self._samples[idx],                           # (L,3): [t_norm, x, var_id]
            "static": self._statics[idx],                                # (D,)
            "label": torch.tensor(self._labels[idx], dtype=torch.long),  # ()
            "ts_id": self._ts_ids[idx],                                  # int
        }
        if self.enable_agg_mask and (self._assign_list is not None):
            item["assign"] = self._assign_list[idx]                     # (L,)
        return item

    # ---------------- collate ---------------- #
    def collate_fn(self, batch: list[dict]) -> dict:
        return self.long_collate_fn(batch) if self.output_style == "long" else self.matrix_collate_fn(batch)

    def long_collate_fn(self, batch: list[dict]) -> dict:
        ts_list, val_list, vid_list, pad_list = [], [], [], []
        stat_list, y_list, id_list, assign_list = [], [], [], []

        for item in batch:
            smp = item["sample_long"]  # (L,3)
            if smp.numel() == 0:
                ts_list.append(torch.zeros(0, dtype=torch.float32))
                val_list.append(torch.zeros(0, dtype=torch.float32))
                vid_list.append(torch.zeros(0, dtype=torch.long))
                pad_list.append(torch.zeros(0, dtype=torch.float32))
            else:
                t = smp[:, 0].to(torch.float32)
                x = smp[:, 1].to(torch.float32)
                raw_vid = smp[:, 2].to(torch.long).tolist()
                col_vid = torch.tensor([self.var2col[int(v)] for v in raw_vid], dtype=torch.long)
                pad = torch.ones_like(x, dtype=torch.float32)
                ts_list.append(t)
                val_list.append(x)
                vid_list.append(col_vid)
                pad_list.append(pad)

            stat_list.append(item["static"].to(torch.float32))
            y_list.append(item["label"].to(torch.long))
            id_list.append(torch.tensor(item["ts_id"], dtype=torch.long))
            assign_list.append(item.get("assign", None))

        batch_timestamp = pad_sequence(ts_list, batch_first=True, padding_value=0.0)
        batch_value     = pad_sequence(val_list, batch_first=True, padding_value=0.0)
        batch_var_idx   = pad_sequence(vid_list, batch_first=True, padding_value=0)
        batch_pad_mask  = pad_sequence(pad_list, batch_first=True, padding_value=0.0)

        batch_static = (
            torch.zeros(len(stat_list), 0, dtype=torch.float32)
            if stat_list[0].numel() == 0 else torch.stack(stat_list, dim=0)
        )
        batch_label = torch.stack(y_list, dim=0)
        batch_ts_id = torch.stack(id_list, dim=0)

        out = {
            "batch_timestamp": batch_timestamp,   # (B, Nmax)
            "batch_value":     batch_value,       # (B, Nmax)
            "batch_var_idx":   batch_var_idx,     # (B, Nmax) -> 0..V-1
            "batch_pad_mask":  batch_pad_mask,    # (B, Nmax) 1=valid
            "batch_static":    batch_static,      # (B, D)
            "batch_label":     batch_label,       # (B,)
            "batch_ts_id":     batch_ts_id,       # (B,)
        }

        # 聚合掩码（可选）
        if self.enable_agg_mask and any(a is not None for a in assign_list):
            B, Nmax = batch_pad_mask.shape
            Lout = int(self.hyper_num_nodes)
            batch_agg_mask = torch.zeros(B, Lout, Nmax, dtype=torch.float32, device=batch_pad_mask.device)
            for b, assign in enumerate(assign_list):
                if assign is None:
                    continue
                Li = int((batch_pad_mask[b] > 0).sum().item())
                if Li == 0:
                    continue
                a = assign[:Li].to(torch.int64)
                valid = a >= 0
                if valid.any():
                    rows = a[valid]
                    cols = torch.arange(Li, device=batch_agg_mask.device)[valid]
                    batch_agg_mask[b].index_put_((rows, cols), torch.ones_like(rows, dtype=torch.float32), accumulate=False)
            out["batch_agg_mask"] = batch_agg_mask  # (B, Lout, Nmax)

        return out

    @torch.no_grad()
    def matrix_collate_fn(self, batch: list[dict]) -> dict:
        V = len(self._all_var_ids)
        device = batch[0]["sample_long"].device
        dtype = batch[0]["sample_long"].dtype

        obs_t_list, obs_v_list, obs_m_list = [], [], []
        stat_list, y_list, id_list = [], [], []

        qscale = 1e9

        for item in batch:
            smp = item["sample_long"]  # (L,3): [t_norm, x, vid]
            if smp.numel() == 0:
                obs_t_list.append(torch.zeros(0, dtype=dtype, device=device))
                obs_v_list.append(torch.zeros(0, V, dtype=dtype, device=device))
                obs_m_list.append(torch.zeros(0, V, dtype=dtype, device=device))
            else:
                t = smp[:, 0].to(dtype)
                x = smp[:, 1].to(dtype)
                vid = smp[:, 2].to(torch.long)

                tq = torch.round(t * qscale) / qscale
                t_u, inv = torch.unique(tq, sorted=True, return_inverse=True)
                L = int(t_u.shape[0])

                vals = torch.zeros(L, V, dtype=dtype, device=device)
                mask = torch.zeros(L, V, dtype=dtype, device=device)
                col = torch.tensor([self.var2col[int(v.item())] for v in vid], dtype=torch.long, device=device)

                mask.index_put_((inv, col), torch.ones_like(x, dtype=dtype), accumulate=False)
                vals.index_put_((inv, col), x, accumulate=False)

                obs_t_list.append(t_u)
                obs_v_list.append(vals)
                obs_m_list.append(mask)

            stat_list.append(item["static"].to(torch.float32))
            y_list.append(item["label"].to(torch.long))
            id_list.append(torch.tensor(item["ts_id"], dtype=torch.long, device=device))

        observed_tp   = pad_sequence(obs_t_list, batch_first=True) 
        observed_data = pad_sequence(obs_v_list, batch_first=True)
        observed_mask = pad_sequence(obs_m_list, batch_first=True)

        batch_static = (
            torch.zeros(len(stat_list), 0, dtype=torch.float32, device=device)
            if stat_list[0].numel() == 0 else torch.stack(stat_list, dim=0).to(device)
        )
        batch_label = torch.stack(y_list, dim=0).to(device)
        batch_ts_id = torch.stack(id_list, dim=0).to(device)

        return {
            "observed_data": observed_data,  # (B, T_max, V)
            "observed_tp":   observed_tp,    # (B, T_max) 
            "observed_mask": observed_mask,  # (B, T_max, V)
            "batch_static":  batch_static,   # (B, D)
            "batch_label":   batch_label,    # (B,)
            "batch_ts_id":   batch_ts_id,    # (B,)
        }

    def _make_cache_name(self) -> str:
        parts = [
            re.sub(r"\W+", "", str(self.data_name)),
            f"its_cls_ds",
            f"split-{self.split}",
            f"out-{self.output_style}",
            f"t{int(self.time_scale)}",
            f"V{len(self._all_var_ids)}",
            f"D{(self.split_static.shape[1] if self.split_static.size > 0 else 0)}",
            f"N{len(self._samples)}",
            f"agg{int(self.enable_agg_mask)}",
            f"Lout{self.num_hyper_nodes_per_var}",
            f"it{self.agg_gp_train_iters}",
            f"lr{self.agg_gp_lr}",
            f"bs{self.agg_sample_ratio}",
            f"dyn{int(self.agg_dynamic)}",
            f"normv{int(self.agg_norm_across_vars)}",
        ]
        return "its_cls_ds__" + "__".join(map(str, parts)) + ".pt"

    def _make_gp_cache_name(self) -> str:
        parts = [
            re.sub(r"\W+", "", str(self.data_name)),
            "cls",
            f"V{len(self._all_var_ids)}",
            f"Lout{self.num_hyper_nodes_per_var}",
            f"it{self.agg_gp_train_iters}",
            f"lr{self.agg_gp_lr}",
            f"bs{self.agg_sample_ratio}",
            f"dyn{int(self.agg_dynamic)}",
            f"normv{int(self.agg_norm_across_vars)}",
        ]
        return "cached_gp_models__" + "__".join(map(str, parts)) + ".pt"

    def save_cache(self) -> None:
        payload = {
            "config": self._config,
            "samples": [t.detach().cpu() for t in self._samples],
            "statics": [s.detach().cpu() for s in self._statics],
            "labels":  np.asarray(self._labels, dtype=np.int64),
            "ts_ids":  np.asarray(self._ts_ids, dtype=np.int64),
            "all_var_ids": list(self._all_var_ids),
            "var2col": dict(self.var2col),
            "hyper_num_nodes": int(self.hyper_num_nodes),
        }
        if getattr(self, "_assign_list", None) is not None:
            payload["assign_list"] = [
                (a.detach().cpu() if isinstance(a, torch.Tensor) else None)
                for a in self._assign_list
            ]
        torch.save(payload, self.cache_path)

    def load_from_cache(self) -> bool:
        if not (self.use_cache and os.path.exists(self.cache_path)):
            return False
        try:
            payload = torch.load(self.cache_path, map_location="cpu")
        except Exception:
            return False
        if not isinstance(payload, dict):
            return False
        if payload.get("config") != self._config:
            return False

        self._samples = payload["samples"]
        self._statics = payload["statics"]
        self._labels = list(payload["labels"])
        self._ts_ids = list(payload["ts_ids"])
        self._all_var_ids = list(payload["all_var_ids"])
        self.var2col = dict(payload["var2col"])
        self.hyper_num_nodes = int(payload.get(
            "hyper_num_nodes",
            self.num_hyper_nodes_per_var * max(1, len(self._all_var_ids))
        ))
        self._assign_list = payload.get("assign_list", None)

        if getattr(self, "num_hyper_nodes_per_var", None) is None:
            V = max(1, len(self._all_var_ids))
            self.num_hyper_nodes_per_var = int(round(self.hyper_num_nodes / V))

        self.num_vars = len(self._all_var_ids)
        return True
