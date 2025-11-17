import torch
from abc import ABC, abstractmethod
import numpy as np


class Scaler(ABC):
    def __init__(self, axis=None):
        self.axis = axis

    @abstractmethod
    def fit(self, data):
        pass

    @abstractmethod
    def transform(self, data):
        pass

    @abstractmethod
    def inverse_transform(self, data):
        pass

    def save(self, path):
        np.savez(path, **self.__dict__)

    def load(self, path):
        data = np.load(path)
        for key in data.files:
            setattr(self, key, data[key])


class ListMinMaxScaler(Scaler):
    def __init__(self, axis=0, min=None, max=None, *args, **kwargs):
        super(ListMinMaxScaler, self).__init__(axis)
        self.min = min
        self.max = max

    def fit(self, data):
        all_mins = []
        all_maxs = []

        for _, _, vals, mask in data:
            masked_vals = np.where(mask.bool(), vals, np.nan)
            all_mins.append(np.nanmin(masked_vals, axis=self.axis))
            all_maxs.append(np.nanmax(masked_vals, axis=self.axis))

        self.min = np.nanmin(all_mins, axis=self.axis) if all_mins else np.array([])
        self.max = np.nanmax(all_maxs, axis=self.axis) if all_maxs else np.array([])

    def transform(self, data, index=None):
        transformed_data = []
        for record_id, tt, vals, mask in data:
            masked_vals = np.where(mask.bool(), vals, np.nan)
            if index is None:
                masked_vals = (masked_vals - self.min) / (self.max - self.min)
            else:
                masked_vals[:, index] = (masked_vals[:, index] - self.min[index]) / (
                    self.max[index] - self.min[index]
                )
            # fill masked values with 0
            masked_vals = torch.tensor(np.where(mask.bool(), masked_vals, 0))
            # also fill nan values with 0
            masked_vals = torch.tensor(np.nan_to_num(masked_vals))
            transformed_data.append((record_id, tt, masked_vals, mask))

        return transformed_data

    def inverse_transform(self, data, index=None):
        inverse_transformed_data = []
        for record_id, tt, vals, mask in data:
            if not isinstance(vals, np.ndarray):
                vals = vals.numpy()
            masked_vals = np.where(mask.bool(), vals, np.nan)
            if index is None:
                masked_vals = masked_vals * (self.max - self.min) + self.min
            else:
                masked_vals[:, index] = (
                    masked_vals[:, index] * (self.max[index] - self.min[index])
                    + self.min[index]
                )
            # fill masked values with 0
            masked_vals = torch.tensor(np.where(mask.bool(), masked_vals, 0))
            inverse_transformed_data.append((record_id, tt, masked_vals, mask))
        return inverse_transformed_data

    def inverse_transform_mx(self, data, index=None, var_idx=None):
        vars_num = len(self.min)
        inverse_data = np.zeros(data.shape)
        for i in range(vars_num):
            var_mask = var_idx == i
            if index is None:
                var_data = (data * (self.max[i] - self.min[i]) + self.min[i]) * var_mask
                inverse_data += var_data
            else:
                var_data = (
                    data[:, index] * (self.max[i] - self.min[i]) + self.min[i]
                ) * var_mask
                inverse_data[:, index] += var_data
        return inverse_data


class ListStandardScaler(Scaler):
    def __init__(self, axis=0, mean=None, std=None, *args, **kwargs):
        super(ListStandardScaler, self).__init__()
        self.mean = mean
        self.std = std

    def fit(self, data):
        all_vals = []

        for _, _, vals, mask in data:
            # Convert the mask to boolean and apply it
            masked_vals = np.where(mask, vals, np.nan)
            all_vals.append(masked_vals)

        if all_vals:
            all_vals = np.concatenate(all_vals)
            self.mean = np.nanmean(all_vals, axis=self.axis)
            self.std = np.nanstd(all_vals, axis=self.axis)

    def transform(self, data, index=None):
        transformed_data = []
        for record_id, tt, vals, mask in data:
            masked_vals = np.where(mask, vals, np.nan)
            if index is None:
                masked_vals = (masked_vals - self.mean) / np.clip(
                    self.std, 1e-5, np.inf
                )
            else:
                masked_vals[:, index] = (
                    masked_vals[:, index] - self.mean[index]
                ) / np.clip(self.std[index], 1e-5, np.inf)
            # Replace nan values with zero in masked positions
            masked_vals = torch.tensor(np.where(mask, masked_vals, 0))
            transformed_data.append((record_id, tt, masked_vals, mask))

        return transformed_data

    def inverse_transform(self, data, index=None):
        inverse_transformed_data = []
        for record_id, tt, vals, mask in data:
            if not isinstance(vals, np.ndarray):
                vals = vals.numpy()
            masked_vals = np.where(mask, vals, np.nan)
            if index is None:
                masked_vals = masked_vals * self.std + self.mean
            else:
                masked_vals[:, index] = (
                    masked_vals[:, index] * self.std[index] + self.mean[index]
                )
            # Replace nan values with zero in masked positions
            masked_vals = torch.tensor(np.where(mask, masked_vals, 0))
            inverse_transformed_data.append((record_id, tt, masked_vals, mask))

        return inverse_transformed_data

    def inverse_transform_mx(self, data, index=None):
        if index is None:
            inverse_data = data * self.std + self.mean
        else:
            inverse_data = data
            inverse_data[:, index] = (
                data[:, index] * self.std[index] + self.mean[index]
            )
        return inverse_data

class LongMinMaxScaler:
    def __init__(self, min=None, max=None):
        """
        Min-Max scaler for long-form (ts_id, var_id, timestamp, value) data.
        Each variable (var_id) is scaled independently:
            scaled = (x - min) / (max - min)
        and inverse transform:
            x = scaled * (max - min) + min
        """
        self.min_vals = min  # dict[var_id] -> min value
        self.max_vals = max  # dict[var_id] -> max value

    def fit(self, data):
        """
        Compute min and max for each var_id.

        Parameters
        ----------
        data : np.ndarray or torch.Tensor, shape (N, 4)
            Columns: ts_id, var_id, timestamp, value
        """
        if isinstance(data, dict):
            data = data['train_data']
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()

        self.min_vals = {}
        self.max_vals = {}
        for var_id in np.unique(data[:, 1].astype(int)):
            vals = data[data[:, 1] == var_id, 3]
            self.min_vals[var_id] = np.nanmin(vals)
            self.max_vals[var_id] = np.nanmax(vals)

    def transform(self, data):
        """
        Apply min-max scaling for each var_id.

        Returns
        -------
        Scaled data with the same type as input (numpy or torch.Tensor).
        """
        if isinstance(data, dict):
            key = next(k for k in ["train_data", "valid_data", "test_data"] if k in data)
            data = data[key]
        is_tensor = isinstance(data, torch.Tensor)
        if is_tensor:
            data = data.cpu().numpy()

        scaled = data.copy()
        for var_id in np.unique(scaled[:, 1].astype(int)):
            mask = scaled[:, 1] == var_id
            min_v = self.min_vals[var_id]
            max_v = self.max_vals[var_id]
            scaled[mask, 3] = (scaled[mask, 3] - min_v) / (max_v - min_v if max_v != min_v else 1.0)

        if is_tensor:
            return torch.tensor(scaled, dtype=torch.float32)
        return scaled

    def inverse_transform(self, data):
        """
        Reverse min-max scaling to recover original values.
        """
        is_tensor = isinstance(data, torch.Tensor)
        if is_tensor:
            data = data.cpu().numpy()

        restored = data.copy()
        for var_id in np.unique(restored[:, 1].astype(int)):
            mask = restored[:, 1] == var_id
            min_v = self.min_vals[var_id]
            max_v = self.max_vals[var_id]
            restored[mask, 3] = restored[mask, 3] * (max_v - min_v if max_v != min_v else 1.0) + min_v

        if is_tensor:
            return torch.tensor(restored, dtype=torch.float32)
        return restored

    def save(self, path):
        np.savez(path, **self.__dict__)

    def load(self, path):
        data = np.load(path)
        for key in data.files:
            setattr(self, key, data[key])

class LongStandardScaler:
    def __init__(self, mean=None, std=None):
        """
        Standard scaler for long-form (ts_id, var_id, timestamp, value) data.
        Each variable (var_id) is scaled independently:
            z = (x - mean) / std
        and inverse transform:
            x = z * std + mean
        """
        self.means = mean  # dict[var_id] -> mean
        self.stds = std    # dict[var_id] -> std

    def fit(self, data):
        """
        Fit mean and std for each var_id.

        Parameters
        ----------
        data : np.ndarray or torch.Tensor, shape (N, 4)
            Columns: ts_id, var_id, timestamp, value
        """
        if isinstance(data, dict):
            data = data['train_data']
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()

        self.means = {}
        self.stds = {}
        for var_id in np.unique(data[:, 1].astype(int)):
            vals = data[data[:, 1] == var_id, 3]
            self.means[var_id] = np.nanmean(vals)
            self.stds[var_id] = np.nanstd(vals)

    def transform(self, data):
        """
        Apply standard scaling to each var_id.

        Returns
        -------
        Scaled data with the same type as input (numpy or torch.Tensor).
        """
        if isinstance(data, dict):
            data = data[(k for k in data.keys() if k.endswith("_data"))]
        is_tensor = isinstance(data, torch.Tensor)
        if is_tensor:
            data = data.cpu().numpy()

        scaled = data.copy()
        for var_id in np.unique(scaled[:, 1].astype(int)):
            mask = scaled[:, 1] == var_id
            mean_v = self.means[var_id]
            std_v = self.stds[var_id]
            scaled[mask, 3] = (scaled[mask, 3] - mean_v) / (std_v if std_v != 0 else 1.0)

        if is_tensor:
            return torch.tensor(scaled, dtype=torch.float32)
        return scaled

    def inverse_transform(self, data):
        """
        Reverse standard scaling to recover original values.
        """
        is_tensor = isinstance(data, torch.Tensor)
        if is_tensor:
            data = data.cpu().numpy()

        restored = data.copy()
        for var_id in np.unique(restored[:, 1].astype(int)):
            mask = restored[:, 1] == var_id
            mean_v = self.means[var_id]
            std_v = self.stds[var_id]
            restored[mask, 3] = restored[mask, 3] * (std_v if std_v != 0 else 1.0) + mean_v

        if is_tensor:
            return torch.tensor(restored, dtype=torch.float32)
        return restored

    def save(self, path):
        np.savez(path, **self.__dict__)

    def load(self, path):
        data = np.load(path)
        for key in data.files:
            setattr(self, key, data[key])

class DynStaLongStandardScaler:
    def __init__(
        self,
        eps: float = 1e-18,
        static_categorical_idx=None,
        treat_nonpositive_as_missing: bool = True,
    ):
        self.eps = float(eps)
        self.dyn_mean_ = {}
        self.dyn_std_ = {}
        self.sta_mean_ = {}
        self.sta_std_ = {}

        self.static_categorical_idx = set(static_categorical_idx or [])
        self.treat_nonpositive_as_missing = bool(treat_nonpositive_as_missing)

    def fit(self, data: dict):
        dyn = self._to_np(data.get("train_data"))
        sta = self._to_np(data.get("train_static_data"))

        if dyn is not None and dyn.size > 0:
            vids = dyn[:, 1].astype(int)
            vals = dyn[:, 3]
            for vid in np.unique(vids):
                mask = vids == vid
                obs = vals[mask]
                obs = obs[obs > 0]
                mu = float(np.mean(obs)) if obs.size else 0.0
                sd = float(np.std(obs)) if obs.size else 1.0
                self.dyn_mean_[int(vid)] = mu
                self.dyn_std_[int(vid)] = sd if sd > 0 else 1.0

        if sta is not None and sta.size > 0:
            assert sta.ndim == 2, "train_static_data must be [#ts, #feat]"
            _, D = sta.shape
            for fid in range(D):
                if fid in self.static_categorical_idx:
                    continue
                col = sta[:, fid]
                obs = col[col > 0] if self.treat_nonpositive_as_missing else col
                mu = float(np.mean(obs)) if obs.size else 0.0
                sd = float(np.std(obs)) if obs.size else 1.0
                self.sta_mean_[int(fid)] = mu
                self.sta_std_[int(fid)] = sd if sd > 0 else 1.0
        return self

    def transform(self, data: dict) -> dict:
        out = dict(data)

        data_key = next((k for k in data if k.endswith("_data")), None)
        sta_key  = next((k for k in data if k.endswith("_static_data")), None)

        if data_key is not None and data[data_key] is not None and data[data_key].size > 0:
            din = data[data_key]
            dyn = self._to_np(din).copy()
            vids = dyn[:, 1].astype(int)
            orig = dyn[:, 3].copy()
            for vid in np.unique(vids):
                idx = (vids == vid)
                mu = self.dyn_mean_.get(int(vid), 0.0)
                sd = self.dyn_std_.get(int(vid), 1.0)
                dyn[idx, 3] = (dyn[idx, 3] - mu) / (sd + self.eps)
            if self.treat_nonpositive_as_missing:
                dyn[orig <= 0, 3] = 0.0
            out[data_key] = self._from_like(din, dyn)

        if sta_key is not None and data[sta_key] is not None and data[sta_key].size > 0:
            sin = data[sta_key]
            sta = self._to_np(sin).copy()
            _, D = sta.shape
            for fid in range(D):
                col = sta[:, fid]
                orig = col.copy()
                if fid in self.static_categorical_idx:
                    if self.treat_nonpositive_as_missing:
                        sta[orig <= 0, fid] = 0.0
                    else:
                        sta[:, fid] = col
                else:
                    mu = self.sta_mean_.get(int(fid), 0.0)
                    sd = self.sta_std_.get(int(fid), 1.0)
                    sta[:, fid] = (col - mu) / (sd + self.eps)
                    if self.treat_nonpositive_as_missing:
                        sta[orig <= 0, fid] = 0.0
            out[sta_key] = self._from_like(sin, sta)

        return out

    def inverse_transform(self, data: dict) -> dict:
        out = dict(data)

        data_key = next((k for k in data if k.endswith("_data")), None)
        sta_key  = next((k for k in data if k.endswith("_static_data")), None)

        # 动态还原
        if data_key is not None and data[data_key] is not None and data[data_key].size > 0:
            din = data[data_key]
            dyn = self._to_np(din).copy()
            vids = dyn[:, 1].astype(int)
            for vid in np.unique(vids):
                idx = (vids == vid)
                mu = self.dyn_mean_.get(int(vid), 0.0)
                sd = self.dyn_std_.get(int(vid), 1.0)
                dyn[idx, 3] = dyn[idx, 3] * (sd + self.eps) + mu
            out[data_key] = self._from_like(din, dyn)

        if sta_key is not None and data[sta_key] is not None and data[sta_key].size > 0:
            sin = data[sta_key]
            sta = self._to_np(sin).copy()
            _, D = sta.shape
            for fid in range(D):
                if fid in self.static_categorical_idx:
                    continue
                mu = self.sta_mean_.get(int(fid), 0.0)
                sd = self.sta_std_.get(int(fid), 1.0)
                sta[:, fid] = sta[:, fid] * (sd + self.eps) + mu
            out[sta_key] = self._from_like(sin, sta)

        return out

    def save(self, path: str) -> None:
        np.savez(
            path,
            eps=self.eps,
            dyn_mean=self._dict_to_kv(self.dyn_mean_),
            dyn_std=self._dict_to_kv(self.dyn_std_),
            sta_mean=self._dict_to_kv(self.sta_mean_),
            sta_std=self._dict_to_kv(self.sta_std_)
        )

    def load(self, path: str):
        z = np.load(path, allow_pickle=True)
        self.eps = float(z["eps"])
        self.dyn_mean_ = self._kv_to_dict(z["dyn_mean"])
        self.dyn_std_  = self._kv_to_dict(z["dyn_std"])
        self.sta_mean_ = self._kv_to_dict(z["sta_mean"])
        self.sta_std_  = self._kv_to_dict(z["sta_std"])
        self.static_categorical_idx = z["static_categorical_idx"]
        self.treat_nonpositive_as_missing = z["treat_nonpositive_as_missing"]
        return self

    @staticmethod
    def _to_np(x):
        if x is None:
            return None
        if isinstance(x, np.ndarray):
            return x
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        raise TypeError(f"Unsupported array type: {type(x)}")

    @staticmethod
    def _from_like(example, np_arr):
        if isinstance(example, torch.Tensor):
            return torch.tensor(np_arr, dtype=example.dtype)
        return np_arr

    @staticmethod
    def _dict_to_kv(d):
        if not d:
            return np.zeros((0, 2), dtype=np.float64)
        k = np.array(list(d.keys()), dtype=np.int32).reshape(-1, 1)
        v = np.array(list(d.values()), dtype=np.float64).reshape(-1, 1)
        return np.concatenate([k, v], axis=1)

    @staticmethod
    def _kv_to_dict(kv):
        out = {}
        if kv.size == 0:
            return out
        for k, v in kv:
            out[int(k)] = float(v)
        return out
