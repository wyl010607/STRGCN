import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from typing import Dict, Tuple, List

from data_processors.abs import AbstractDataProcessor


class Physionet2012Processor(AbstractDataProcessor):
    """ """

    # -------------------------- Dataset Metadata -------------------------- #

    DYNAMIC_FEATURES: List[str] = [
        'Age', 'Gender', 'Height', 'ICUType', 'Weight', 'Albumin', 'ALP', 'ALT', 'AST', 'Bilirubin', 'BUN',
        'Cholesterol', 'Creatinine', 'DiasABP', 'FiO2', 'GCS', 'Glucose', 'HCO3', 'HCT', 'HR', 'K', 'Lactate', 'Mg',
        'MAP', 'MechVent', 'Na', 'NIDiasABP', 'NIMAP', 'NISysABP', 'PaCO2', 'PaO2', 'pH', 'Platelets', 'RespRate',
        'SaO2', 'SysABP', 'Temp', 'TroponinI', 'TroponinT', 'Urine', 'WBC'
    ]
    STATIC_FEATURES: List[str] = []               # no separate static matrix here
    CLASS_LABELS: List[str] = []                  # not implemented
    REG_LABELS: List[str] = []                    # not implemented


    def __init__(self, *args, **kwargs):
        super().__init__("physionet_2012", *args, **kwargs)
        self.set_dirs = ["set-a", "set-b", "set-c"]
        self.quantization_sec = 360 # 0.1 hours
        self.PARAMS = self.DYNAMIC_FEATURES.copy()

    def _read_one_record(self, file_path: Path) -> Tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]:

        record_id = file_path.stem  # string id as in B
        # Read as DataFrame; lines are like "Time,Parameter,Value"
        df = pd.read_csv(file_path, sep=",")
        # Defensive: drop rows where required fields are missing
        df = df.dropna(subset=["Time", "Parameter", "Value"])

        # Convert "HH:MM" to float hours
        hhmm = df["Time"].astype(str).str.split(":", expand=True)
        hh = pd.to_numeric(hhmm[0], errors="coerce").fillna(0.0)
        mm = pd.to_numeric(hhmm[1], errors="coerce").fillna(0.0)
        df["t_sec"] = hh * 3600 + mm * 60

        q = self.quantization_sec
        if q > 0:
            df["t_sec"] = np.round(df["t_sec"] / q) * q

        # Keep only variables we recognize
        df = df[df["Parameter"].isin(self.PARAMS)].copy()
        if df.empty:
            # Return an empty skeleton (rare; but keep shapes consistent)
            tt = torch.tensor([0.0], dtype=torch.float32)
            D = len(self.PARAMS)
            vals = torch.zeros((1, D), dtype=torch.float32)
            mask = torch.zeros((1, D), dtype=torch.float32)
            return record_id, tt, vals, mask

        # Average-aggregate duplicates at the same (t, variable)
        df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
        df = df.dropna(subset=["Value"])
        gp = df.groupby(["t_sec", "Parameter"], as_index=False)["Value"].mean()

        # Now pivot to time-major dense matrix (vals, mask); keep times sorted
        gp = gp.sort_values(["t_sec", "Parameter"])
        times = np.sort(gp["t_sec"].unique().astype(float))
        D = len(self.PARAMS)
        vals = np.zeros((len(times), D), dtype=np.float32)
        mask = np.zeros((len(times), D), dtype=np.float32)

        name_to_col = {name: j for j, name in enumerate(self.PARAMS)}
        t_to_row = {t: i for i, t in enumerate(times)}

        for _, row in gp.iterrows():
            i = t_to_row[float(row["t_sec"])]
            j = name_to_col[row["Parameter"]]
            vals[i, j] = float(row["Value"])
            mask[i, j] = 1.0

        tt = torch.tensor(times, dtype=torch.float32)
        vals_t = torch.tensor(vals, dtype=torch.float32)
        mask_t = torch.tensor(mask, dtype=torch.float32)
        return record_id, tt, vals_t, mask_t

    def load_data(self) -> List[Tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Read all records from set-a/b/c into a single list in a deterministic order.
        """
        patients: List[Tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for sd in self.set_dirs:
            dir_path = self.raw_data_path / sd
            if not dir_path.exists():
                raise FileNotFoundError(f"Missing directory: {dir_path}")
            # Deterministic order
            for fname in sorted(os.listdir(dir_path)):
                if not fname.endswith(".txt"):
                    continue
                rec = self._read_one_record(dir_path / fname)
                patients.append(rec)
        return patients

    def split_data(self, data) -> Tuple[list, list, list]:
        """
          seen, test = train_test_split(total, train_size=0.8, random_state=42, shuffle=True)
          train, val  = train_test_split(seen,  train_size=0.75, random_state=42, shuffle=False)
        """
        from sklearn.model_selection import train_test_split as tts
        seen, test = tts(data, train_size=0.8, random_state=42, shuffle=True)
        train, val = tts(seen, train_size=0.75, random_state=42, shuffle=False)
        return train, val, test

    @staticmethod
    def _build_id_maps(
        patients: List[Tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]],
        var_names: List[str],
    ) -> Tuple[Dict[str, int], Dict[int, str], Dict[int, str]]:
        """
        Build mapping on the full list order, so ts_id is stable across splits.
        """
        ordered_ids = [str(rid) for rid, _, _, _ in patients]
        rid2ts = {rid: i for i, rid in enumerate(ordered_ids)}
        ts_id_to_name = {i: rid for i, rid in enumerate(ordered_ids)}
        val_id_to_name = {i: n for i, n in enumerate(var_names)}
        return rid2ts, ts_id_to_name, val_id_to_name

    @staticmethod
    def _subset_to_long(
        subset: List[Tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]],
        rid2ts: Dict[str, int],
        V: int,
    ) -> np.ndarray:
        """
        Convert (record_id, tt[h], vals[T,V], mask[T,V]) to long ndarray:
          columns = [ts_id, val_id, timestamp, value]
        Only keep entries where mask == 1.  timestamp unit is HOURS (float).
        """
        rows = []
        for rid, tt, vals, mask in subset:
            ts_id = float(rid2ts[str(rid)])
            t = tt.numpy()            # (T,)
            x = vals.numpy()          # (T,V)
            m = mask.numpy()          # (T,V)
            obs_pos = np.argwhere(m > 0.5)
            # obs_pos rows: [i_time, j_var]
            for i, j in obs_pos:
                rows.append([ts_id, float(j), float(t[i]), float(x[i, j])])
        if not rows:
            return np.zeros((0, 4), dtype=np.float64)
        return np.asarray(rows, dtype=np.float64)

    # ------------------------------- tasks -------------------------------- #
    def process_prediction(self):

        data = self.load_data()  
        train_set, val_set, test_set = self.split_data(data)

        test_ids = [rid for rid, _, _, _ in test_set]
        print("Dataset n_samples:", len(data), len(train_set), len(val_set), len(test_set))
        print("Test record ids (first 20):", test_ids[:20])
        print("Test record ids (last 20):", test_ids[-20:])

        rid2ts, ts_map, val_map = self._build_id_maps(data, self.PARAMS)
        V = len(self.PARAMS)

        x_tr = self._subset_to_long(train_set, rid2ts, V)
        x_va = self._subset_to_long(val_set, rid2ts, V)
        x_te = self._subset_to_long(test_set, rid2ts, V)
        # seen_data for scaling, = train + val
        seen_data = np.vstack([x_tr, x_va])

        # Expose feature names (B-style unified space as "dynamic")
        self.DYNAMIC_FEATURES = self.PARAMS.copy()

        return dict(
            train_data=x_tr,
            valid_data=x_va,
            test_data=x_te,
            seen_data=seen_data,
            static_feature=np.array([], dtype=object),
            dynamic_feature=np.array(self.DYNAMIC_FEATURES, dtype=object),
            ts_id_to_name=np.array([ts_map], dtype=object),
            val_id_to_name=np.array([val_map], dtype=object),
        )

    process_imputation = process_prediction

    def process_classification(self):
        """
        Not implemented in this B-style unification (no Outcomes loaded).
        """
        raise NotImplementedError("Classification is not implemented in this B-style processor (no labels loaded).")

