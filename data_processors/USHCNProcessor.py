# uschn_processor.py
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
from sklearn.model_selection import train_test_split

from data_processors.abs import AbstractDataProcessor


class USHCNProcessor(AbstractDataProcessor):
    """
    USHCN processor
    """

    DYNAMIC_FEATURES: List[str] = ["value"]
    STATIC_FEATURES: List[str] = []
    CLASS_LABELS: List[str] = ["Label"]
    REG_LABELS: List[str] = []

    def __init__(self, *args, **kwargs):
        super().__init__("ushcn", *args, **kwargs)
        self.csv_file: Path = self.raw_data_path / "small_chunked_sporadic.csv"
        self.target_n_months: float = 48.0
        self.time_raw_max: float = 200.0


    def _build_tuples(self, df: pd.DataFrame) -> Tuple[List[tuple], int]:

        value_cols = [c for c in df.columns if c.startswith("Value_")]
        mask_cols = [f"Mask_{c.split('_', 1)[1]}" for c in value_cols]
        V = len(value_cols)
        self.DYNAMIC_FEATURES = [f"label_{j}" for j in range(V)]

        scale = float(48.0) / float(200.0)
        df = df.copy()
        df["Time_norm"] = df["Time"].astype(float) * scale # 48 months

        tuples = []
        # Keep original ID order in the file
        for rid, g in df.groupby("ID", sort=False):
            tt = g["Time_norm"].to_numpy(dtype=np.float64)      # (T,)
            vals = g[value_cols].to_numpy(dtype=np.float64)     # (T, V)
            mask = g[mask_cols].to_numpy(dtype=np.float64)      # (T, V)
            order = np.argsort(tt, kind="mergesort")
            tuples.append((int(rid), tt[order], vals[order], mask[order]))
        return tuples, V

    # ---------------------------- splitting ---------------------------- #
    def split_data(self, data: List[tuple]) -> Tuple[List[tuple], List[tuple], List[tuple]]:
        """
        Two-step split (seed=42):
          seen, test = train_test_split(data,  train_size=0.8, random_state=42, shuffle=True)
          train, val = train_test_split(seen, train_size=0.75, random_state=42, shuffle=False)
        """
        seen, test = train_test_split(data, train_size=0.8, random_state=42, shuffle=True)
        train, val = train_test_split(seen, train_size=0.75, random_state=42, shuffle=False)
        return train, val, test

    # --------------------------- long builder --------------------------- #
    def _build_id_maps(self, data: List[tuple]) -> Dict[int, int]:
        """Map record_id -> ts_id (0..N-1) using the global order in 'data'."""
        ordered_rids = [int(rec_id) for rec_id, _, _, _ in data]
        return {rid: i for i, rid in enumerate(ordered_rids)}

    def _subset_to_long(self, subset: List[tuple], rid2ts: Dict[int, int], V: int) -> np.ndarray:
        """
        Convert a subset to a long ndarray [ts_id, val_id, timestamp, value] (float64),
        keeping only entries where mask == 1.
        """
        rows = []
        for rec_id, tt, vals, mask in subset:
            ts_id = float(rid2ts[int(rec_id)])
            for j in range(V):
                obs = mask[:, j] > 0.5
                if not np.any(obs):
                    continue
                tj = tt[obs].astype(np.float64)
                xj = vals[obs, j].astype(np.float64)
                ts_col = np.full_like(tj, ts_id, dtype=np.float64)
                val_col = np.full_like(tj, float(j), dtype=np.float64)
                rows.append(np.stack([ts_col, val_col, tj, xj], axis=1))
        if not rows:
            return np.zeros((0, 4), dtype=np.float64)
        return np.concatenate(rows, axis=0)

    # --------------------------- public API --------------------------- #
    def load_data(self) -> Tuple[List[tuple], int]:
        """Load CSV and return tuple list plus variable count V."""
        df = pd.read_csv(self.csv_file)
        df["ID"] = df["ID"].astype("int64")
        value_cols = [c for c in df.columns if c.startswith("Value_")]
        mask_cols = [f"Mask_{c.split('_', 1)[1]}" for c in value_cols]
        V = len(value_cols)
        # Simple dynamic feature names: label_0..label_{V-1}
        self.DYNAMIC_FEATURES = [f"label_{j}" for j in range(V)]

        scale = float(48.0) / float(200.0)
        df = df.copy()
        df["Time_norm"] = df["Time"].astype(float) * scale # 48 months

        data = []
        # Keep original ID order in the file
        for rid, g in df.groupby("ID", sort=False):
            tt = g["Time_norm"].to_numpy(dtype=np.float64)      # (T,)
            vals = g[value_cols].to_numpy(dtype=np.float64)     # (T, V)
            mask = g[mask_cols].to_numpy(dtype=np.float64)      # (T, V)
            order = np.argsort(tt, kind="mergesort")
            data.append((int(rid), tt[order], vals[order], mask[order]))
        return data, V

    def process_prediction(self):
        data, V = self.load_data()
        train_set, val_set, test_set = self.split_data(data)
        ordered_rids = [int(rec_id) for rec_id, _, _, _ in data]
        rid2ts = {rid: i for i, rid in enumerate(ordered_rids)}
        x_tr = self._subset_to_long(train_set, rid2ts, V)
        x_va = self._subset_to_long(val_set,  rid2ts, V)
        x_te = self._subset_to_long(test_set, rid2ts, V)
        # seen_data for scaling, = train + val
        seen_data = np.vstack([x_tr, x_va])


        ts_id_to_name = {rid2ts[r]: r for r in rid2ts}
        val_id_to_name = {j: f"label_{j}" for j in range(V)}

        return dict(
            train_data=x_tr,
            valid_data=x_va,
            test_data=x_te,
            seen_data=seen_data,
            static_feature=np.array([], dtype=object),
            dynamic_feature=np.array(self.DYNAMIC_FEATURES, dtype=object),
            ts_id_to_name=np.array([ts_id_to_name], dtype=object),
            val_id_to_name=np.array([val_id_to_name], dtype=object),
        )

    process_imputation = process_prediction

    def process_classification(self):
        """Not available (no labels)."""
        raise NotImplementedError("Classification is not implemented for USHCNProcessor.")
