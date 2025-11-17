# mimic3_processor.py
import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from data_processors.abs import AbstractDataProcessor


class MIMIC3Processor(AbstractDataProcessor):
    """
    MIMIC-III processor (PT-only):
    - Read directly from mimic.pt (list of tuples: (record_id, tt[h], vals[T,V], mask[T,V])).
    """

    DYNAMIC_FEATURES: List[str] = ["value"]
    STATIC_FEATURES: List[str] = []
    CLASS_LABELS: List[str] = ["Label"]  # 0=alive, 1=death
    REG_LABELS: List[str] = []

    # ------------------------------------------------------------ #
    # Constructor
    # ------------------------------------------------------------ #
    def __init__(self, *args, **kwargs):
        super().__init__("mimic3", *args, **kwargs)

        self.pt_data_path = self.raw_data_path / "mimic-iii-ext-tpatchgnn-1.0.0"/ "mimic.pt"
        self.varmap_file = self.raw_data_path / "mimic-iii-ext-tpatchgnn-1.0.0"/ "variable_dict.csv"

    def load_pt_data(self):
        data_ = torch.load(self.pt_data_path, map_location="cpu")
        vm = pd.read_csv(self.varmap_file)
        var_names = vm["LABEL"].astype(str).tolist()
        self.DYNAMIC_FEATURES = var_names
        return data_, var_names

    def split_data(self, data) -> Tuple[list, list, list]:
        """
          seen, test = train_test_split(total, train_size=0.8, random_state=42, shuffle=True)
          train, val  = train_test_split(seen,  train_size=0.75, random_state=42, shuffle=False)
        """
        from sklearn.model_selection import train_test_split as tts
        seen, test = tts(data, train_size=0.8, random_state=42, shuffle=True)
        train, val = tts(seen, train_size=0.75, random_state=42, shuffle=False)
        return train, val, test

    def _build_id_maps(
        self, data, var_names: List[str]
    ) -> Tuple[Dict[int, int], Dict[int, str], Dict[int, str]]:
        """
        Build:
          - rid2ts: record_id -> ts_id (using the order of all data)
          - ts_id_to_name: ts_id -> record_id (as str for consistency with earlier code)
          - val_id_to_name: val_id -> variable name
        """
        ordered_rids = np.asarray([int(rec_id) for rec_id, _, _, _ in data], dtype=int)  # keep stored order
        rid2ts = {int(rid): i for i, rid in enumerate(ordered_rids)}
        ts_id_to_name = {i: int(rid) for i, rid in enumerate(ordered_rids)}
        val_id_to_name = {i: name for i, name in enumerate(var_names)}
        return rid2ts, ts_id_to_name, val_id_to_name

    def _subset_to_long_array(
        self,
        subset: list,
        rid2ts: Dict[int, int],
        var_names: List[str],
    ) -> np.ndarray:

        V = len(var_names)
        rows = []
        for rec_id, tt, vals, mask in subset:
            ts_id = rid2ts[int(rec_id)]
            t = np.asarray(tt * 3600, dtype=np.int32)        # [T]
            x = np.asarray(vals, dtype=np.float64)      # [T, V]
            m = np.asarray(mask, dtype=np.float64)      # [T, V]
            # For each variable column, append observed points
            for j in range(V):
                obs = m[:, j] > 0.5
                if not np.any(obs):
                    continue
                # Stack as [ts_id, val_id, timestamp, value]
                tj = t[obs]
                xj = x[obs, j]
                ts_col = np.full_like(tj, fill_value=float(ts_id), dtype=np.float64)
                val_col = np.full_like(tj, fill_value=float(j), dtype=np.float64)
                rows.append(np.stack([ts_col, val_col, tj, xj], axis=1))
        if not rows:
            return np.zeros((0, 4), dtype=np.float64)
        return np.concatenate(rows, axis=0).astype(np.float64)

    def process_prediction(self):
        data, var_names = self.load_pt_data()

        train_set, val_set, test_set = self.split_data(data)

        test_record_ids = [int(rec_id) for rec_id, _, _, _ in test_set]
        print("Dataset n_samples:", len(data), len(train_set), len(val_set), len(test_set))
        print("Test record ids (first 20):", test_record_ids[:20])
        print("Test record ids (last 20):", test_record_ids[-20:])

        rid2ts, ts_id_to_name, val_id_to_name = self._build_id_maps(data, var_names)

        x_tr = self._subset_to_long_array(train_set, rid2ts, var_names)
        x_va = self._subset_to_long_array(val_set, rid2ts, var_names)
        x_te = self._subset_to_long_array(test_set, rid2ts, var_names)

        seen_data = np.vstack([x_tr, x_va])

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

    # Imputation uses the same inputs/outputs as prediction
    process_imputation = process_prediction

    def process_classification(self):
        """
        Not implemented: we do not have supervised labels for records.
        """
        raise NotImplementedError(
            "Classification is not implemented because per-record labels are unavailable."
        )


# ---------------- demo ---------------- #
if __name__ == "__main__":
    global_args = {
        "data_name": "mimic3",
    }
    p = MIMIC3Processor(
        task_name="prediction",
        allowed_tasks=["prediction", "imputation", "classification"],
        global_args=global_args
    )
    res = p.process()
    for k, v in res.items():
        print(k, v.shape if isinstance(v, np.ndarray) else type(v))
