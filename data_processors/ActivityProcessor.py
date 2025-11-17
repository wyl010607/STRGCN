# activity_processor.py
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from sklearn.model_selection import train_test_split

from data_processors.abs import AbstractDataProcessor


class ActivityProcessor(AbstractDataProcessor):
    """
    CSV-like (txt) Activity processor aligned with PersonActivity (B):
    - Read raw "ConfLongDemo_JSI.txt" directly (no .pt cache, no chunking).
    """

    DYNAMIC_FEATURES: List[str] = ["x", "y", "z"]
    STATIC_FEATURES: List[str] = []
    CLASS_LABELS: List[str] = ["label"]  
    REG_LABELS: List[str] = []

    def __init__(self, *args, **kwargs):
        super().__init__("activity", *args, **kwargs)
        # Raw file
        self.data_file: Path = self.raw_data_path / "ConfLongDemo_JSI.txt"


        self.tag_ids: List[str] = [
            "010-000-024-033",  # ANKLE_LEFT
            "010-000-030-096",  # ANKLE_RIGHT
            "020-000-033-111",  # CHEST
            "020-000-032-221",  # BELT
        ]
        self.tag_alias: Dict[str, str] = {
            "010-000-024-033": "ANKLE_LEFT",
            "010-000-030-096": "ANKLE_RIGHT",
            "020-000-033-111": "CHEST",
            "020-000-032-221": "BELT",
        }
        self.tag_dict: Dict[str, int] = {tid: i for i, tid in enumerate(self.tag_ids)}
        self.axes: List[str] = ["x", "y", "z"]

    # ---------------------------- splitting ---------------------------- #
    def split_data(self, data: List[tuple]) -> Tuple[List[tuple], List[tuple], List[tuple]]:
        """
        Two-step split (seed=42) to match B:
          seen, test = train_test_split(data,  train_size=0.8,  random_state=42, shuffle=True)
          train, val  = train_test_split(seen, train_size=0.75, random_state=42, shuffle=False)
        """
        seen, test = train_test_split(data, train_size=0.8, random_state=42, shuffle=True)
        train, val = train_test_split(seen, train_size=0.75, random_state=42, shuffle=False)
        return train, val, test

    def _subset_to_long(self, subset: List[tuple], rid2ts: Dict[str, int], V: int) -> np.ndarray:
        """
        Convert a subset to a long ndarray [ts_id, val_id, timestamp, value] (float64),
        keeping only entries where mask == 1.
        """
        rows = []
        for rec_id, tt, vals, mask in subset:
            ts_id = float(rid2ts[rec_id])
            # vals/mask: (T, V), tt: (T,)
            for j in range(V):
                obs = mask[:, j] > 0.5
                if not np.any(obs):
                    continue
                tj = tt[obs].astype(np.float64)          # ms (float64)
                xj = vals[obs, j].astype(np.float64)     # value
                ts_col = np.full_like(tj, ts_id, dtype=np.float64)
                val_col = np.full_like(tj, float(j), dtype=np.float64)
                rows.append(np.stack([ts_col, val_col, tj, xj], axis=1))
        if not rows:
            return np.zeros((0, 4), dtype=np.float64)
        return np.concatenate(rows, axis=0)

    # --------------------------- public API --------------------------- #
    def load_data(self) -> Tuple[List[tuple], int]:
        """
        Parse ConfLongDemo_JSI.txt and build a list of tuples:
          (record_id:str, tt_ms:np.ndarray[T], vals:np.ndarray[T,V], mask:np.ndarray[T,V])
        """
        if not self.data_file.exists():
            raise FileNotFoundError(f"File not found: {self.data_file}")

        records: List[tuple] = []
        num_tags = len(self.tag_ids)
        V = num_tags * 3  # x,y,z per tag

        def _flush_current(rec_id, tt_list, vals_list, mask_list):
            if rec_id is None:
                return
            # Convert to arrays (T, V)
            tt_arr = np.asarray(tt_list, dtype=np.float64)  # ms
            vals_arr = np.stack(vals_list, axis=0).reshape(len(tt_list), V).astype(np.float64)
            mask_arr = np.stack(mask_list, axis=0).reshape(len(tt_list), V).astype(np.float64)
            # Ensure sorted by time (stable sort)
            order = np.argsort(tt_arr, kind="mergesort")
            records.append((rec_id, tt_arr[order], vals_arr[order], mask_arr[order]))

        with self.data_file.open("r", encoding="utf-8") as f:
            prev_record = None
            first_tp = None
            prev_time_ms = None

            tt_list, vals_list, mask_list, nobs_list = [], [], [], []

            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                # Format in file: record_id, tag_id, time, date, val1, val2, val3, label
                parts = raw.split(",")
                if len(parts) != 8:
                    # Skip malformed line
                    continue

                cur_record, tag_id, time_str, date_str, v1, v2, v3, label = parts
                if tag_id not in self.tag_ids:
                    continue

                # Convert time to ms since the first timestamp of this record
                t_raw = float(time_str)
                if cur_record != prev_record:
                    # Flush previous record
                    _flush_current(prev_record, tt_list, vals_list, mask_list)

                    # Init new record accumulators
                    prev_record = cur_record
                    first_tp = t_raw
                    prev_time_ms = 0.0

                    tt_list = [0.0]
                    vals_list = [np.zeros((num_tags, 3), dtype=np.float32)]
                    mask_list = [np.zeros((num_tags, 3), dtype=np.float32)]
                    nobs_list = [np.zeros((num_tags,), dtype=np.float32)]
                # Compute ms
                time_ms = float(np.round((t_raw - first_tp) / 1e4))

                # Start a new row if time changes
                if time_ms != prev_time_ms:
                    tt_list.append(time_ms)
                    vals_list.append(np.zeros((num_tags, 3), dtype=np.float32))
                    mask_list.append(np.zeros((num_tags, 3), dtype=np.float32))
                    nobs_list.append(np.zeros((num_tags,), dtype=np.float32))
                    prev_time_ms = time_ms

                tag_idx = self.tag_dict[tag_id]
                value_vec = np.array([float(v1), float(v2), float(v3)], dtype=np.float32)

                # Average if multiple observations for the same (time, tag)
                n_obs = nobs_list[-1][tag_idx]
                if n_obs > 0:
                    prev_val = vals_list[-1][tag_idx].copy()
                    new_val = (prev_val * n_obs + value_vec) / (n_obs + 1.0)
                    vals_list[-1][tag_idx] = new_val
                else:
                    vals_list[-1][tag_idx] = value_vec

                mask_list[-1][tag_idx] = 1.0
                nobs_list[-1][tag_idx] = n_obs + 1.0

            # Flush last record
            _flush_current(prev_record, tt_list, vals_list, mask_list)

        # Fill dynamic feature names in the same order (tag-major, then axis x,y,z)
        dyn_names: List[str] = []
        for tag in self.tag_ids:
            alias = self.tag_alias.get(tag, tag)
            for ax in self.axes:
                dyn_names.append(f"{ax}_{alias}")
        self.DYNAMIC_FEATURES = dyn_names

        return records, V

    def process_prediction(self):
        data, V = self.load_data()

        # Split exactly like B
        train_set, val_set, test_set = self.split_data(data)

        test_ids = [rec_id for rec_id, _, _, _ in test_set]
        print("Dataset n_samples:", len(data), len(train_set), len(val_set), len(test_set))
        print("Test record ids (first 20):", test_ids[:20])
        print("Test record ids (last 20):", test_ids[-20:])

        # Global id maps (keep consistent ts_id across splits)
        ordered = [rec_id for rec_id, _, _, _ in data]
        rid2ts =  {rid: i for i, rid in enumerate(ordered)}
        ts_id_to_name = {rid2ts[r]: r for r in rid2ts}
        val_id_to_name = {}
        for tag_idx, tag in enumerate(self.tag_ids):
            alias = self.tag_alias.get(tag, tag)
            for ax_idx, ax in enumerate(self.axes):
                j = tag_idx * 3 + ax_idx
                val_id_to_name[j] = f"{ax}_{alias}"

        # Convert subsets to long ndarrays
        x_tr = self._subset_to_long(train_set, rid2ts, V)
        x_va = self._subset_to_long(val_set,  rid2ts, V)
        x_te = self._subset_to_long(test_set, rid2ts, V)
        # seen_data for scaling, = train + val
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

    # Imputation uses the same inputs/outputs as prediction here
    process_imputation = process_prediction

    def process_classification(self):
        """Not implemented (this processor focuses on forecasting-style long outputs)."""
        raise NotImplementedError("Classification is not implemented for ActivityProcessor.")
