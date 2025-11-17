# p12_cls_processor.py
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any

from data_processors.abs import AbstractDataProcessor


class P12CLSProcessor(AbstractDataProcessor):
    """
    P12 classification processor (PTdict_list-based):
    Read directly from P12 processed_data/PTdict_list.npy and processed_data/arr_outcomes.npy
    """

    DYNAMIC_FEATURES: List[str] = []
    STATIC_FEATURES: List[str] = []
    CLASS_LABELS: List[str] = ["Label"]
    REG_LABELS: List[str] = []

    def __init__(self, *args, split_idx: int = 1, **kwargs):
        super().__init__("P12", *args, **kwargs)
        self.split_idx = int(split_idx)

        self.base_path = Path(self.raw_data_path or ".")
        self.processed_dir = self.base_path / "processed_data"
        self.splits_dir = self.base_path / "splits"

    # ---------------- I/O ---------------- #
    def _load_ptdict_list_and_labels(self) -> Tuple[np.ndarray, np.ndarray]:
        Pdict_list = np.load(self.processed_dir / "PTdict_list.npy", allow_pickle=True)
        arr_outcomes = np.load(self.processed_dir / "arr_outcomes.npy", allow_pickle=True)
        y = arr_outcomes[:, -1].astype(np.int64).reshape(-1)  # 0/1
        return Pdict_list, y

    def _load_official_split(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        idx_train, idx_val, idx_test = np.load(
            self.splits_dir / f"phy12_split{self.split_idx}.npy",
            allow_pickle=True
        )
        return idx_train.astype(int), idx_val.astype(int), idx_test.astype(int)

    # -------------- metadata ------------- #
    def _infer_feature_names(self, example: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        T, F = example["arr"].shape
        D = len(example["extended_static"])
        dyn_names = [f"var_{j}" for j in range(F)]
        static_names = [f"static_{j}" for j in range(D)]
        return dyn_names, static_names

    def _build_id_maps_all(self, N_total: int):
        rid2ts = {int(r): int(r) for r in range(N_total)}
        ts2rid = {int(r): int(r) for r in range(N_total)}
        return rid2ts, ts2rid

    # --------------- convert -------------- #
    def _subset_to_long_array(
        self,
        subset_indices: np.ndarray,
        Pdict_list: np.ndarray,
        rid2ts: Dict[int, int],
        keep_nonpositive: bool = False,
    ) -> np.ndarray:
        rows: List[np.ndarray] = []
        for rid in subset_indices:
            ex = Pdict_list[int(rid)]
            arr = np.asarray(ex["arr"], dtype=np.float64)               # (T,F)
            tim = np.asarray(ex["time"], dtype=np.float64).reshape(-1)  # (T,), minutes
            ts_id = float(rid2ts[int(rid)])

            T, F = arr.shape
            for j in range(F):
                xj = arr[:, j]
                obs = np.ones_like(xj, dtype=bool) if keep_nonpositive else (xj > 0)
                if not np.any(obs):
                    continue
                tj = tim[obs]
                vj = xj[obs]
                ts_col = np.full_like(tj, fill_value=ts_id, dtype=np.float64)
                val_col = np.full_like(tj, fill_value=float(j), dtype=np.float64)
                rows.append(np.stack([ts_col, val_col, tj, vj], axis=1))

        if not rows:
            return np.zeros((0, 4), dtype=np.float64)
        return np.concatenate(rows, axis=0).astype(np.float64)

    # ---------------- main ---------------- #
    def process_classification(self):
        # 1) load
        Pdict_list, y_all = self._load_ptdict_list_and_labels()
        idx_tr, idx_va, idx_te = self._load_official_split()

        # 2) names
        dyn_names, static_names = self._infer_feature_names(Pdict_list[0])
        self.DYNAMIC_FEATURES = dyn_names
        self.STATIC_FEATURES = static_names

        # 3) id maps
        N_total = len(Pdict_list)
        rid2ts, ts2rid = self._build_id_maps_all(N_total)

        # 4) split ids (ts_id)
        ts_tr = np.asarray([rid2ts[int(r)] for r in idx_tr], dtype=np.int64)
        ts_va = np.asarray([rid2ts[int(r)] for r in idx_va], dtype=np.int64)
        ts_te = np.asarray([rid2ts[int(r)] for r in idx_te], dtype=np.int64)

        # 5) long dynamics per split
        x_tr = self._subset_to_long_array(idx_tr, Pdict_list, rid2ts)
        x_va = self._subset_to_long_array(idx_va, Pdict_list, rid2ts)
        x_te = self._subset_to_long_array(idx_te, Pdict_list, rid2ts)

        # 6) static wide table per split（直接分好）
        D = len(Pdict_list[0]["extended_static"])
        S_all = np.zeros((N_total, D), dtype=np.float64)
        for rid in range(N_total):
            S_all[rid2ts[rid], :] = np.asarray(Pdict_list[rid]["extended_static"], dtype=np.float64)
        train_static_data = S_all[ts_tr]
        valid_static_data = S_all[ts_va]
        test_static_data  = S_all[ts_te]

        # 7) labels （全量+按 split）
        labels_per_ts = np.asarray([y_all[ts2rid[ts]] for ts in range(N_total)], dtype=np.int64)
        train_label = y_all[idx_tr]
        valid_label = y_all[idx_va]
        test_label  = y_all[idx_te]

        # 8) maps
        val_id_to_name = {i: name for i, name in enumerate(self.DYNAMIC_FEATURES)}
        ts_id_to_name = {i: i for i in range(N_total)}

        # 9) log
        print(f"P12 total N={N_total} | split sizes: "
              f"train={len(idx_tr)} val={len(idx_va)} test={len(idx_te)}")

        return dict(
            # dynamics (long)
            train_data=x_tr,              # (Nt,4)
            valid_data=x_va,
            test_data=x_te,
            # statics (wide, split)
            train_static_data=train_static_data,   # (N_tr, D)
            valid_static_data=valid_static_data,   # (N_va, D)
            test_static_data=test_static_data,     # (N_te, D)

            static_feature=np.array(self.STATIC_FEATURES, dtype=object),
            dynamic_feature=np.array(self.DYNAMIC_FEATURES, dtype=object),
            ts_id_to_name=np.array([ts_id_to_name], dtype=object),
            val_id_to_name=np.array([val_id_to_name], dtype=object),

            labels_per_ts=labels_per_ts,
            train_label=train_label,
            valid_label=valid_label,
            test_label=test_label,

            train_ts_ids=ts_tr,            # (N_tr,)
            valid_ts_ids=ts_va,
            test_ts_ids=ts_te,
            meta=dict(
                base_path=str(self.base_path),
                split_idx=int(self.split_idx),
                time_unit="minute",
            ),
        )


# ---------------- demo ---------------- #
if __name__ == "__main__":
    global_args = {}
    p = P12CLSProcessor(
        task_name="classification",
        allowed_tasks=["classification"],
        global_args=global_args,
        split_idx=1,
    )
    res = p.process()
    for k, v in res.items():
        if isinstance(v, np.ndarray):
            print(k, v.shape, v.dtype)
        else:
            print(k, type(v))
