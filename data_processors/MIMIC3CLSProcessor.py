# mimic3_cls_processor.py
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any

from data_processors.abs import AbstractDataProcessor


class MIMIC3CLSProcessor(AbstractDataProcessor):

    DYNAMIC_FEATURES: List[str] = []
    STATIC_FEATURES: List[str] = []  # MIMIC-III 
    CLASS_LABELS: List[str] = ["Label"]
    REG_LABELS: List[str] = []

    def __init__(self, *args, **kwargs):

        super().__init__("mimic3cls", *args, **kwargs)
        self.base_path = Path(self.raw_data_path or ".")

    # ---------------- I/O ---------------- #
    def _load_split_arrays(self):
        bx = self.base_path
        Ptrain = np.load(bx / "mimic3_train_x.npy", allow_pickle=True)
        Pval   = np.load(bx / "mimic3_val_x.npy",   allow_pickle=True)
        Ptest  = np.load(bx / "mimic3_test_x.npy",  allow_pickle=True)
        ytrain = np.load(bx / "mimic3_train_y.npy", allow_pickle=True).reshape(-1)
        yval   = np.load(bx / "mimic3_val_y.npy",   allow_pickle=True).reshape(-1)
        ytest  = np.load(bx / "mimic3_test_y.npy",  allow_pickle=True).reshape(-1)
        return Ptrain, Pval, Ptest, ytrain.astype(np.int64), yval.astype(np.int64), ytest.astype(np.int64)

    def _make_dyn_names(self, V: int) -> List[str]:
        return [f"var_{j}" for j in range(V)]

    # --------------- convert -------------- #
    def _subset_to_long_array(self, subset: np.ndarray) -> np.ndarray:

        rows = []
        for ts_id, rec in enumerate(subset):
            #  tt, vals, mask, Tlen idx=1: tt, idx=2: vals, idx=3: mask, idx=4: Tlen
            tt = rec[1].astype(np.float64)
            vals = rec[2]
            mask = rec[3]
            tlen = int(rec[4])
            T, V = vals.shape
            T = min(T, int(tlen))
            tt = tt[:T]
            vals = vals[:T]
            mask = mask[:T]

            for j in range(V):
                xj = vals[:, j].astype(np.float64)
                if mask is not None and isinstance(mask, np.ndarray) and mask.shape == vals.shape:
                    obs = mask[:, j] > 0.5
                else:
                    obs = ~np.isnan(xj)

                if not np.any(obs):
                    continue
                tj = tt[obs]
                vj = xj[obs]
                ts_col = np.full_like(tj, fill_value=float(ts_id), dtype=np.float64)
                val_col = np.full_like(tj, fill_value=float(j), dtype=np.float64)
                rows.append(np.stack([ts_col, val_col, tj, vj], axis=1))

        if not rows:
            return np.zeros((0, 4), dtype=np.float64)
        return np.concatenate(rows, axis=0).astype(np.float64)

    # ---------------- main ---------------- #
    def process_classification(self):
        Ptrain, Pval, Ptest, ytr, yva, yte = self._load_split_arrays()

        V = Ptrain[0][2].shape[1]
        self.DYNAMIC_FEATURES = [f"var_{j}" for j in range(V)]
        self.STATIC_FEATURES = []  # MIMIC-III 

        x_tr = self._subset_to_long_array(Ptrain)
        x_va = self._subset_to_long_array(Pval)
        x_te = self._subset_to_long_array(Ptest)

        train_static_data = np.zeros((len(Ptrain), 0), dtype=np.float64)
        valid_static_data = np.zeros((len(Pval),   0), dtype=np.float64)
        test_static_data  = np.zeros((len(Ptest),  0), dtype=np.float64)

        val_id_to_name = {i: name for i, name in enumerate(self.DYNAMIC_FEATURES)}
        ts_id_to_name = {i: i for i in range(len(Ptrain) + len(Pval) + len(Ptest))}

        print(f"MIMIC3 splits: train={len(Ptrain)} val={len(Pval)} test={len(Ptest)}")
        print("Long shapes:", x_tr.shape, x_va.shape, x_te.shape)

        return dict(
            # dynamics (long)
            train_data=x_tr,
            valid_data=x_va,
            test_data=x_te,

            # statics (wide, split; empty because d_static=0)
            train_static_data=train_static_data,
            valid_static_data=valid_static_data,
            test_static_data=test_static_data,

            # names / maps
            static_feature=np.array(self.STATIC_FEATURES, dtype=object),
            dynamic_feature=np.array(self.DYNAMIC_FEATURES, dtype=object),
            ts_id_to_name=np.array([ts_id_to_name], dtype=object),
            val_id_to_name=np.array([val_id_to_name], dtype=object),

            # labels (already split)
            train_label=ytr,
            valid_label=yva,
            test_label=yte,

            # meta
            meta=dict(
                base_path=str(self.base_path),
                time_unit="hour", 
                has_static=False,
            ),
        )


# ---------------- demo ---------------- #
if __name__ == "__main__":
    p = MIMIC3CLSProcessor(
        task_name="classification",
        allowed_tasks=["classification"],
        global_args={}
    )
    res = p.process()
    for k, v in res.items():
        if isinstance(v, np.ndarray):
            print(k, v.shape, v.dtype)
        else:
            print(k, type(v))
