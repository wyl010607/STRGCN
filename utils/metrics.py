import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
from sklearn.preprocessing import label_binarize

def get_mae(y_true, y_pred):
    """
    Mean Absolute Error (MAE)
    Formula: MAE = mean(|y_true - y_pred|)
    """
    return np.mean(np.abs(y_true - y_pred))


def get_mse(y_true, y_pred):
    """
    Mean Squared Error (MSE)
    Formula: MSE = mean((y_true - y_pred)^2)
    """
    return ((y_true - y_pred) ** 2).mean()


def get_rmse(y_true, y_pred):
    """
    Root Mean Squared Error (RMSE)
    Formula: RMSE = sqrt(mean((y_true - y_pred)^2))
    """
    return np.sqrt(((y_true - y_pred) ** 2).mean())


def get_mape(y_true, y_pred):
    """
    Mean Absolute Percentage Error (MAPE)
    Formula: MAPE = mean(|(y_true - y_pred) / y_true|) * 100
    """
    non_zero_mask = y_true != 0
    y_true_masked = y_true[non_zero_mask]
    y_pred_masked = y_pred[non_zero_mask]
    mape = np.mean(np.abs((y_true_masked - y_pred_masked) / y_true_masked)) * 100
    return mape


def get_rmspe(y_true, y_pred):
    """
    Root Mean Square Percentage Error (RMSPE)
    Formula: RMSPE = sqrt(mean(((y_true - y_pred) / y_true)^2)) * 100
    """
    non_zero_mask = y_true != 0
    rmspe = (
        np.sqrt(
            np.mean(
                np.square(
                    (y_true[non_zero_mask] - y_pred[non_zero_mask])
                    / y_true[non_zero_mask]
                )
            )
        )
        * 100
    )
    return rmspe


def get_bce(y_true, y_pred):
    """
    Binary Cross-Entropy (BCE)
    Formula: BCE = -y_true*log(y_pred) - (1 - y_true)*log(1 - y_pred)
    """
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)


def get_cce(y_true, y_pred):
    """
    Categorical Cross-Entropy (CCE)
    Formula: CCE = -sum(y_true*log(y_pred))
    """
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.sum(y_true * np.log(y_pred), axis=1).mean()


def get_hinge(y_true, y_pred):
    """
    Hinge Loss
    Formula: Hinge = mean(max(0, 1 - y_true*y_pred))
    """
    return np.mean(np.maximum(1 - y_true * y_pred, 0))


def get_masked_mae(y_true, y_pred, mask, reduce="mean", eps=1e-8):
    """
    Mean Absolute Error (MAE), Note we take the average across variables
    Formula: MAE = mean(|y_true - y_pred|)
    """
    D = y_true.shape[-1]
    y_true_2d = y_true.reshape(-1, D)
    y_pred_2d = y_pred.reshape(-1, D)
    mask_2d = mask.reshape(-1, D)

    num = np.sum(np.abs(y_pred_2d - y_true_2d) * mask_2d, axis=0)
    den = np.sum(mask_2d, axis=0)

    mae_list = [num[d] / max(den[d], eps) for d in range(D) if den[d] > 0]

    if reduce == "mean":
        return float(np.mean(mae_list))
    elif reduce == "sum":
        return float(np.sum(mae_list))
    else:
        raise ValueError("reduce must be 'mean' or 'sum'")


def get_masked_mae_long(y_true, y_pred, mask, var_idx, reduce="mean", eps=1e-8):
    """
    Mean Absolute Error (MAE), Note we take the average across variables
    Input shape: (L,), with var_idx (L,) specifying the variable id.
    Formula: MAE = mean(|y_true - y_pred|)
    """
    mae_list = []
    for vid in np.unique(var_idx):
        sel = var_idx == vid
        w = mask[sel]
        den = np.sum(w)
        if den <= 0:
            continue
        e = np.abs(y_pred[sel] - y_true[sel])
        mae = np.sum(e * w) / max(den, eps)
        mae_list.append(mae)

    if reduce == "mean":
        return float(np.mean(mae_list))
    elif reduce == "sum":
        return float(np.sum(mae_list))
    else:
        raise ValueError("reduce must be 'mean' or 'sum'")


def get_masked_mse(y_true, y_pred, mask, reduce="mean", eps=1e-8):
    """
    Mean Squared Error (MSE), Note we take the average across variables
    Formula: MSE = mean((y_true - y_pred)^2)
    """
    D = y_true.shape[-1]
    y_true_2d = y_true.reshape(-1, D)
    y_pred_2d = y_pred.reshape(-1, D)
    mask_2d = mask.reshape(-1, D)

    num = np.sum(((y_pred_2d - y_true_2d) ** 2) * mask_2d, axis=0)
    den = np.sum(mask_2d, axis=0)

    mse_list = [num[d] / max(den[d], eps) for d in range(D) if den[d] > 0]

    if reduce == "mean":
        return float(np.mean(mse_list))
    elif reduce == "sum":
        return float(np.sum(mse_list))
    else:
        raise ValueError("reduce must be 'mean' or 'sum'")


def get_masked_mse_long(y_true, y_pred, mask, var_idx, reduce="mean", eps=1e-8):
    """
    Mean Squared Error (MSE), Note we take the average across variables
    Formula: MSE = mean((y_true - y_pred)^2)
    """

    mse_list = []
    for vid in np.unique(var_idx):
        sel = var_idx == vid
        w = mask[sel]
        den = np.sum(w)
        if den <= 0:
            continue
        e2 = (y_pred[sel] - y_true[sel]) ** 2
        mse = np.sum(e2 * w) / max(den, eps)
        mse_list.append(mse)
    if reduce == "mean":
        return float(np.mean(mse_list))
    elif reduce == "sum":
        return float(np.sum(mse_list))
    else:
        raise ValueError("reduce must be 'mean' or 'sum'")


def get_masked_rmse(y_true, y_pred, mask, reduce="mean", eps=1e-8):
    """
    Root Mean Squared Error (RMSE), Note we take the average across variables
    Formula: RMSE = sqrt(mean((y_true - y_pred)^2))
    """
    D = y_true.shape[-1]
    # Flatten all dimensions except the last one
    y_true_2d = y_true.reshape(-1, D)
    y_pred_2d = y_pred.reshape(-1, D)
    mask_2d = mask.reshape(-1, D)

    # Per-variable numerator and denominator
    num = np.sum(((y_pred_2d - y_true_2d) ** 2) * mask_2d, axis=0)  # (D,)
    den = np.sum(mask_2d, axis=0)  # (D,)

    mse_list = [num[d] / max(den[d], eps) for d in range(D) if den[d] > 0]

    if reduce == "mean":
        return float(np.sqrt(np.mean(mse_list)))
    elif reduce == "sum":
        return float(np.sqrt(np.sum(mse_list)))
    else:
        raise ValueError("reduce must be either 'mean' or 'sum'")


def get_masked_rmse_long(y_true, y_pred, mask, var_idx, reduce="mean", eps=1e-8):
    """
    Root Mean Squared Error (RMSE), Note we take the average across variables
    Input shape: (L,), with var_idx (L,) specifying which variable each sample belongs to.
    Formula: RMSE = sqrt(mean((y_true - y_pred)^2))
    """
    mse_list = []
    for vid in np.unique(var_idx):
        sel = var_idx == vid
        w = mask[sel]
        den = np.sum(w)
        if den <= 0:
            continue
        e2 = (y_pred[sel] - y_true[sel]) ** 2
        mse = np.sum(e2 * w) / max(den, eps)
        mse_list.append(mse)

    if reduce == "mean":
        return float(np.sqrt(np.mean(mse_list)))
    elif reduce == "sum":
        return float(np.sqrt(np.sum(mse_list)))
    else:
        raise ValueError("reduce must be either 'mean' or 'sum'")


def get_masked_mape(y_true, y_pred, mask, reduce="mean", eps=1e-8):
    """
    Mean Absolute Percentage Error (MAPE), Note we take the average across variables
    Formula: MAPE = mean(|(y_true - y_pred) / y_true|) * 100
    """
    D = y_true.shape[-1]
    y_true_2d = y_true.reshape(-1, D)
    y_pred_2d = y_pred.reshape(-1, D)
    mask_2d = mask.reshape(-1, D)

    mape_list = []
    for d in range(D):
        sel_true = y_true_2d[:, d]
        sel_pred = y_pred_2d[:, d]
        sel_mask = mask_2d[:, d]
        non_zero = sel_true != 0
        w = sel_mask[non_zero]
        if np.sum(w) <= 0:
            continue
        ape = (
            np.abs((sel_pred[non_zero] - sel_true[non_zero]) / sel_true[non_zero])
            * w
            * 100
        )
        mape = np.sum(ape) / np.sum(w)
        mape_list.append(mape)

    if reduce == "mean":
        return float(np.mean(mape_list))
    elif reduce == "sum":
        return float(np.sum(mape_list))
    else:
        raise ValueError("reduce must be 'mean' or 'sum'")


def get_masked_mape_long(y_true, y_pred, mask, var_idx, reduce="mean", eps=1e-8):
    """
    Mean Absolute Percentage Error (MAPE), Note we take the average across variables
    Formula: MAPE = mean(|(y_true - y_pred) / y_true|) * 100
    """
    mape_list = []
    for vid in np.unique(var_idx):
        sel = var_idx == vid
        sel_true = y_true[sel]
        sel_pred = y_pred[sel]
        sel_mask = mask[sel]
        non_zero = sel_true != 0
        w = sel_mask[non_zero]
        if np.sum(w) <= 0:
            continue
        ape = (
            np.abs((sel_pred[non_zero] - sel_true[non_zero]) / sel_true[non_zero])
            * w
            * 100
        )
        mape = np.sum(ape) / np.sum(w)
        mape_list.append(mape)

    if reduce == "mean":
        return float(np.mean(mape_list))
    elif reduce == "sum":
        return float(np.sum(mape_list))
    else:
        raise ValueError("reduce must be 'mean' or 'sum'")


def get_accuracy(y_true, y_pred):
    """
    Accuracy
    - 二分类/多分类均可。
    - y_pred 可以是：
        (N,) 已离散的类别标签 或 二分类分数/概率(自动0.5阈值)；
        (N, C) 每类分数/概率（取 argmax）。
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred)

    if y_pred.ndim == 2:
        # (N,1) 视作二分类概率/分数；(N,C>=2) 取 argmax
        if y_pred.shape[1] == 1:
            y_hat = (y_pred[:, 0] >= 0.5).astype(int)
        else:
            y_hat = np.argmax(y_pred, axis=1)
    else:
        # (N,) 若非{0,1}集合，按0.5阈值二分类
        uniq = np.unique(y_pred)
        if set(uniq).issubset({0, 1}):
            y_hat = y_pred.astype(int)
        else:
            y_hat = (y_pred >= 0.5).astype(int)

    return float(accuracy_score(y_true, y_hat))


def get_auroc(y_true, y_score):
    """
    AUROC
    - 二分类：y_score 为正类分数/概率 (N,)、或 (N,2) 时取[:,1]
    - 多分类：y_score 形状 (N,C) 时使用 OVR + macro
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_score = np.asarray(y_score)

    if y_score.ndim == 2 and y_score.shape[1] > 2:
        return float(roc_auc_score(y_true, y_score, multi_class="ovr", average="macro"))
    elif y_score.ndim == 2 and y_score.shape[1] == 2:
        return float(roc_auc_score(y_true, y_score[:, 1]))
    else:
        return float(roc_auc_score(y_true, y_score))


def get_auprc(y_true, y_score):
    """
    AUPRC (Average Precision)
    - 二分类：y_score 为正类分数/概率 (N,)、或 (N,2) 时取[:,1]
    - 多分类：y_score 为 (N,C) 时，对 y_true 做 one-vs-rest 后 macro 平均
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_score = np.asarray(y_score)

    if y_score.ndim == 2 and y_score.shape[1] > 2:
        classes = np.unique(y_true)
        Y = label_binarize(y_true, classes=classes)
        return float(average_precision_score(Y, y_score, average="macro"))
    elif y_score.ndim == 2 and y_score.shape[1] == 2:
        return float(average_precision_score(y_true, y_score[:, 1]))
    else:
        return float(average_precision_score(y_true, y_score))
