import numpy as np
import warnings

import tqdm
from tqdm import tqdm
import torch
from trainers.abs import AbstractTrainer
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast
import utils.metrics as metrics_module


class IrrFCGraphForecastTrainer(AbstractTrainer):
    """
    A Trainer subclass for Irr data forecasting.
    """

    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        scaler,
        model_save_dir_path,
        result_save_dir_path,
        max_epoch_num,
        max_iter_num=None,
        enable_early_stop=False,
        early_stop_patience=5,
        eval_metric_key="loss",
        eval_metric_min_is_best=True,
        enable_autocast=False,
        enable_distributed=False,
        do_inverse_scaling=True,
        global_args=None,
        *args,
        **kwargs,
    ):
        """
        Parameters
        ----------
        model : torch.nn.Module
            The neural network model for single step forecasting.
        optimizer : torch.optim.Optimizer
            The optimizer to use for training the model.
        scheduler : torch.optim.lr_scheduler
            The learning rate scheduler.
        scaler : Scaler
            Scaler object used for normalizing and denormalizing data.
        model_save_path : str
            Path to save the trained model.
        result_save_dir_path : str
            Directory path to save training and evaluation results.
        max_epoch_num : int
            The maximum number of epochs for training.
        enable_early_stop : bool, optional
            Whether to enable early stopping (default is False).
        early_stop_patience : int, optional
            Number of epochs with no improvement after which training will be stopped (default is 5).
        early_stop_min_is_best : bool, optional
            Flag to indicate if lower values of loss indicate better performance (default is True).
        """
        super().__init__(
            model,
            optimizer,
            scheduler,
            scaler,
            model_save_dir_path,
            result_save_dir_path,
            max_epoch_num,
            max_iter_num,
            enable_early_stop,
            early_stop_patience,
            eval_metric_key,
            eval_metric_min_is_best,
            enable_autocast,
            enable_distributed,
            do_inverse_scaling,
            global_args,
            *args,
            **kwargs,
        )
        if self.enable_autocast:
            self.grad_scaler = GradScaler()
        self.num_nodes = kwargs.get("num_nodes", None)

    def loss_func(self, y_pred, y_true, *args, **kwargs):
        mask = kwargs.get("mask", None)
        loss = ((y_pred - y_true) ** 2) * mask
        return loss.sum() / mask.sum()

    def train_one_epoch(self, data_loader):
        self.model.train()
        total_loss = 0
        if self.rank == 0:
            tqmd_ = tqdm(data_loader)
        else:
            tqmd_ = data_loader
        for batch_dict in tqmd_:
            # move_batch_data_to_device
            for key, val in batch_dict.items():
                if isinstance(val, torch.Tensor):
                    batch_dict[key] = val.to(self.device, non_blocking=True)
            batch_X_values = batch_dict["batch_value"] * (
                ~(batch_dict["batch_pred_mask"].bool())
            )
            with autocast(enabled=self.enable_autocast):
                prediction = self.model(
                    batch_value=batch_X_values,
                    batch_timestamp=batch_dict["batch_timestamp"],
                    batch_var_idx=batch_dict["batch_var_idx"],
                    batch_pred_mask=batch_dict["batch_pred_mask"],
                    batch_pad_mask=batch_dict.get("batch_pad_mask", None),
                    batch_time_id=batch_dict.get("batch_time_id", None),
                    batch_agg_mask=batch_dict.get("batch_agg_mask", None),
                ).squeeze(-1)
                loss = self.loss_func(
                    prediction,
                    batch_dict["batch_value"],
                    mask=batch_dict["batch_pred_mask"] * batch_dict["batch_pad_mask"],
                )
            self.optimizer.zero_grad()
            if self.enable_autocast:
                self.grad_scaler.scale(loss).backward()
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            if self.rank == 0:
                tqmd_.set_description("loss is {:.4f}".format(loss.item()))
            total_loss += loss.item()
        if self.enable_distributed:
            total_loss_tensor = torch.tensor(total_loss).to(self.device)
            dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
            avg_loss = total_loss_tensor.item() / (
                dist.get_world_size() * len(data_loader)
            )
        else:
            avg_loss = total_loss / len(data_loader)
        return avg_loss

    def _get_eval_result(
        self, y_pred, y_true, mask, var_idx, metrics=("mae", "rmse", "mape"), reduce="mean"
    ):
        """
        Compute evaluation metrics for the given predictions and true values.

        Parameters
        ----------
        y_pred
            Predicted values.
        y_true
            True values.
        metrics : list of str
            Metrics to be computed.

        Returns
        -------
        list
            List of computed metric values.
        """
        eval_results = []
        for metric_name in metrics:
            eval_func_name = "get_{}".format(metric_name)
            eval_func = getattr(metrics_module, eval_func_name, None)

            if eval_func is None:
                raise AttributeError(
                    f"Function '{eval_func_name}' not found in 'utils.metrics'."
                )

            result = eval_func(y_true, y_pred, mask, var_idx=var_idx, reduce=reduce)
            eval_results.append(result)

        return eval_results

    @torch.no_grad()
    def evaluate(self, data_loader, metrics, **kwargs):
        """
        Evaluate the model on the provided dataset.

        Parameters
        ----------
        data_loader : DataLoader
            DataLoader containing the evaluation data.
        metrics : list of str
            List of metric names to evaluate the model performance.
        Returns
        -------
        tuple
            train_loss : float
                The average loss on the training data.
            eval_results : list of tuple
                List of tuples of the form (metric_name, metric_value).
            y_pred : numpy.ndarray
                The predicted values.
            y_true : numpy.ndarray
                The true values.
        """
        self.model.eval()
        y_true, y_pred, y_mask, y_var_idx, tol_loss, data_num = [], [], [], [], 0, 0
        if self.rank == 0:
            tqmd_ = tqdm(data_loader)
        else:
            tqmd_ = data_loader
        for batch_dict in tqmd_:
            # move_batch_data_to_device
            for key in batch_dict:
                batch_dict[key] = batch_dict[key].to(self.device)
            batch_X_values = batch_dict["batch_value"] * (
                ~(batch_dict["batch_pred_mask"].bool())
            )
            with autocast(enabled=self.enable_autocast):
                prediction = (
                    self.model(
                        batch_value=batch_X_values,
                        batch_timestamp=batch_dict["batch_timestamp"],
                        batch_var_idx=batch_dict["batch_var_idx"],
                        batch_pred_mask=batch_dict["batch_pred_mask"],
                        batch_pad_mask=batch_dict.get("batch_pad_mask", None),
                        batch_time_id=batch_dict.get("batch_time_id", None),
                        batch_agg_mask=batch_dict.get("batch_agg_mask", None),
                    )
                    .squeeze(-1)
                    .detach()
                )
            loss = self.loss_func(
                prediction,
                batch_dict["batch_value"],
                mask=batch_dict["batch_pred_mask"] * batch_dict["batch_pad_mask"],
            )
            tol_loss += loss.item()
            data_num += 1
            y_true.append(batch_dict["batch_value"])
            y_pred.append(prediction)
            y_mask.append(batch_dict["batch_pred_mask"])
            y_var_idx.append(batch_dict["batch_var_idx"])

            if self.rank == 0:
                tqmd_.set_description("eval loss {:.4f}".format(loss.item()))

        if self.enable_distributed:
            total_loss_tensor = torch.tensor(tol_loss).to(self.device)
            dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
            avg_loss = total_loss_tensor.item() / (dist.get_world_size() * data_num)
        else:
            avg_loss = tol_loss / data_num

        y_pred_list = []
        y_true_list = []
        y_mask_list = []
        y_var_idx_list = []
        for i in range(len(y_pred)):
            epoch_y_pred = y_pred[i].cpu().numpy()
            epoch_y_true = y_true[i].cpu().numpy()
            epoch_mask = y_mask[i].cpu().numpy()
            epoch_var_idx = y_var_idx[i].cpu().numpy()
            if self.do_inverse_scaling:
                epoch_y_pred = self.scaler.inverse_transform_mx(
                    epoch_y_pred, var_idx=epoch_var_idx
                )
                epoch_y_true = self.scaler.inverse_transform_mx(
                    epoch_y_true, var_idx=epoch_var_idx
                )
            y_pred_list.append(epoch_y_pred)
            y_true_list.append(epoch_y_true)
            y_mask_list.append(epoch_mask)
            y_var_idx_list.append(epoch_var_idx)
        # create a tensor to store all y_pred and y_true

        if self.enable_distributed:
            raise NotImplementedError("Distributed evaluation is not implemented yet. ")

        else:
            concat_y_pred = np.concatenate(
                [batch.reshape(-1) for batch in y_pred_list], axis=0
            )
            concat_y_true = np.concatenate(
                [batch.reshape(-1) for batch in y_true_list], axis=0
            )
            concat_y_mask = np.concatenate(
                [batch.reshape(-1) for batch in y_mask_list], axis=0
            )
            concat_y_var_idx = np.concatenate(
                [batch.reshape(-1) for batch in y_var_idx_list], axis=0
            )
            eval_results = self._get_eval_result(
                concat_y_pred, concat_y_true, concat_y_mask, concat_y_var_idx, metrics, reduce="mean"
            )
            print("Evaluate result: ", end=" ")
            for metric_name, eval_ret in zip(metrics, eval_results):
                print("{}:  {:.4f}".format(metric_name.upper(), eval_ret), end="  ")
            print()

        return (
            avg_loss,
            list(zip(metrics, eval_results)),
            concat_y_pred if self.rank == 0 else None,
            concat_y_true if self.rank == 0 else None,
        )
