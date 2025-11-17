import warnings

import tqdm
from tqdm import tqdm
import torch
from trainers.abs import AbstractTrainer
from trainers.IrrMaskingForecastTrainer import IrrMaskingForecastTrainer
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast
import utils.metrics as metrics_module


class IrrMaskingForecastCustomLossTrainer(IrrMaskingForecastTrainer):
    """
    A Trainer subclass for Irr data forecasting. use model inside custom loss function
    """

    def train_one_epoch(self, data_loader):
        self.model.train()
        total_loss = 0
        if self.rank == 0:
            tqmd_ = tqdm(data_loader)
        else:
            tqmd_ = data_loader
        for batch_dict in tqmd_:
            # move_batch_data_to_device
            for key in batch_dict:
                batch_dict[key] = batch_dict[key].type(torch.float32).to(self.device)
            with autocast(enabled=self.enable_autocast):
                output = self.model(
                    batch_dict["observed_data"],
                    batch_dict["observed_tp"],
                    batch_dict["observed_mask"],
                    batch_dict["target_forecast_data"],
                    batch_dict["target_forecast_tp"],
                    batch_dict["target_forecast_mask"],
                    test_flag=False,
                )
                prediction, loss = output["pred"], output["loss"]

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
        self, y_pred, y_true, mask, metrics=("mae", "rmse", "mape"), reduce="mean"
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

            result = eval_func(y_true, y_pred, mask, reduce=reduce)
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
        y_true, y_pred, y_mask, tol_loss, data_num = [], [], [], 0, 0
        test_flag = kwargs.get("test_flag", False)
        if self.rank == 0:
            tqmd_ = tqdm(data_loader)
        else:
            tqmd_ = data_loader
        for batch_dict in tqmd_:
            # move_batch_data_to_device
            for key in batch_dict:
                batch_dict[key] = batch_dict[key].type(torch.float32).to(self.device)
            with autocast(enabled=self.enable_autocast):
                output = self.model(
                    batch_dict["observed_data"],
                    batch_dict["observed_tp"],
                    batch_dict["observed_mask"],
                    batch_dict["target_forecast_data"],
                    batch_dict["target_forecast_tp"],
                    batch_dict["target_forecast_mask"],
                    test_flag=test_flag,
                )
                prediction, loss = output["pred"], output["loss"]
            tol_loss += loss.item()
            data_num += 1
            y_true.append(batch_dict["target_forecast_data"])
            y_pred.append(prediction)
            y_mask.append(batch_dict["target_forecast_mask"])

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
        for i in range(len(y_pred)):
            epoch_y_pred = y_pred[i].cpu().numpy()
            epoch_y_true = y_true[i].cpu().numpy()
            epoch_mask = y_mask[i].cpu().numpy()
            if self.do_inverse_scaling:
                epoch_y_pred = self.scaler.inverse_transform_mx(epoch_y_pred)
                epoch_y_true = self.scaler.inverse_transform_mx(epoch_y_true)
            y_pred_list.append(epoch_y_pred)
            y_true_list.append(epoch_y_true)
            y_mask_list.append(epoch_mask)
        # concat y_pred_list and y_true_list
        iter_num, y_pred_tp_max, y_true_tp_max = 0, 0, 0
        for i in range(len(y_pred_list)):
            iter_num += y_pred_list[i].shape[0]
            y_pred_tp_max = max(y_pred_tp_max, y_pred_list[i].shape[1])
            y_true_tp_max = max(y_true_tp_max, y_true_list[i].shape[1])
        # geather iter_num and y_pred_tp_max, y_true_tp_max
        if self.enable_distributed:
            iter_num_tensor = torch.tensor(iter_num, device=self.device)
            y_pred_tp_max_tensor = torch.tensor(y_pred_tp_max, device=self.device)
            y_true_tp_max_tensor = torch.tensor(y_true_tp_max, device=self.device)
            dist.all_reduce(iter_num_tensor, op=dist.ReduceOp.MAX)
            dist.all_reduce(y_pred_tp_max_tensor, op=dist.ReduceOp.MAX)
            dist.all_reduce(y_true_tp_max_tensor, op=dist.ReduceOp.MAX)
            iter_num = iter_num_tensor.item()
            y_pred_tp_max = y_pred_tp_max_tensor.item()
            y_true_tp_max = y_true_tp_max_tensor.item()
        # create a tensor to store all y_pred and y_true
        concat_y_pred_tensor = torch.zeros(
            iter_num, y_pred_tp_max, y_pred_list[0].shape[2]
        )
        concat_y_true_tensor = torch.zeros(
            iter_num, y_true_tp_max, y_pred_list[0].shape[2]
        )
        concat_y_mask_tensor = torch.zeros(
            iter_num, y_true_tp_max, y_pred_list[0].shape[2]
        )
        start_idx = 0
        for i in range(len(y_pred_list)):
            end_idx = start_idx + y_pred_list[i].shape[0]
            concat_y_pred_tensor[
                start_idx:end_idx, : y_pred_list[i].shape[1]
            ] = torch.tensor(y_pred_list[i])
            concat_y_true_tensor[
                start_idx:end_idx, : y_true_list[i].shape[1]
            ] = torch.tensor(y_true_list[i])
            concat_y_mask_tensor[
                start_idx:end_idx, : y_true_list[i].shape[1]
            ] = torch.tensor(y_mask_list[i])
            start_idx = end_idx

        if self.enable_distributed:
            if self.rank == 0:
                gathered_concat_y_pred_tensor = [
                    torch.zeros_like(concat_y_pred_tensor)
                    for _ in range(dist.get_world_size())
                ]
                gathered_concat_y_true_tensor = [
                    torch.zeros_like(concat_y_true_tensor)
                    for _ in range(dist.get_world_size())
                ]
                gathered_concat_y_mask_tensor = [
                    torch.zeros_like(concat_y_mask_tensor)
                    for _ in range(dist.get_world_size())
                ]
                dist.gather(
                    concat_y_pred_tensor,
                    gather_list=gathered_concat_y_pred_tensor,
                    dst=0,
                )
                dist.gather(
                    concat_y_true_tensor,
                    gather_list=gathered_concat_y_true_tensor,
                    dst=0,
                )
                dist.gather(
                    concat_y_mask_tensor,
                    gather_list=gathered_concat_y_mask_tensor,
                    dst=0,
                )
                concat_y_pred = (
                    torch.cat(gathered_concat_y_pred_tensor, dim=0).cpu().numpy()
                )
                concat_y_true = (
                    torch.cat(gathered_concat_y_true_tensor, dim=0).cpu().numpy()
                )
                concat_y_mask = (
                    torch.cat(gathered_concat_y_mask_tensor, dim=0).cpu().numpy()
                )
                eval_results = self._get_eval_result(
                    concat_y_pred, concat_y_true, concat_y_mask, metrics, reduce="mean"
                )
                print("Evaluate result: ", end=" ")
                for metric_name, eval_ret in zip(metrics, eval_results):
                    print("{}:  {:.4f}".format(metric_name.upper(), eval_ret), end="  ")
                print()
            else:
                concat_y_pred, concat_y_true, concat_y_mask = None, None, None
                dist.gather(concat_y_pred_tensor, dst=0)
                dist.gather(concat_y_true_tensor, dst=0)
                dist.gather(concat_y_mask_tensor, dst=0)
                eval_results = [0 for _ in metrics]

        else:
            concat_y_pred = concat_y_pred_tensor.cpu().numpy()
            concat_y_true = concat_y_true_tensor.cpu().numpy()
            concat_y_mask = concat_y_mask_tensor.cpu().numpy()
            eval_results = self._get_eval_result(
                concat_y_pred, concat_y_true, concat_y_mask, metrics, reduce="mean"
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
