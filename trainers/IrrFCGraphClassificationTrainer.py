# trainers/irr_cls_trainer.py
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
import torch.distributed as dist

from trainers.abs import AbstractTrainer
import utils.metrics as metrics_module


class IrrFCGraphClassificationTrainer(AbstractTrainer):
    """
    A Trainer subclass for irregular multivariate time-series CLASSIFICATION.
    - Loss: CrossEntropyLoss
    - Default metrics: accuracy / auroc / auprc
    - Returns & saves probabilities (softmax), not raw predictions
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
        do_inverse_scaling=False,       
        global_args=None,
        *args,
        **kwargs,
    ):
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
        self.criterion = torch.nn.CrossEntropyLoss()
        self.num_nodes = kwargs.get("num_nodes", None)

    # ------------ loss ------------
    def loss_func(self, logits, labels, *args, **kwargs):
        """
        Cross-Entropy over logits of shape (B, C) and labels of shape (B,)
        """
        return self.criterion(logits, labels)

    # ------------ train one epoch ------------
    def train_one_epoch(self, data_loader):
        self.model.train()
        total_loss = 0.0
        iterator = tqdm(data_loader) if self.rank == 0 else data_loader

        for batch_dict in iterator:
            # move to device
            for key, val in batch_dict.items():
                if isinstance(val, torch.Tensor):
                    batch_dict[key] = val.to(self.device, non_blocking=True)

            with autocast(enabled=self.enable_autocast):
                logits = self.model(
                    batch_value=batch_dict.get("batch_value", None),
                    batch_timestamp=batch_dict.get("batch_timestamp", None),
                    batch_var_idx=batch_dict.get("batch_var_idx", None),
                    batch_pad_mask=batch_dict.get("batch_pad_mask", None),
                    batch_agg_mask=batch_dict.get("batch_agg_mask", None),
                    batch_static=batch_dict.get("batch_static", None),
                )  # -> (B, C)
                loss = self.loss_func(logits, batch_dict["batch_label"])

            self.optimizer.zero_grad()
            if self.enable_autocast:
                self.grad_scaler.scale(loss).backward()
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            if self.rank == 0:
                iterator.set_description(f"loss {loss.item():.4f}")
            total_loss += loss.item()

        # world average loss
        if self.enable_distributed:
            total_loss_tensor = torch.tensor(total_loss, device=self.device)
            dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
            avg_loss = total_loss_tensor.item() / (dist.get_world_size() * len(data_loader))
        else:
            avg_loss = total_loss / len(data_loader)
        return avg_loss

    def _get_eval_result(
        self,y_true, y_pred, metrics=("accuracy", "auroc", "auprc")
    ):
        eval_results = []
        for metric_name in metrics:
            eval_func_name = "get_{}".format(metric_name)
            eval_func = getattr(metrics_module, eval_func_name, None)

            if eval_func is None:
                raise AttributeError(
                    f"Function '{eval_func_name}' not found in 'utils.metrics'."
                )

            result = eval_func(y_true, y_pred)
            eval_results.append(result)

        return eval_results

    @torch.no_grad()
    def evaluate(self, data_loader, metrics=("accuracy", "auroc", "auprc"), **kwargs):
        """
        Evaluate classification model.
        Returns:
          avg_loss, list[(metric, value)], y_prob (for saving), y_true
        """
        self.model.eval()
        iterator = tqdm(data_loader) if self.rank == 0 else data_loader
#
        total_loss = 0.0
        n_steps = 0

        all_probs = []
        all_labels = []

        for batch_dict in iterator:
            # move to device
            for key, val in batch_dict.items():
                if isinstance(val, torch.Tensor):
                    batch_dict[key] = val.to(self.device, non_blocking=True)

            with autocast(enabled=self.enable_autocast):
                logits = self.model(
                    batch_value=batch_dict.get("batch_value", None),
                    batch_timestamp=batch_dict.get("batch_timestamp", None),
                    batch_var_idx=batch_dict.get("batch_var_idx", None),
                    batch_pad_mask=batch_dict.get("batch_pad_mask", None),
                    batch_agg_mask=batch_dict.get("batch_agg_mask", None),
                    batch_static=batch_dict.get("batch_static", None),
                )  # (B, C) or (B,) / (B,1) for edge cases

                loss = self.loss_func(logits, batch_dict["batch_label"])

            total_loss += loss.item()
            n_steps += 1

            if logits.dim() == 1 or logits.shape[-1] == 1:

                p1 = torch.sigmoid(logits.view(-1))
                probs = torch.stack([1.0 - p1, p1], dim=-1)
            else:
                probs = F.softmax(logits, dim=-1)

            all_probs.append(probs.detach().cpu())
            all_labels.append(batch_dict["batch_label"].detach().cpu())

            if self.rank == 0:
                iterator.set_description(f"eval loss {loss.item():.4f}")

        # average loss
        if self.enable_distributed:
            total_loss_tensor = torch.tensor(total_loss, device=self.device)
            dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
            avg_loss = total_loss_tensor.item() / (dist.get_world_size() * n_steps)
        else:
            avg_loss = total_loss / max(n_steps, 1)

        # gather to numpy (non-distributed version)
        if self.enable_distributed:
            raise NotImplementedError("Distributed evaluation is not implemented yet for classification.")
        y_prob = torch.cat(all_probs, dim=0).numpy()   # (N, C)
        y_true = torch.cat(all_labels, dim=0).numpy()  # (N,)

        # compute metrics via utils.metrics
        eval_results = self._get_eval_result(y_true, y_prob)

        if self.rank == 0:
            print("Evaluate result: ", end="")
            for name, val in zip(metrics, eval_results):
                print(f"{name.upper()}: {val:.4f}  ", end="")
            print()


        return (
            avg_loss,
            list(zip(metrics, eval_results)),
            y_prob if self.rank == 0 else None,
            y_true if self.rank == 0 else None,
        )
