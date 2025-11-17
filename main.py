import json
import os
import sys
import random
import yaml
import argparse
import numpy as np
import torch
import trainers
import models
import datasets
import data_processors
import warnings
from utils import scaler
from utils.json_serializable import json_serializable
from utils.balanced_batch_sampler import BalancedBatchSampler
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

#os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
#torch.backends.cudnn.deterministic = True
#torch.use_deterministic_algorithms(True)

def setup_distributed():
    try:  # try nccl else gloo
        dist.init_process_group("nccl")
    except:
        dist.init_process_group("gloo")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def cleanup_distributed():
    dist.destroy_process_group()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def load_config(data_path):
    with open(data_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def main(args):
    rank = int(os.environ["LOCAL_RANK"]) if args.distributed else 0
    world_size = int(os.environ["WORLD_SIZE"]) if args.distributed else 1
    result_save_dir_path = args.result_save_dir_path
    model_save_dir_path = args.model_save_dir_path
    if not os.path.exists(result_save_dir_path):
        os.makedirs(result_save_dir_path)
    if not os.path.exists(model_save_dir_path):
        os.makedirs(model_save_dir_path)

    config_all = load_config(args.config_path)

    global_args = config_all.get("global_config", {})
    train_config = config_all["train_config"]
    model_config = config_all["model_config"]
    data_config = config_all["data_config"]
    task_name = global_args.get("task_name")

    # basic config
    batch_size = train_config["batch_size"]
    use_balanced_sampler = train_config.get(
        "use_balanced_sampler", False
    )  # for cls task
    load_checkpoint = train_config["load_checkpoint"]
    metrics = train_config.get("metrics", ["mae", "rmse", "mape"])
    device = train_config.get("device", "cuda")
    global_args["device"] = device

    # set random seeds for reproducibility
    set_seed(train_config["random_seed"])

    # ----------------------- Load data ------------------------
    data_processor_class = getattr(
        sys.modules["data_processors"], data_config["data_processor_name"]
    )
    data_processor = data_processor_class(
        task_name=global_args["task_name"], global_args=global_args
    )
    processed_data = data_processor.process()

    train_data, valid_data, test_data = (
        {
            "train_data": processed_data["train_data"],
            "train_static_data": processed_data.get("train_static_data", None),
            "train_label": processed_data.get("train_label", None),
            "train_ts_ids": processed_data.get("train_ts_ids", None),
        },
        {
            "valid_data": processed_data["valid_data"],
            "valid_static_data": processed_data.get("valid_static_data", None),
            "valid_label": processed_data.get("valid_label", None),
            "valid_ts_ids": processed_data.get("valid_ts_ids", None),
        },
        {
            "test_data": processed_data["test_data"],
            "test_static_data": processed_data.get("test_static_data", None),
            "test_label": processed_data.get("test_label", None),
            "test_ts_ids": processed_data.get("test_ts_ids", None),
        },
    )

    # scale data
    scaler_name = data_config.get("scaler_name", None)
    if scaler_name is None:
        scaler = None
        warnings.warn(
            "scaler_name is not specified, no scaling will be applied to the data"
        )
    else:
        scaler_class = getattr(sys.modules["utils.scaler"], data_config["scaler_name"])
        scaler = scaler_class(**data_config["scaler_params"])
        if task_name == "prediction":
            seen_data = processed_data["seen_data"]
        else:
            seen_data = train_data
        scaler.fit(seen_data)
        # save scaler
        scaler.save(os.path.join(model_save_dir_path, "scaler"))

        train_data = scaler.transform(train_data)
        valid_data = scaler.transform(valid_data)
        test_data = scaler.transform(test_data)

    # dataset
    dataset_class = getattr(sys.modules["datasets"], data_config["dataset_name"])
    train_dataset = dataset_class(
        train_data,
        type="train",
        **data_config["dataset_params"],
        global_args=global_args,
    )
    valid_dataset = dataset_class(
        valid_data,
        type="valid",
        **data_config["dataset_params"],
        global_args=global_args,
    )
    test_dataset = dataset_class(
        test_data, type="test", **data_config["dataset_params"], global_args=global_args
    )

    if args.distributed:
        train_sampler = DistributedSampler(train_dataset)
        valid_sampler = DistributedSampler(valid_dataset, shuffle=False)
        test_sampler = DistributedSampler(test_dataset, shuffle=False)
    else:
        train_sampler, valid_sampler, test_sampler = None, None, None

    if use_balanced_sampler:
        train_labels = np.asarray(getattr(train_dataset, "_labels"), dtype=np.int64)

        repeat_minority = train_config.get("repeat_minority", 3)
        drop_last = data_config["dataloader_params"].get("drop_last", True)

        balanced_batch_sampler = BalancedBatchSampler(
            labels=train_labels,
            batch_size=batch_size,
            repeat_minority=repeat_minority,
            drop_last=drop_last,
        )

        _dl_params = {
            k: v
            for k, v in data_config["dataloader_params"].items()
            if k not in ("batch_size", "shuffle", "drop_last")
        }
        train_dataloader = DataLoader(
            train_dataset,
            batch_sampler=balanced_batch_sampler,
            collate_fn=train_dataset.collate_fn
            if hasattr(train_dataset, "collate_fn")
            else None,
            **_dl_params,
        )
    else:
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            collate_fn=train_dataset.collate_fn
            if hasattr(train_dataset, "collate_fn")
            else None,
            **data_config["dataloader_params"],
        )

    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        sampler=valid_sampler,
        collate_fn=valid_dataset.collate_fn
        if hasattr(valid_dataset, "collate_fn")
        else None,
        **data_config["dataloader_params"],
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        sampler=test_sampler,
        collate_fn=test_dataset.collate_fn
        if hasattr(test_dataset, "collate_fn")
        else None,
        **data_config["dataloader_params"],
    )

    # ------------------------- Model ---------------------------

    # model
    model_class = getattr(sys.modules["models"], model_config["model_name"])
    if args.distributed:
        model = model_class(**model_config["model_params"], global_args=global_args).to(
            torch.device("cuda", rank)
        )
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    else:
        model = model_class(**model_config["model_params"], global_args=global_args).to(
            device
        )

    # ------------------------- Trainer -------------------------
    # only train the parameters that require grad
    trained_parameters = []
    for p in model.parameters():
        if p.requires_grad is True:
            trained_parameters.append(p)

    # Optimizer
    optimizer_class = getattr(
        sys.modules["torch.optim"], train_config["optimizer_name"]
    )
    optimizer = optimizer_class(trained_parameters, **train_config["optimizer_params"])

    # scheduler
    scheduler_class = getattr(
        sys.modules["torch.optim.lr_scheduler"], train_config["scheduler_name"]
    )
    if train_config["scheduler_name"] == "OneCycleLR":
        # pct_start and max_lr are required for OneCycleLR
        assert "pct_start" in train_config["scheduler_params"]
        assert "max_lr" in train_config["scheduler_params"]

        scheduler = scheduler_class(
            optimizer=optimizer,
            steps_per_epoch=len(train_dataloader),
            epochs=train_config["trainer_params"]["max_epoch_num"],
            **train_config["scheduler_params"],
        )
    else:
        scheduler = scheduler_class(optimizer, **train_config["scheduler_params"])

    # trainer
    trainer_class = getattr(sys.modules["trainers"], train_config["trainer_name"])

    trainer = trainer_class(
        model,
        optimizer,
        scheduler,
        scaler,
        model_save_dir_path,
        result_save_dir_path,
        enable_distributed=args.distributed,
        global_args=global_args,
        **train_config["trainer_params"],
    )

    # load checkpoint
    if load_checkpoint:
        trainer.load_checkpoint()

    # ------------------------- Train & Test ------------------------
    if rank == 0:
        config = {
            "args": vars(args),
            "config": config_all,
            "global_args": global_args,
        }
        print("Configuration: ", config)
        config = json_serializable(config)
        with open(os.path.join(model_save_dir_path, "config.json"), "w") as f:
            json.dump(config, f, indent=4)

        print("Start training.")

    best_eval_result, epoch_results = trainer.train(
        train_dataloader, valid_dataloader, metrics=metrics
    )
    # wait for all processes to finish training
    if args.distributed:
        dist.barrier()

    test_result, y_pred, y_true = trainer.test(test_dataloader, metrics=metrics)

    result = {}
    if rank == 0:
        # save y_pred, y_true to self.result_save_dir/y_pred.npy, y_true.npy
        np.save(os.path.join(result_save_dir_path, "test_y_pred.npy"), y_pred)
        np.save(os.path.join(result_save_dir_path, "test_y_true.npy"), y_true)

        # save results
        result = {
            "config": config,
            "test_result": test_result,
            "best_eval_result": best_eval_result,
            "epoch_results": epoch_results,
        }
        result = json_serializable(result)
        with open(os.path.join(result_save_dir_path, "result.json"), "w") as f:
            json.dump(result, f, indent=4)

        print("Training finished.")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        default="./config/new_arch/STRGCN/FINALCONF/STRGCN_Physionet2012.yaml",
        help="Config path of Trainer",
    )
    parser.add_argument(
        "--model_save_dir_path",
        type=str,
        default="./model_states/new_arch_STRGCN_Physionet_config1",
        help="Model save path",
    )

    parser.add_argument(
        "--result_save_dir_path",
        type=str,
        default="./results/new_arch_STRGCN_Physionet_example1",
        help="Result save path",
    )

    parser.add_argument(
        "--distributed", action="store_true", help="Enable distributed training"
    )

    args = parser.parse_args()
    if args.distributed:
        setup_distributed()
    main(args)
