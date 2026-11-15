import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import TestSetLoader, TrainSetLoader
from metrics import PD_FA_sc, ROCMetric, SamplewiseSigmoidMetric, SigmoidMetric
from net import Net
from utils import get_optimizer, seed_pytorch


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="Train SFDANet")
parser.add_argument(
    "--model_names",
    default=["SFDANet"],
    nargs="+",
    choices=["SFDANet"],
)
parser.add_argument(
    "--dataset_names",
    default=["NUAA-SIRST"],
    nargs="+",
)
parser.add_argument(
    "--dataset_dir",
    default="./datasets",
    type=str,
    help="Dataset root relative to the repository root",
)
parser.add_argument("--batchSize", type=int, default=16)
parser.add_argument("--patchSize", type=int, default=256)
parser.add_argument("--lr", default=0.001, type=float)
parser.add_argument("--min_lr", default=1e-5, type=float)
parser.add_argument("--weight_decay", default=0.0, type=float)
parser.add_argument("--nEpochs", type=int, default=1000)
parser.add_argument("--pr", type=int, default=150)
parser.add_argument(
    "--save",
    default="./logs",
    type=str,
    help="Checkpoint and log directory relative to the repository root",
)
parser.add_argument("--resume", default=None, nargs="+")
parser.add_argument("--pretrained", default=None, nargs="+")
parser.add_argument(
    "--optimizer_name",
    default="Adam_FW2",
    choices=["Adam_FW2"],
)
parser.add_argument(
    "--BCE_weight",
    type=float,
    default=0.1,
    help="Weight of the BCE term",
)
parser.add_argument(
    "--softIoU_weight",
    type=float,
    default=1.0,
    help="Weight of the Soft-IoU term",
)
parser.add_argument(
    "--lossc",
    default="BCE_softIoU",
    choices=["BCE_softIoU", "BCE_softiiou"],
)
parser.add_argument("--threads", type=int, default=4)
parser.add_argument("--threshold", type=float, default=0.5)
parser.add_argument("--intervals", type=int, default=1)
parser.add_argument("--val_interval", type=int, default=1)
parser.add_argument("--seed", type=int, default=42)

opt = parser.parse_args()
if opt.lossc == "BCE_softiiou":
    opt.lossc = "BCE_softIoU"


def _strip_module_prefix(state_dict):
    return {
        key[7:] if key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }


def _load_checkpoint(net, checkpoint_path, strict):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    return net.load_state_dict(
        _strip_module_prefix(state_dict),
        strict=strict,
    )


def _evaluation_state_dict(net):
    if isinstance(net, nn.DataParallel):
        return net.module.state_dict()
    return net.state_dict()


def train():
    if opt.nEpochs < 1:
        raise ValueError("--nEpochs must be at least 1")
    if opt.val_interval < 1:
        raise ValueError("--val_interval must be at least 1")
    if opt.BCE_weight < 0 or opt.softIoU_weight < 0:
        raise ValueError("Loss weights must be non-negative")
    if opt.weight_decay < 0:
        raise ValueError("--weight_decay must be non-negative")

    train_set = TrainSetLoader(
        dataset_dir=opt.dataset_dir,
        dataset_name=opt.dataset_name,
        patch_size=opt.patchSize,
    )
    train_loader = DataLoader(
        dataset=train_set,
        num_workers=opt.threads,
        batch_size=opt.batchSize,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )
    test_set = TestSetLoader(
        opt.dataset_dir,
        opt.dataset_name,
        opt.dataset_name,
    )
    test_loader = DataLoader(
        dataset=test_set,
        num_workers=opt.threads,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
    )

    net = Net(
        model_name=opt.model_name,
        mode="train",
        BCE_weight=opt.BCE_weight,
        softIoU_weight=opt.softIoU_weight,
        lossc=opt.lossc,
    )
    epoch_state = 0

    if opt.resume:
        checkpoint_path = opt.resume[0]
        checkpoint = torch.load(checkpoint_path, map_location=device)
        _load_checkpoint(net, checkpoint_path, strict=True)
        epoch_state = int(checkpoint.get("epoch", 0))
        print("Resumed from:", checkpoint_path, "epoch:", epoch_state)
    if opt.pretrained:
        checkpoint_path = opt.pretrained[0]
        incompatible = _load_checkpoint(net, checkpoint_path, strict=False)
        print(
            "Loaded pretrained:",
            checkpoint_path,
            "missing:",
            list(incompatible.missing_keys),
            "unexpected:",
            list(incompatible.unexpected_keys),
        )

    net.train()
    net = net.to(device)
    print("device_count:", torch.cuda.device_count())
    print("batchSize:", opt.batchSize)
    print("nEpochs:", opt.nEpochs)
    print("loss:", opt.lossc)
    print(
        "# model parameters: %.2f M"
        % (sum(parameter.numel() for parameter in net.parameters()) / 1e6)
    )
    opt.f.write(
        "model parameters: %.2f M\n"
        % (sum(parameter.numel() for parameter in net.parameters()) / 1e6)
    )

    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs")
        net = nn.DataParallel(net)

    optimizer, scheduler = get_optimizer(
        net,
        opt.optimizer_name,
        "CosineAnnealingLR",
        {
            "lr": opt.lr,
            "weight_decay": opt.weight_decay,
        },
        {
            "epochs": opt.nEpochs,
            "min_lr": opt.min_lr,
        },
    )
    for _ in range(epoch_state):
        scheduler.step()

    for attr, value in vars(opt).items():
        if attr == "f":
            continue
        print(f"{attr}: {value}")
        opt.f.write(f"{attr}: {value}\n")

    checkpoint_stem = (
        f"{opt.model_name}_{opt.lossc}_"
        f"{opt.BCE_weight}_{opt.softIoU_weight}"
    )
    miou_save_path = os.path.join(
        opt.save,
        opt.dataset_name,
        checkpoint_stem + "_best_miou.pth.tar",
    )
    os.makedirs(os.path.dirname(miou_save_path), exist_ok=True)

    best_miou = 0.0
    if os.path.exists(miou_save_path):
        previous = torch.load(miou_save_path, map_location="cpu")
        best_miou = float(
            previous.get("best_miou", previous.get("current_miou", 0.0))
        )

    iou_metric = SigmoidMetric()
    niou_metric = SamplewiseSigmoidMetric(1, score_thresh=opt.threshold)
    roc = ROCMetric(1, 10)
    total_loss_list = []
    total_loss_epoch = []

    for idx_epoch in range(epoch_state, opt.nEpochs):
        net.train()
        for img, gt_mask in train_loader:
            img = img.to(device, non_blocking=True)
            gt_mask = gt_mask.to(device, non_blocking=True)

            pred = net(img)
            loss = net.loss(pred, gt_mask)
            total_loss_epoch.append(loss.detach().cpu())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()
        if (idx_epoch + 1) % opt.intervals == 0:
            total_loss_list.append(float(np.array(total_loss_epoch).mean()))
            current_lr = optimizer.param_groups[0]["lr"]
            message = (
                time.ctime()[4:-5]
                + " Epoch---%d, total_loss---%f, lr---%f"
                % (idx_epoch + 1, total_loss_list[-1], current_lr)
            )
            print(message)
            opt.f.write(message + "\n")
            total_loss_epoch = []

        epoch_number = idx_epoch + 1
        should_validate = (
            epoch_number > opt.pr
            and (epoch_number - opt.pr) % opt.val_interval == 0
        ) or epoch_number == opt.nEpochs
        if not should_validate:
            continue

        evaluation_state = _evaluation_state_dict(net)
        (
            current_miou,
            current_niou,
            current_pd,
            current_fa,
            current_f1,
        ) = test(
            evaluation_state,
            test_loader,
            iou_metric,
            niou_metric,
            roc,
        )

        if round(current_miou, 4) > best_miou:
            best_miou = round(current_miou, 4)
            message = (
                f"New best mIoU: {current_miou:.4f} "
                f"at epoch {idx_epoch + 1}"
            )
            print(message)
            opt.f.write(message + "\n")
            save_checkpoint(
                {
                    "epoch": idx_epoch + 1,
                    "state_dict": evaluation_state,
                    "total_loss": total_loss_list,
                    "best_miou": best_miou,
                    "current_miou": current_miou,
                    "current_niou": current_niou,
                    "current_pd": current_pd,
                    "current_fa": current_fa,
                    "current_f1": current_f1,
                    "BCE_weight": opt.BCE_weight,
                    "softIoU_weight": opt.softIoU_weight,
                },
                miou_save_path,
            )


def test(ckpt, test_loader, iou_metric, niou_metric, roc):
    net = Net(
        model_name=opt.model_name,
        mode="test",
        BCE_weight=opt.BCE_weight,
        softIoU_weight=opt.softIoU_weight,
        lossc=opt.lossc,
    )
    net.load_state_dict(ckpt)
    net.eval()
    net = net.to(device)

    eval_pd_fa = PD_FA_sc()
    iou_metric.reset()
    niou_metric.reset()
    roc.reset()

    with torch.no_grad():
        for img, gt_mask, size, _ in test_loader:
            img = img.to(device, non_blocking=True)
            pred = net(img)
            if isinstance(pred, (list, tuple)):
                pred = pred[-1]

            height, width = int(size[0]), int(size[1])
            pred = pred[:, :, :height, :width].cpu()
            gt_mask = gt_mask[:, :, :height, :width]

            iou_metric.update(pred, gt_mask)
            niou_metric.update(pred, gt_mask)
            roc.update(pred, gt_mask)
            eval_pd_fa.update(
                pred.sigmoid()[0, 0] > opt.threshold,
                gt_mask[0, 0],
                [height, width],
            )

    _, _, _, _, f1_score = roc.get()
    pix_acc, current_miou = iou_metric.get()
    _, current_niou = niou_metric.get()
    current_pd, current_fa = eval_pd_fa.get()
    message = (
        "pixAcc: %.4f| mIoU: %.4f | nIoU: %.4f | "
        "Pd: %.4f| Fa: %.4f |F1: %.4f"
        % (
            pix_acc * 100,
            current_miou * 100,
            current_niou * 100,
            current_pd * 100,
            current_fa * 1e6,
            f1_score * 100,
        )
    )
    print(message)
    opt.f.write(message + "\n")
    return (
        current_miou,
        current_niou,
        current_pd * 100,
        current_fa * 1e6,
        f1_score * 100,
    )


def save_checkpoint(state, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(state, save_path)
    return save_path


if __name__ == "__main__":
    seed_pytorch(opt.seed)
    for dataset_name in opt.dataset_names:
        opt.dataset_name = dataset_name
        for model_name in opt.model_names:
            opt.model_name = model_name
            os.makedirs(opt.save, exist_ok=True)

            timestamp = (
                time.ctime()
                .replace(" ", "_")
                .replace(":", "_")
            )
            log_path = os.path.join(
                opt.save,
                f"{opt.dataset_name}_{opt.model_name}_{timestamp}.txt",
            )
            with open(log_path, "w", encoding="utf-8") as opt.f:
                train()
                print()
