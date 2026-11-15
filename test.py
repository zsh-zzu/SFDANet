import argparse
import os

import torch
from torch.utils.data import DataLoader, Subset

from dataset import TestSetLoader, TestSetLoaderRe256
from metrics import PD_FA_sc, ROCMetric, SamplewiseSigmoidMetric, SigmoidMetric
from net import Net


def _strip_module_prefix(state_dict):
    return {
        key[7:] if key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }


def _build_jobs(model_names, dataset_names, checkpoint_paths):
    if len(model_names) == len(dataset_names) == len(checkpoint_paths):
        return list(zip(model_names, dataset_names, checkpoint_paths))
    if len(model_names) == 1 and len(checkpoint_paths) == len(dataset_names):
        return [
            (model_names[0], dataset_name, checkpoint_path)
            for dataset_name, checkpoint_path
            in zip(dataset_names, checkpoint_paths)
        ]
    raise ValueError(
        "Provide one checkpoint per dataset, or equal-length "
        "model/dataset/checkpoint lists."
    )


def evaluate(model_name, dataset_name, checkpoint_path):
    if args.test_size > 0:
        test_set = TestSetLoaderRe256(
            args.dataset_dir,
            dataset_name,
            dataset_name,
            base_size=args.test_size,
        )
    else:
        test_set = TestSetLoader(
            args.dataset_dir,
            dataset_name,
            dataset_name,
        )
    if args.max_images > 0:
        test_set = Subset(
            test_set,
            range(min(args.max_images, len(test_set))),
        )

    test_loader = DataLoader(
        dataset=test_set,
        num_workers=args.threads,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
    )

    net = Net(
        model_name=model_name,
        mode="test",
        BCE_weight=0.1,
        softIoU_weight=1.0,
        lossc="BCE_softIoU",
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    net.load_state_dict(_strip_module_prefix(state_dict))
    net.eval()
    net = net.to(device)

    iou_metric = SigmoidMetric()
    niou_metric = SamplewiseSigmoidMetric(1, score_thresh=args.threshold)
    roc = ROCMetric(1, 10)
    pd_fa = PD_FA_sc()

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
            pd_fa.update(
                pred.sigmoid()[0, 0] > args.threshold,
                gt_mask[0, 0],
                [height, width],
            )

    pix_acc, miou = iou_metric.get()
    _, niou = niou_metric.get()
    pd, fa = pd_fa.get()
    _, _, _, _, f1_score = roc.get()
    message = (
        "%s | pixAcc: %.4f | mIoU: %.4f | nIoU: %.4f | "
        "Pd: %.4f | Fa: %.4f | F1: %.4f"
        % (
            dataset_name,
            pix_acc * 100,
            miou * 100,
            niou * 100,
            pd * 100,
            fa * 1e6,
            f1_score * 100,
        )
    )
    print(message)

    result_dir = os.path.join(args.save_log, dataset_name)
    os.makedirs(result_dir, exist_ok=True)
    result_path = os.path.join(result_dir, model_name + ".txt")
    with open(result_path, "w", encoding="utf-8") as result_file:
        result_file.write(os.path.abspath(checkpoint_path) + "\n")
        result_file.write(message + "\n")


parser = argparse.ArgumentParser(description="Evaluate SFDANet")
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
    "--pth_dirs",
    required=True,
    nargs="+",
    help="One checkpoint path per dataset",
)
parser.add_argument("--dataset_dir", default="./datasets", type=str)
parser.add_argument(
    "--test_size",
    default=0,
    type=int,
    help="Resize to this square size; 0 keeps the original resolution",
)
parser.add_argument("--threshold", default=0.5, type=float)
parser.add_argument("--threads", default=0, type=int)
parser.add_argument("--max_images", default=0, type=int)
parser.add_argument("--save_log", default="./results", type=str)

args = parser.parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    for job in _build_jobs(
        args.model_names,
        args.dataset_names,
        args.pth_dirs,
    ):
        evaluate(*job)
