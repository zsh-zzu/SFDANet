import torch
import torch.nn as nn


class SoftIoULoss(nn.Module):
    def forward(self, preds, gt_masks):
        if isinstance(preds, (list, tuple)):
            losses = []
            for pred in preds:
                pred_prob = torch.sigmoid(pred)
                intersection = pred_prob * gt_masks
                loss = (
                    intersection.sum() + 1
                ) / (
                    pred_prob.sum()
                    + gt_masks.sum()
                    - intersection.sum()
                    + 1
                )
                losses.append(1 - loss.mean())
            return torch.stack(losses).mean()

        pred_prob = torch.sigmoid(preds)
        intersection = pred_prob * gt_masks
        loss = (
            intersection.sum() + 1
        ) / (
            pred_prob.sum()
            + gt_masks.sum()
            - intersection.sum()
            + 1
        )
        return 1 - loss.mean()


class BCE_SC_sigmoid(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()

    def forward(self, preds, gt_masks):
        if isinstance(preds, (list, tuple)):
            losses = [
                self.bce(torch.sigmoid(pred), gt_masks)
                for pred in preds
            ]
            return torch.stack(losses).mean()
        return self.bce(torch.sigmoid(preds), gt_masks)


class CombinedBCEIoULoss(nn.Module):
    def __init__(self, bce_weight=0.1, soft_iou_weight=1.0):
        super().__init__()
        if bce_weight < 0 or soft_iou_weight < 0:
            raise ValueError("Loss weights must be non-negative")
        self.bce_loss = BCE_SC_sigmoid()
        self.iou_loss = SoftIoULoss()
        self.bce_weight = bce_weight
        self.soft_iou_weight = soft_iou_weight

    def forward(self, preds, gt_masks):
        bce = self.bce_loss(preds, gt_masks)
        soft_iou = self.iou_loss(preds, gt_masks)
        return (
            self.bce_weight * bce
            + self.soft_iou_weight * soft_iou
        )
