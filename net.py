from torch import nn

from loss import CombinedBCEIoULoss
from model.SFDANet import SFDANet


class Net(nn.Module):
    def __init__(
        self,
        model_name,
        mode,
        BCE_weight,
        softIoU_weight,
        lossc,
    ):
        super().__init__()
        if model_name != "SFDANet":
            raise ValueError("This release only provides the SFDANet model")
        if lossc != "BCE_softIoU":
            raise ValueError("This release only provides the BCE_softIoU loss")
        if mode not in ("train", "test"):
            raise ValueError("mode must be 'train' or 'test'")

        self.model_name = model_name
        self.model = SFDANet(1)
        self.cal_loss = CombinedBCEIoULoss(
            bce_weight=BCE_weight,
            soft_iou_weight=softIoU_weight,
        )

    def forward(self, img):
        return self.model(img)

    def loss(self, pred, gt_mask):
        return self.cal_loss(pred, gt_mask)
