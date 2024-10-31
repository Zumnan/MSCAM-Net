import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import F1Score, Precision, Recall, Dice, MeanMetric, JaccardIndex

class MedicalSegmentationModel(pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        num_classes: int = 1,  
        init_lr: float = 0.001,
        optimizer_name: str = "Adam",
        weight_decay: float = 1e-4,
        use_scheduler: bool = False,
        scheduler_name: str = "multistep_lr",
        num_epochs: int = 20,
    ):
        super().__init__()

        
        self.save_hyperparameters()

        
        self.model = MSCAMNet(n_channels=3, n_classes=self.hparams.num_classes)
        
        
        self.mean_train_loss = MeanMetric()
        self.mean_train_f1 = F1Score(num_classes=None, threshold=0.5, average="macro", task="binary")
        self.mean_valid_loss = MeanMetric()
        self.mean_valid_f1 = F1Score(num_classes=None, threshold=0.5, average="macro", task="binary")
        self.precision = Precision(num_classes=None, threshold=0.5, average="macro", task="binary")
        self.recall = Recall(num_classes=None, threshold=0.5, average="macro", task="binary")
        
        
        self.dice = Dice(num_classes=None, threshold=0.5, average="micro")
        
        
        self.iou = JaccardIndex(num_classes=1, threshold=0.5, task="binary")

    def compute_iou(self, predicted, target):
        
        predicted = torch.sigmoid(predicted) > 0.5
        target = target.bool()  

        
        intersection = (predicted & target).float().sum(dim=(1, 2, 3))
        union = (predicted | target).float().sum(dim=(1, 2, 3))
        iou = (intersection + 1e-6) / (union + 1e-6)  

        mean_iou = iou.mean()  
        return mean_iou

    def forward(self, data):
        outputs, aux1, aux2, aux3 = self.model(data)
        upsampled_logits = F.interpolate(outputs, size=data.shape[-2:], mode="bilinear", align_corners=False)
        aux1 = F.interpolate(aux1, size=data.shape[-2:], mode="bilinear", align_corners=False)
        aux2 = F.interpolate(aux2, size=data.shape[-2:], mode="bilinear", align_corners=False)
        aux3 = F.interpolate(aux3, size=data.shape[-2:], mode="bilinear", align_corners=False)
        return upsampled_logits, aux1, aux2, aux3

    def training_step(self, batch, batch_idx):
        data, target = batch
        logits, aux1, aux2, aux3 = self(data)  

        
        
        logits = logits.squeeze(1)  

        
        # Calculate loss with auxiliary outputs
        loss = binary_segmentation_loss(logits, target, aux1=aux1, aux2=aux2, aux3=aux3)


        self.mean_train_loss.update(loss, weight=data.shape[0])
        self.mean_train_f1.update(logits.detach(), target)
        precision = self.precision(logits.detach(), target)
        recall = self.recall(logits.detach(), target)
        iou = self.iou(logits.detach(), target)  
        dice = self.dice(logits.detach(), target)

        self.log("train/batch_loss", self.mean_train_loss, prog_bar=True, logger=False)
        self.log("train/batch_f1", self.mean_train_f1, prog_bar=True, logger=False)
        self.log("train/precision", precision, prog_bar=True, logger=False)
        self.log("train/recall", recall, prog_bar=True, logger=False)
        self.log("train/iou", iou, prog_bar=True, logger=False)
        self.log("train/dice", dice, prog_bar=True, logger=False)

        return loss

    def on_train_epoch_end(self):
        
        self.log("train/loss", self.mean_train_loss, prog_bar=True)
        self.log("train/f1", self.mean_train_f1, prog_bar=True)
        self.log("epoch", self.current_epoch)

    def validation_step(self, batch, batch_idx):
        data, target = batch
        logits, aux1, aux2, aux3 = self(data)  

        
        
        logits = logits.squeeze(1)  

        
        # Calculate loss with auxiliary outputs
        loss = binary_segmentation_loss(logits, target, aux1=aux1, aux2=aux2, aux3=aux3)


        
        self.mean_valid_loss.update(loss, weight=data.shape[0])
        self.mean_valid_f1.update(logits, target)
        precision = self.precision(logits.detach(), target)
        recall = self.recall(logits.detach(), target)
        iou = self.iou(logits.detach(), target)  
        dice = self.dice(logits.detach(), target)

        
        self.log("valid/batch_loss", self.mean_valid_loss, prog_bar=True, logger=True)
        self.log("valid/batch_f1", self.mean_valid_f1, prog_bar=True, logger=True)
        self.log("valid/precision", precision, prog_bar=True, logger=True)
        self.log("valid/recall", recall, prog_bar=True, logger=True)
        self.log("valid/iou", iou, prog_bar=True, logger=True)
        self.log("valid/dice", dice, prog_bar=True, logger=True)

        return loss

    def on_validation_epoch_end(self):
        
        self.log("valid/loss", self.mean_valid_loss, prog_bar=True)
        self.log("valid/f1", self.mean_valid_f1, prog_bar=True)
        self.log("epoch", self.current_epoch)

    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.hparams.optimizer_name)(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.hparams.init_lr,
            weight_decay=self.hparams.weight_decay,
        )

        if self.hparams.use_scheduler:
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[self.trainer.max_epochs // 2,], gamma=0.1)

            lr_scheduler_config = {"scheduler": lr_scheduler, "interval": "epoch", "name": "multi_step_lr"}
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}

        return optimizer
