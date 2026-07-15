from typing import Any, Dict, Tuple

import torch
from pytorch_lightning import LightningModule
from torchmetrics import AveragePrecision, MaxMetric, MeanMetric


class TaggingModule(LightningModule):
    """LightningModule for multi-label audio tagging."""

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        loss: str,
        opt_warmup: bool,
        learning_rate: float,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.net = net
        self.warmup = opt_warmup

        if loss == 'bce':
            self.criterion = torch.nn.BCELoss()
        elif loss == 'cross_entropy':
            self.criterion = torch.nn.CrossEntropyLoss()
        elif loss == 'bcelogit':
            self.criterion = torch.nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unknown loss: {loss}")

        num_labels = net.num_class
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.val_mAP_best = MaxMetric()
        # torchmetrics handles cross-process synchronisation under DDP
        self.ap = AveragePrecision(task="multilabel", num_labels=num_labels, average=None)
        self.ap_test = AveragePrecision(task="multilabel", num_labels=num_labels, average=None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def on_train_start(self) -> None:
        # validation sanity checks run before training starts,
        # so we reset the metrics to avoid polluting them
        self.val_loss.reset()
        self.ap.reset()
        self.val_mAP_best.reset()

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y = batch
        preds = self.forward(x)
        loss = self.criterion(preds, y)
        return loss, preds, y

    def on_train_batch_start(self, batch: Any, batch_idx: int) -> None:
        # linear learning-rate warmup over the first 1000 steps
        if not self.warmup:
            return
        global_step = self.trainer.global_step
        optimizer = self.optimizers()
        if global_step <= 1000 and global_step % 50 == 0:
            warm_lr = (global_step / 1000) * self.hparams.optimizer.keywords['lr']
            for param_group in optimizer.param_groups:
                param_group['lr'] = warm_lr
            self.log('lr', warm_lr, on_step=True, on_epoch=False, logger=True)
        current_lr = next(iter(optimizer.param_groups))['lr']
        self.log('cur-lr', current_lr, on_step=False, on_epoch=True, logger=True)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss, preds, y = self.model_step(batch)
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        loss, preds, targets = self.model_step(batch)
        self.ap.update(preds, targets.long())
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def on_validation_epoch_end(self) -> None:
        mAP = self.ap.compute().mean()
        self.val_mAP_best(mAP)
        self.log("val/mAP", mAP, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/mAP_best", self.val_mAP_best.compute(), on_step=False, on_epoch=True, sync_dist=True)
        self.ap.reset()

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        loss, preds, targets = self.model_step(batch)
        self.ap_test.update(preds, targets.long())
        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def on_test_epoch_end(self) -> None:
        mAP = self.ap_test.compute().mean()
        self.log("test/mAP", mAP, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.ap_test.reset()

    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
