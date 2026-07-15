import os
from typing import Any, Dict, List, Optional, Tuple

import hydra
import pytorch_lightning as pl
import rootutils
import torch
from omegaconf import DictConfig
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import Logger

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=False)

from src.utils import (
    RankedLogger,
    extras,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)
from src.utils.utils import load_weights

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Train the model and optionally evaluate it on the eval set.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    pretrained = cfg.get('pretrained')
    if pretrained in ['audioset', 'img']:
        load_weights(model, cfg, pretrained)
        log.info(f"Loaded '{pretrained}' pretrained weights from {cfg.get('pretrain_path')}")
    else:
        log.info("Training from scratch")

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "callbacks": callbacks,
        "trainer": trainer,
    }
    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))
        log.info("Training completed!")

    # Evaluate the best checkpoint on the eval set
    if cfg.get("eval"):
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best checkpoint not found! Using current weights for testing...")
            ckpt_path = None
        log.info(f"Evaluating best checkpoint: {ckpt_path}")
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

    # Evaluate a weighted average of all saved checkpoints (checkpoint averaging)
    if cfg.get("wa"):
        log.info("Evaluating with weighted average model")
        ckpt_dir = cfg.callbacks.get('model_checkpoint').get('dirpath')
        model_ckpt = []
        for ckpt in os.listdir(ckpt_dir):
            if ckpt.endswith('.ckpt'):
                model_ckpt.append(torch.load(os.path.join(ckpt_dir, ckpt), map_location='cpu')['state_dict'])

        model: LightningModule = hydra.utils.instantiate(cfg.model)
        own_state = model.state_dict()
        for name, params in own_state.items():
            stacked = torch.stack([d[name].float() for d in model_ckpt], dim=0)
            own_state[name] = stacked.mean(dim=0)
        model.load_state_dict(own_state)

        trainer.test(model=model, datamodule=datamodule)
        torch.save(model.net.state_dict(), os.path.join(ckpt_dir, 'wa.pth.tar'))

    metrics = trainer.callback_metrics
    return metrics, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training."""
    extras(cfg)
    train(cfg)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
