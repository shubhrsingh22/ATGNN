import os
from typing import Any, Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from src.data.dataset import FSDDataset


class FSDDataModule(LightningDataModule):
    """DataModule for FSD50K.

    Expects the JSON manifests and label CSV produced by `data_prep/prep_fsd50k.py`:
      {json_path}/fsd_tr_full.json, fsd_val_full.json, fsd_eval_full.json
    Each manifest entry points to an HDF5 file holding the resampled waveform.
    """

    def __init__(
        self,
        json_path: str,
        data_dir: str,
        meta_path: str,
        label_csv_pth: str,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        persistent_workers: bool,
        sr: int,
        fmin: int,
        fmax: int,
        num_mels: int,
        window_type: str,
        target_len: int,
        freqm: int,
        timem: int,
        mixup: float,
        norm_mean: float,
        norm_std: float,
        num_devices: int,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.json_path = json_path
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.label_csv_pth = label_csv_pth
        self.data_dir = data_dir
        self.meta_path = meta_path
        self.num_devices = int(num_devices)

        self.train_json = os.path.join(json_path, 'fsd_tr_full.json')
        self.val_json = os.path.join(json_path, 'fsd_val_full.json')
        self.eval_json = os.path.join(json_path, 'fsd_eval_full.json')
        self.audio_conf = {
            'sr': sr, 'fmin': fmin, 'fmax': fmax, 'num_mels': num_mels,
            'window_type': window_type, 'target_len': target_len,
            'freqm': freqm, 'timem': timem,
            'norm_mean': norm_mean, 'norm_std': norm_std, 'mixup': mixup,
        }

    def setup(self, stage: Optional[str] = None) -> None:
        for path in (self.train_json, self.val_json, self.eval_json, self.label_csv_pth):
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"{path} not found. Run `python data_prep/prep_fsd50k.py` first."
                )
        self.train_dataset = FSDDataset(self.train_json, self.audio_conf, mode='train', label_csv=self.label_csv_pth)
        self.val_dataset = FSDDataset(self.val_json, self.audio_conf, mode='val', label_csv=self.label_csv_pth)
        self.eval_dataset = FSDDataset(self.eval_json, self.audio_conf, mode='eval', label_csv=self.label_csv_pth)

    def train_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            shuffle=False,
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.eval_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            shuffle=False,
            drop_last=False,
        )
