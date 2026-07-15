import csv
import json
import random

import h5py
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset


def label_to_index(label_csv):
    """Map AudioSet-style label MIDs to integer indices from a CSV with columns `index`, `mid`."""
    index_dict = {}
    with open(label_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            index_dict[row['mid']] = int(row['index'])
    return index_dict


class FSDDataset(Dataset):
    """FSD50K dataset reading waveforms from HDF5 files and computing log-mel features on the fly.

    Manifest format: {"data": [{"wav": "/path/to/clip.h5", "labels": "mid1,mid2,..."}, ...]}
    """

    def __init__(self, data_json, conf, mode=None, label_csv=None):
        super().__init__()
        with open(data_json, 'r') as fp:
            data_json = json.load(fp)
        self.data = data_json['data']
        self.conf = conf
        self.mode = mode
        self.mixup = conf['mixup']
        self.index_dict = label_to_index(label_csv)
        self.num_labels = len(self.index_dict)

        self.num_mels = conf['num_mels']
        self.fmin = conf['fmin']
        self.fmax = conf['fmax']
        self.sr = conf['sr']
        self.window_type = conf['window_type']
        self.target_len = conf['target_len']
        self.freqm = conf['freqm']
        self.timem = conf['timem']
        self.norm_mean = conf['norm_mean']
        self.norm_std = conf['norm_std']

    def __len__(self):
        return len(self.data)

    def wav_2_fbank(self, waveform, waveform2=None):
        """Compute a Kaldi-style log-mel filterbank, optionally mixing two waveforms (mixup)."""
        if waveform2 is None:
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            waveform = waveform - waveform.mean()
            mix_lambda = 0
        else:
            waveform1 = waveform
            if waveform1.dim() == 1:
                waveform1 = waveform1.unsqueeze(0)
            if waveform2.dim() == 1:
                waveform2 = waveform2.unsqueeze(0)

            waveform1 = waveform1 - waveform1.mean()
            waveform2 = waveform2 - waveform2.mean()

            if waveform1.shape[1] != waveform2.shape[1]:
                if waveform1.shape[1] > waveform2.shape[1]:
                    temp_wav = torch.zeros(1, waveform1.shape[1])
                    temp_wav[0, 0:waveform2.shape[1]] = waveform2
                    waveform2 = temp_wav
                else:
                    waveform2 = waveform2[0, 0:waveform1.shape[1]].unsqueeze(0)

            mix_lambda = np.random.beta(10, 10)
            mix_waveform = mix_lambda * waveform1 + (1 - mix_lambda) * waveform2
            waveform = mix_waveform - mix_waveform.mean()

        fbank = torchaudio.compliance.kaldi.fbank(
            waveform, htk_compat=True, sample_frequency=self.sr, use_energy=False,
            window_type=self.window_type, num_mel_bins=self.num_mels, dither=0.0, frame_shift=10,
        )

        n_frames = fbank.shape[0]
        if n_frames < self.target_len:
            fbank = torch.nn.ZeroPad2d((0, 0, 0, self.target_len - n_frames))(fbank)
        elif n_frames > self.target_len:
            fbank = fbank[0:self.target_len, :]

        return fbank, mix_lambda

    def _load_audio(self, idx):
        with h5py.File(self.data[idx]['wav'], 'r') as f:
            return torch.from_numpy(f['audio'][:])

    def _label_vector(self, label_str, weight=1.0, out=None):
        if out is None:
            out = np.zeros(self.num_labels)
        for label in label_str.split(','):
            out[self.index_dict[label]] += weight
        return out

    def __getitem__(self, idx):
        if self.mode == 'train':
            if random.random() < self.mixup:
                mixup_idx = random.randint(0, len(self.data) - 1)
                fbank, mix_lambda = self.wav_2_fbank(self._load_audio(idx), self._load_audio(mixup_idx))
                label_list = self._label_vector(self.data[idx]['labels'], weight=mix_lambda)
                label_list = self._label_vector(self.data[mixup_idx]['labels'], weight=1 - mix_lambda, out=label_list)
            else:
                fbank, _ = self.wav_2_fbank(self._load_audio(idx))
                label_list = self._label_vector(self.data[idx]['labels'])
            label_list = torch.FloatTensor(label_list)

            # SpecAugment
            freqm = torchaudio.transforms.FrequencyMasking(self.freqm)
            timem = torchaudio.transforms.TimeMasking(self.timem)
            fbank = torch.transpose(fbank, 0, 1).unsqueeze(0)
            fbank = timem(freqm(fbank))
            fbank = torch.transpose(fbank.squeeze(0), 0, 1)

            fbank = (fbank - self.norm_mean) / self.norm_std
            return fbank, label_list
        else:
            fbank, _ = self.wav_2_fbank(self._load_audio(idx))
            label_list = torch.FloatTensor(self._label_vector(self.data[idx]['labels']))
            fbank = (fbank - self.norm_mean) / self.norm_std
            return fbank, label_list
