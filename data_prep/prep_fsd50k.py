"""Prepare FSD50K for training.

Resamples every clip to 16 kHz, stores each waveform in an HDF5 file, and writes
JSON manifests (train / val / eval) plus the label CSV used by the datamodule.

Usage:
    python data_prep/prep_fsd50k.py \
        --data_dir /path/to/FSD50K \
        --out_dir /path/to/experiments/hdf_fsd50k \
        --num_workers 32

`--data_dir` must contain dev_audio/, eval_audio/ and ground_truth/{dev.csv,eval.csv,vocabulary.csv}.
"""

import argparse
import csv
import json
import os
from functools import partial
from multiprocessing import Pool

import h5py
import librosa


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Root of the extracted FSD50K dataset')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='Output directory for HDF5 files and JSON manifests')
    parser.add_argument('--target_sr', type=int, default=16000)
    parser.add_argument('--num_workers', type=int, default=os.cpu_count())
    return parser.parse_args()


def resample_and_save_to_hdf5(file_path, hdf_dir, target_sr=16000):
    hdf5_path = os.path.join(hdf_dir, os.path.basename(file_path).replace('.wav', '.h5'))
    if os.path.exists(hdf5_path):
        return hdf5_path
    audio, sr = librosa.load(file_path, sr=None)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    with h5py.File(hdf5_path, 'w') as f:
        f.create_dataset('audio', data=audio.astype('float32'), compression="gzip")
    return hdf5_path


def read_ground_truth(csv_path, has_split):
    """Read dev.csv / eval.csv. Returns a list of (fname, mids, split)."""
    rows = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            split = row['split'] if has_split else None
            rows.append((row['fname'], row['mids'], split))
    return rows


def write_label_csv(vocabulary_csv, out_csv):
    """Convert FSD50K vocabulary.csv (index,name,mid without header) to the label CSV
    with header (index,mid,display_name) expected by the datamodule."""
    with open(vocabulary_csv, 'r') as f:
        rows = list(csv.reader(f))
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['index', 'mid', 'display_name'])
        for index, name, mid in rows:
            writer.writerow([index, mid, name])
    print(f'Wrote {len(rows)} classes to {out_csv}')


def process_split(rows, audio_dir, hdf_dir, target_sr, num_workers):
    os.makedirs(hdf_dir, exist_ok=True)
    wav_paths = [os.path.join(audio_dir, f'{fname}.wav') for fname, _, _ in rows]
    with Pool(num_workers) as pool:
        hdf5_paths = pool.map(partial(resample_and_save_to_hdf5, hdf_dir=hdf_dir, target_sr=target_sr), wav_paths)
    return hdf5_paths


def main():
    args = parse_args()
    gt_dir = os.path.join(args.data_dir, 'ground_truth')
    datafiles_dir = os.path.join(args.out_dir, 'datafiles')
    os.makedirs(datafiles_dir, exist_ok=True)

    write_label_csv(os.path.join(gt_dir, 'vocabulary.csv'),
                    os.path.join(datafiles_dir, 'class_labels_indices.csv'))

    # dev set -> train + val manifests
    dev_rows = read_ground_truth(os.path.join(gt_dir, 'dev.csv'), has_split=True)
    print(f'Processing {len(dev_rows)} dev clips...')
    dev_h5 = process_split(dev_rows, os.path.join(args.data_dir, 'dev_audio'),
                           os.path.join(args.out_dir, 'dev_audio'), args.target_sr, args.num_workers)
    tr_data, val_data = [], []
    for (fname, mids, split), h5_path in zip(dev_rows, dev_h5):
        entry = {'wav': h5_path, 'labels': mids}
        if split == 'train':
            tr_data.append(entry)
        elif split == 'val':
            val_data.append(entry)
        else:
            raise ValueError(f'Unrecognised split "{split}" for {fname}')

    # eval set
    eval_rows = read_ground_truth(os.path.join(gt_dir, 'eval.csv'), has_split=False)
    print(f'Processing {len(eval_rows)} eval clips...')
    eval_h5 = process_split(eval_rows, os.path.join(args.data_dir, 'eval_audio'),
                            os.path.join(args.out_dir, 'eval_audio'), args.target_sr, args.num_workers)
    eval_data = [{'wav': h5_path, 'labels': mids} for (_, mids, _), h5_path in zip(eval_rows, eval_h5)]

    for name, data in [('fsd_tr_full.json', tr_data), ('fsd_val_full.json', val_data), ('fsd_eval_full.json', eval_data)]:
        with open(os.path.join(datafiles_dir, name), 'w') as f:
            json.dump({'data': data}, f, indent=1)

    print(f'Done. train={len(tr_data)} val={len(val_data)} eval={len(eval_data)}')
    print(f'Manifests written to {datafiles_dir}')


if __name__ == '__main__':
    main()
