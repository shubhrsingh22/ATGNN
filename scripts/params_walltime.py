"""Parameter counts and walltime benchmark for ATGNN (encoder-only and full).

Measures, for each variant on a single GPU in float32 (no JIT / torch.compile):
  - trainable parameter count
  - training step time: forward + BCEWithLogits loss + backward + Adam step
  - inference time: forward only under torch.no_grad()

Input matches FSD50K training: batch 24 log-mel spectrograms of 1024 frames x
128 mels (10 s at 16 kHz). float32 is used for both variants for comparability
(the full model requires fp32; see README).

Usage:
    python scripts/params_walltime.py [--batch_size 24] [--device cuda:0] \
        [--out results/params_walltime.md]
"""

import argparse
import csv
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.components.atgnn import ATGNN  # noqa: E402

MODELS = {
    "atgnn_s_encoder_only": dict(size='s', num_class=200, use_mlg=False),
    "atgnn_s_full": dict(size='s', num_class=200, use_mlg=True),
}

FRAMES, MELS, NUM_CLASSES = 1024, 128, 200


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def time_loop(fn, warmup, iters, device):
    for _ in range(warmup):
        fn()
    sync(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    sync(device)
    return (time.perf_counter() - t0) / iters * 1000.0  # ms


def benchmark(name, kwargs, batch_size, device):
    torch.manual_seed(0)
    model = ATGNN(**kwargs).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    x = torch.randn(batch_size, FRAMES, MELS, device=device)
    y = torch.randint(0, 2, (batch_size, NUM_CLASSES), device=device).float()
    opt = torch.optim.Adam(model.parameters(), lr=5e-4)
    crit = torch.nn.BCEWithLogitsLoss()

    model.train()

    def train_step():
        opt.zero_grad(set_to_none=True)
        loss = crit(model(x), y)
        loss.backward()
        opt.step()

    train_ms = time_loop(train_step, warmup=3, iters=10, device=device)

    model.eval()

    def infer_step():
        with torch.no_grad():
            model(x)

    infer_ms = time_loop(infer_step, warmup=5, iters=20, device=device)

    return dict(model=name, params=n_params,
                train_ms_per_step=round(train_ms, 1),
                infer_ms_per_batch=round(infer_ms, 1),
                infer_ms_per_clip=round(infer_ms / batch_size, 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch_size", type=int, default=24)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="results/params_walltime.md")
    args = ap.parse_args()
    device = torch.device(args.device)

    rows = []
    for name, kwargs in MODELS.items():
        rows.append(benchmark(name, kwargs, args.batch_size, device))
        print(rows[-1])

    gpu = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    header = (
        f"# Parameter counts and walltime (ATGNN, FSD50K input)\n\n"
        f"GPU: {gpu} | PyTorch {torch.__version__} (CUDA {torch.version.cuda}) | "
        f"float32, no JIT/torch.compile | batch {args.batch_size} x 10 s clips "
        f"({FRAMES} frames x {MELS} mels) | Adam, BCEWithLogits loss.\n"
        f"Train step = forward+backward+optimizer; inference under no_grad. "
        f"Mean over 10 (train) / 20 (inference) iterations after warmup.\n\n"
    )
    table = "| Model | Params (M) | Train step (ms) | Inference (ms/batch) | Inference (ms/clip) |\n"
    table += "|---|---|---|---|---|\n"
    for r in rows:
        table += (f"| {r['model']} | {r['params']/1e6:.1f} | {r['train_ms_per_step']} "
                  f"| {r['infer_ms_per_batch']} | {r['infer_ms_per_clip']} |\n")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(header + table)
    with open(args.out.replace(".md", ".csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()
