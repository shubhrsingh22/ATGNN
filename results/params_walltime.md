# Parameter counts and walltime (ATGNN, FSD50K input)

GPU: NVIDIA L40S | PyTorch 2.2.0+cu118 (CUDA 11.8) | float32, no JIT/torch.compile | batch 24 x 10 s clips (1024 frames x 128 mels) | Adam, BCEWithLogits loss.
Train step = forward+backward+optimizer; inference under no_grad. Mean over 10 (train) / 20 (inference) iterations after warmup.

| Model | Params (M) | Train step (ms) | Inference (ms/batch) | Inference (ms/clip) |
|---|---|---|---|---|
| atgnn_s_encoder_only | 26.8 | 258.0 | 121.8 | 5.08 |
| atgnn_s_full | 35.4 | 289.9 | 136.8 | 5.7 |
