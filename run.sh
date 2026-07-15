#!/bin/bash
# Train ATGNN on FSD50K with 4 GPUs (DDP).
# Set DATA_DIR to the extracted FSD50K root and EXP_DIR to a scratch directory,
# or override paths on the command line (see configs/paths/default.yaml).

set -e

max_epochs=50
min_epochs=50
num_devices=4
batch_size=24 # per device (effective 96 on 4 GPUs)

HYDRA_FULL_ERROR=1 python src/train.py \
  trainer.max_epochs=$max_epochs \
  trainer.min_epochs=$min_epochs \
  trainer.devices=$num_devices \
  trainer.strategy=ddp \
  data.batch_size=$batch_size \
  "$@"
