#!/bin/bash
# Full pipeline on one GPU. Submit from the repo root:  sbatch slurm/run_pipeline.sbatch
# Each stage is cached/resumable, so resubmitting after a timeout continues where it stopped.
#SBATCH --job-name=uavadv
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm/%x-%j.out
set -euo pipefail
# module load python/3.11 cuda   # adjust to your cluster
# source .venv/bin/activate
CFG="--config configs/lab.yaml"
python scripts/prepare_data.py $CFG
[ -f runs/baseline/weights/best.pt ] || python scripts/train.py $CFG
python scripts/calibrate_defense.py $CFG
python scripts/run_all.py $CFG
python scripts/train.py $CFG --set adv_train.enabled=true
python scripts/run_all.py $CFG --weights runs/advtrain/weights/best.pt --tag advtrain
python scripts/plot_results.py $CFG
