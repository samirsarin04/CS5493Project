#!/bin/bash

set -e

cd /scratch/general/vast/u0760641/FENStockFish
mkdir -p logs/slurm

source .venv/bin/activate

export TRANSFORMERS_OFFLINE=1
export HF_HOME=/scratch/general/vast/$USER/.cache/huggingface

python llm_inference.py --limit 3 --output test_output.csv

cat test_output.csv
