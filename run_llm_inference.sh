#!/bin/bash
#SBATCH --job-name=llm-chess
#SBATCH --mem=100G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/%j.out
#SBATCH --error=logs/slurm/%j.err
#SBATCH --gres=gpu:rtxpr6000bl:1
#SBATCH --partition=soc-gpu-class-grn
#SBATCH --account=cs6966
#SBATCH --qos=soc-gpu-class-grn

# Usage:
#   sbatch run_llm_inference.sh llama1b
#   sbatch run_llm_inference.sh llama8b
#   sbatch run_llm_inference.sh qwen25
#
# To submit all three at once:
#   for m in llama1b llama8b qwen25; do sbatch run_llm_inference.sh $m; done

set -e

MODEL=${1:?"Usage: sbatch run_llm_inference.sh <model>  (llama1b | llama8b | qwen25)"}

cd $SLURM_SUBMIT_DIR

source /scratch/general/vast/u0760641/CS5966-Interpretable-Routing/.venv/bin/activate

export TRANSFORMERS_OFFLINE=1
export HF_HOME=/scratch/general/vast/$USER/.cache/huggingface
export SCRATCH=/scratch/general/vast/$USER

echo "Starting model: $MODEL at $(date)"

python llm_inference.py \
    --model "$MODEL" \
    --input data/fuzz_fens.csv \
    --output /scratch/general/vast/$USER/llm_results_${MODEL}.csv

echo "Job finished: $(date)"
