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
#   sbatch run_llm_inference.sh qwen25
#   sbatch run_llm_inference.sh chessSLM data/chaotic_fens.csv
#
# To submit all four at once:
#   for m in llama1b llama8b qwen25 chessSLM; do sbatch run_llm_inference.sh $m; done
#   for m in llama1b llama8b qwen25 chessSLM; do sbatch run_llm_inference.sh $m data/chaotic_fens.csv; done

set -e

MODEL=${1:?"Usage: sbatch run_llm_inference.sh <model>  (llama1b | llama8b | qwen25 | chessSLM)"}
INPUT=${2:-data/fuzz_fens.csv}

cd $SLURM_SUBMIT_DIR

source /scratch/general/vast/u0760641/CS5966-Interpretable-Routing/.venv/bin/activate

export TRANSFORMERS_OFFLINE=1
export HF_HOME=/scratch/general/vast/$USER/.cache/huggingface
export SCRATCH=/scratch/general/vast/$USER

echo "Starting model: $MODEL  input: $INPUT  at $(date)"

python llm_inference.py \
    --model "$MODEL" \
    --input "$INPUT"

echo "Job finished: $(date)"
