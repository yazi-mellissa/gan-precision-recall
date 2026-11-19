#!/bin/bash
#SBATCH --partition=mesonet
#SBATCH --account=m25146
#SBATCH --nodes=1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --job-name=eval_plain
#SBATCH --output=outputs/%x_%j.out
#SBATCH --error=outputs/%x_%j.err
# Score trained generators exactly as the platform would: plain sampling with generate.py, then eval/evaluate.py.
# Each argument is a folder holding G.pth; results go to <folder>/results_plain.json.
# Usage (from the repo root): sbatch scripts/eval_plain.sbatch runs/e12_*_seed1 runs/e12_*_seed1/ema

export DATA="/projects/m25146/data/"
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

for dir in "$@"; do
    echo "=== $dir ==="
    python generate.py --ckpt_dir "$dir" --out_dir "$dir/samples"
    python eval/evaluate.py --samples_dir "$dir/samples" --run_dir "$dir" --out "$dir/results_plain.json"
done
