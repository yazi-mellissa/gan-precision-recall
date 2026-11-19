#!/bin/bash
# Slurm array job launcher for training: reads line $SLURM_ARRAY_TASK_ID of a given file.
# Usage:
#   bash scripts/train_array.sh experiments/e1_vanilla.txt
# or:
#   sbatch scripts/train_array.sh experiments/e1_vanilla.txt
# Longer jobs (default limit 2 h):
#   TIME=03:30:00 bash scripts/train_array.sh experiments/<file>.txt

FILE=$1
if [ -z "$FILE" ]; then
    echo "Usage: $0 <experiment_file.txt>"
    exit 1
fi

if [ ! -f "$FILE" ]; then
    echo "Error: File $FILE not found."
    exit 1
fi

# Count executable lines (exclude empty and comment lines)
N=$(grep -v '^[[:space:]]*#' "$FILE" | grep -v '^[[:space:]]*$' | wc -l)

if [ "$N" -eq 0 ]; then
    echo "No executable commands found in $FILE"
    exit 1
fi

# If invoked outside Slurm, submit itself as an array job
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    JOB_NAME="tr_$(basename "$FILE" .txt)"
    mkdir -p outputs
    echo "Submitting array job (1-$N) for $FILE with job-name $JOB_NAME..."
    sbatch --partition=mesonet \
           --account=m25146 \
           --nodes=1 \
           -c 8 \
           --mem=32G \
           --gres=gpu:1 \
           --time=${TIME:-02:00:00} \
           --job-name="$JOB_NAME" \
           --output="outputs/%x_%A_%a.out" \
           --error="outputs/%x_%A_%a.err" \
           --array=1-$N \
           "$0" "$FILE"
    exit 0
fi

# Inside Slurm task execution
export DATA="/projects/m25146/data/"
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

CMD=$(grep -v '^[[:space:]]*#' "$FILE" | grep -v '^[[:space:]]*$' | sed -n "${SLURM_ARRAY_TASK_ID}p")
echo "=================================================="
echo "Starting Slurm Task ID: $SLURM_ARRAY_TASK_ID"
echo "Command: $CMD"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "=================================================="

eval "$CMD"

EXIT_CODE=$?
echo "=================================================="
echo "Task completed with exit code $EXIT_CODE at $(date)"
echo "=================================================="
exit $EXIT_CODE
