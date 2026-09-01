#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

DATA_FILE="${MLE_DATA_FILE:-}"
OUTPUT_DIR="${MLE_OUTPUT_DIR:-$REPOSITORY_ROOT/runs/mlebench-lite-12h}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL="${MLE_MODEL:-gpt-5.4}"
REASONING_LEVEL="${MLE_REASONING_LEVEL:-medium}"
NUM_ROUNDS="${MLE_NUM_ROUNDS:-100}"
TIME_BUDGET="${MLE_TIME_BUDGET:-43200}"
CONCURRENCY="${MLE_CONCURRENCY:-22}"
MAX_TOKENS="${MLE_MAX_TOKENS:-100000}"
TEMPERATURE="${MLE_TEMPERATURE:-0.6}"
TASK_SKILLS_DIR="${MLE_TASK_SKILLS_DIR:-$REPOSITORY_ROOT/skills}"
ERROR_SKILL_FILE="${MLE_ERROR_SKILL_FILE:-$REPOSITORY_ROOT/skills/SKILL_error.md}"

usage() {
    echo "Usage: $0 --data-file PATH [options]"
    echo "Options: --output-dir, --model, --reasoning-level, --num-rounds, --time-budget, --concurrency"
    echo "         --max-tokens, --temperature, --task-skills-dir, --error-skill-file"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-file) DATA_FILE="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --reasoning-level) REASONING_LEVEL="$2"; shift 2 ;;
        --num-rounds) NUM_ROUNDS="$2"; shift 2 ;;
        --time-budget) TIME_BUDGET="$2"; shift 2 ;;
        --concurrency) CONCURRENCY="$2"; shift 2 ;;
        --max-tokens) MAX_TOKENS="$2"; shift 2 ;;
        --temperature) TEMPERATURE="$2"; shift 2 ;;
        --task-skills-dir) TASK_SKILLS_DIR="$2"; shift 2 ;;
        --error-skill-file) ERROR_SKILL_FILE="$2"; shift 2 ;;
        --help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -z "$DATA_FILE" ]]; then
    echo "--data-file or MLE_DATA_FILE is required." >&2
    usage >&2
    exit 2
fi

if [[ ! -f "$DATA_FILE" ]]; then
    echo "Task data file not found: $DATA_FILE" >&2
    exit 2
fi

echo "Starting ERS MLE-bench Lite evaluation"
echo "Data file: $DATA_FILE"
echo "Output: $OUTPUT_DIR"
echo "Model: $MODEL ($REASONING_LEVEL)"
echo "Budget: ${TIME_BUDGET}s per task"
echo "Concurrency: $CONCURRENCY"

"$PYTHON_BIN" "$REPOSITORY_ROOT/evaluate_codex.py" \
    --data-file "$DATA_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --model "$MODEL" \
    --reasoning-level "$REASONING_LEVEL" \
    --num-rounds "$NUM_ROUNDS" \
    --time-budget "$TIME_BUDGET" \
    --concurrency "$CONCURRENCY" \
    --branch-strategy adaptive \
    --warmup-branches draft,improve \
    --max-tokens "$MAX_TOKENS" \
    --temperature "$TEMPERATURE" \
    --task-skills-dir "$TASK_SKILLS_DIR" \
    --error-skill-file "$ERROR_SKILL_FILE"

echo "Evaluation complete. Results are in $OUTPUT_DIR"
