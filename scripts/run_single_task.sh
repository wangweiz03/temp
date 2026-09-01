#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

TASK_NAME=""
DATA_FILE=""
OUTPUT_BASE="$REPOSITORY_ROOT/runs"
MODEL="${MLE_MODEL:-gpt-5.4}"
REASONING_LEVEL="${MLE_REASONING_LEVEL:-medium}"
NUM_ROUNDS=3
TIME_BUDGET=43200
MAX_TOKENS=100000
TEMPERATURE=0.6
TASK_SKILLS_DIR="${MLE_TASK_SKILLS_DIR:-$REPOSITORY_ROOT/skills}"
ERROR_SKILL_FILE="${MLE_ERROR_SKILL_FILE:-$REPOSITORY_ROOT/skills/SKILL_error.md}"
BRANCH_STRATEGY="adaptive"
WARMUP_BRANCHES="draft,improve"

usage() {
    echo "Usage: $0 --task-name NAME --data-file PATH [options]"
    echo "Run '$0 --help' for the complete option list."
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --task-name) TASK_NAME="$2"; shift 2 ;;
        --data-file) DATA_FILE="$2"; shift 2 ;;
        --output-dir) OUTPUT_BASE="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --reasoning-level) REASONING_LEVEL="$2"; shift 2 ;;
        --num-rounds) NUM_ROUNDS="$2"; shift 2 ;;
        --time-budget) TIME_BUDGET="$2"; shift 2 ;;
        --max-tokens) MAX_TOKENS="$2"; shift 2 ;;
        --temperature) TEMPERATURE="$2"; shift 2 ;;
        --task-skills-dir) TASK_SKILLS_DIR="$2"; shift 2 ;;
        --error-skill-file) ERROR_SKILL_FILE="$2"; shift 2 ;;
        --branch-strategy) BRANCH_STRATEGY="$2"; shift 2 ;;
        --warmup-branches) WARMUP_BRANCHES="$2"; shift 2 ;;
        --help)
            usage
            echo ""
            echo "Required: --task-name, --data-file"
            echo "Common:   --output-dir, --model, --reasoning-level, --num-rounds"
            echo "Assets:   --task-skills-dir, --error-skill-file"
            exit 0
            ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -z "$TASK_NAME" || -z "$DATA_FILE" ]]; then
    usage >&2
    exit 2
fi

TMPFILE="$(mktemp "${TMPDIR:-/tmp}/ers-single-task.XXXXXX.json")"
trap 'rm -f "$TMPFILE"' EXIT

python3 "$SCRIPT_DIR/select_task.py" \
    --data-file "$DATA_FILE" \
    --task-name "$TASK_NAME" \
    --output-file "$TMPFILE"

OUTPUT_DIR="$OUTPUT_BASE/$TASK_NAME"
python3 "$REPOSITORY_ROOT/evaluate_codex.py" \
    --data-file "$TMPFILE" \
    --output-dir "$OUTPUT_DIR" \
    --model "$MODEL" \
    --reasoning-level "$REASONING_LEVEL" \
    --num-rounds "$NUM_ROUNDS" \
    --time-budget "$TIME_BUDGET" \
    --max-tokens "$MAX_TOKENS" \
    --temperature "$TEMPERATURE" \
    --concurrency 1 \
    --branch-strategy "$BRANCH_STRATEGY" \
    --warmup-branches "$WARMUP_BRANCHES" \
    --task-skills-dir "$TASK_SKILLS_DIR" \
    --error-skill-file "$ERROR_SKILL_FILE"

echo "Done. Results saved to $OUTPUT_DIR"
