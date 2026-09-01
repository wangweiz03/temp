# ERS MLE Framework

ERS MLE Framework is a multi-round search and evaluation harness for MLE-bench Lite and Kaggle-style machine-learning engineering tasks. It uses OpenAI Codex to propose a single-file solution, validates each candidate in an external sandbox, and evolves the next action from execution feedback and compact task memory.

The runtime controller implements Experience Routed Search (ERS), the search component described in the MetaMLE paper. Each round selects one action and exactly one full Skill source:

| Action | Goal | Skill source |
| --- | --- | --- |
| `draft` | Build or restore a strong runnable solution | Task Skill |
| `debug` | Fix the latest concrete failure | Failure Skill |
| `improve` | Raise the validation score | Task Skill |

Every action routes exactly one Skill before generating and validating `solution.py`. Draft rounds first compress the Task Skill into `planning.md`. Debug and improve rounds skip that planning call and pass the routed Failure Skill or Task Skill directly to coding.

## Highlights

- Adaptive `draft`, `debug`, and `improve` action scheduling.
- Single-source Skill routing aligned with the ERS design.
- Explicit planning before draft code generation; direct Skill-to-code execution for debug and improve rounds.
- Internal validation, failure classification, final submission evaluation, grade calculation, and medal reporting.
- Lightweight commit, branch, tag, and memory records without requiring Git inside task runs.
- Bundled task Skills for all 22 MLE-bench Lite tasks plus the failure-prevention Skill.
- Portable single-task and 12-hour full-suite shell entry points.

## Repository layout

```text
.
├── evaluate_codex.py                # ERS controller and evaluation driver
├── prompts.py                       # Runtime planning and coding prompts
├── skills/                          # 22 task Skills and the Failure Skill
├── paper_prompts/                   # Non-runtime appendix prompts as plain text
├── tts_search/
│   ├── eval_utils.py                # Leaderboard, grade, medal, and summary utilities
│   └── reward_func_utils.py         # Sandbox client, code extraction, and rewards
├── scripts/
│   ├── run_single_task.sh           # Run one selected task
│   ├── run_mlebench_lite_12h.sh     # Run the complete 12-hour evaluation
│   └── select_task.py               # Select a task from JSON or Parquet
├── tests/                            # Offline unit tests
├── docs/                             # Method and paper-alignment notes
├── .env.example
└── requirements.txt
```

## Requirements

- Python 3.10 or newer.
- The `codex` CLI available on `PATH` and authenticated for non-interactive `codex exec` calls.
- An MLE sandbox service implementing the API described below.
- A JSON or Parquet task file and the matching validation/submission datasets.

Install the Python dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Copy the environment template and configure the sandbox and data locations:

```bash
cp .env.example .env
```

The repository uses its bundled `skills/` directory by default. `MLE_TASK_SKILLS_DIR` and `MLE_ERROR_SKILL_FILE` are optional overrides, not required external assets.

## Configuration

| Variable | Purpose | Default |
| --- | --- | --- |
| `MLE_SANDBOX_BASE_URL` | Sandbox API base URL | `http://127.0.0.1:6580` |
| `MLE_SANDBOX_API_KEY` | Optional value sent as `X-API-Key` | empty |
| `MLE_TASK_SKILLS_DIR` | Directory containing `SKILL_<task>.md` | `./skills` |
| `MLE_ERROR_SKILL_FILE` | Failure-prevention Skill | `./skills/SKILL_error.md` |
| `MLE_SUBMIT_DATA_ROOT` | Final-evaluation data root | `./external/submission_data` |
| `MLE_LEADERBOARD_DIR` | Root containing `<task>/public_leaderboard.csv` | unset |
| `MLE_HF_ENDPOINT` | Optional model-download endpoint propagated to sandbox jobs | unset |
| `MLE_MODEL` | Model used by the shell scripts | `gpt-5.4` |
| `MLE_REASONING_LEVEL` | Codex reasoning level used by the shell scripts | `medium` |

Command-line arguments override the corresponding defaults. Run `python evaluate_codex.py --help` for the complete list.

## Task file format

The top-level value is a list of task records. A Parquet file must contain rows that convert to the same structure.

```json
[
  {
    "metadata": {
      "task_name": "example-task",
      "cpu_gpu": "cpu",
      "data_dir": "example-task-validation",
      "higher_is_better": true,
      "theoretical_min": 0.0,
      "theoretical_max": 1.0,
      "leaderboard_min": 0.1,
      "leaderboard_max": 0.9,
      "data_description": "Files, schema, and modality summary",
      "task_description": "Metric, target, and submission contract"
    },
    "prompt": [
      {"role": "system", "content": "Execution environment instructions"},
      {"role": "user", "content": "Task and output instructions"}
    ]
  }
]
```

`metadata.data_dir` is the validation data identifier passed to the sandbox. Final evaluation resolves the task under `MLE_SUBMIT_DATA_ROOT`.

## Run the framework

Run every record in a task file:

```bash
python evaluate_codex.py \
  --data-file tasks.parquet \
  --output-dir runs/experiment \
  --model gpt-5.4 \
  --reasoning-level medium \
  --num-rounds 10 \
  --concurrency 2
```

Run one task:

```bash
./scripts/run_single_task.sh \
  --task-name mlsp-2013-birds \
  --data-file tasks.parquet \
  --num-rounds 10
```

Run the complete MLE-bench Lite configuration with a 12-hour per-task budget:

```bash
./scripts/run_mlebench_lite_12h.sh --data-file tasks.parquet
```

The full-suite script mirrors the research configuration: 100 maximum rounds, 43,200 seconds per task, adaptive routing, `draft,improve` warmup actions, 100,000 maximum generation tokens, and 22 concurrent tasks. Every setting can be overridden with a command-line option or environment variable.

## Sandbox API contract

The client uses:

- `POST /api/v1/jobs` to submit `name`, `code`, `data_dir`, `timeout`, `resource_type`, and `environment`.
- `GET /api/v1/jobs/{job_id}` to poll until the status is no longer `running` or `queued`.

The submit response must contain `job_id`. A completed response is expected to expose `status` and `result`; commonly used result fields are `result`, `score`, and `run_log`. Adapt `get_sandbox_result` in `tts_search/reward_func_utils.py` if your execution service uses a different protocol.

## Outputs

Each task output directory contains:

- `commits/<hash>/`: the draft plan or no-planning record, generated solution, validation feedback, summary, and result snapshot for one round.
- `refs/` and `tags/`: action heads and best-candidate references.
- `traces/`: Codex inputs, outputs, diagnostics, and approximate token usage.
- `memories/`: compact task memory and previous revisions.
- `rounds_summary.json` and `stat.json`: task-level summaries.

The top-level output directory also contains the resolved run configuration and aggregate summary.

Traces can contain full task prompts, metadata, generated code, and sandbox logs. Treat generated run directories as operational artifacts: review them before sharing, apply an appropriate retention policy, and never commit credentials or private task data.

## Tests and quality checks

```bash
python -m unittest discover -s tests -v
black --check .
ruff check .
```

The unit tests are offline and do not call Codex, a sandbox, or real benchmark data.

## Paper alignment

The implementation follows the ERS routing and planning flow described in [MetaMLE](https://openreview.net/forum?id=pBRusdNasm). The appendix artifacts are mapped as follows:

- Table 17 baseline experiment prompt: `paper_prompts/baseline_experiment_prompt.txt`; its system instructions are also `SYSTEM_PROMPT` in `prompts.py`.
- Table 18 task-level Skill synthesis prompt: `paper_prompts/task_level_skill_synthesis_prompt.txt`.
- Table 19 task-level Skill example: `skills/SKILL_mlsp-2013-birds.md`.
- Table 20 failure-prevention Skill example: `skills/SKILL_error.md`.

The paper explicitly replaces several long sections with “omitted for brevity” placeholders. The repository retains reasonable full-detail content for those omitted sections while preserving the appendix's non-omitted wording and structure. The runtime intentionally skips planning for debug and improve rounds, as documented in `docs/PAPER_ALIGNMENT.md`; this differs from the paper's all-branch planning description.
