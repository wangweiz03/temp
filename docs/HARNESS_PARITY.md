# Harness parity

The Claude Code transport was compared with the supplied internal reference
implementation. The reference and this repository use the same core loop:

1. Select a `draft`, `debug`, or `improve` action.
2. Route the Task Skill for draft/improve or the Failure Skill for debug.
3. Run planning for draft only; debug and improve code directly.
4. Generate one complete `solution.py`.
5. Validate it in the sandbox and classify failures.
6. Archive the attempt, update search state, and refresh task memory.

The shared functions in `evaluate.py` perform all six steps for both
harnesses. Harness selection changes only command construction, environment
handling, timeout behavior, and conversion of print-only Claude Code output
into `planning.md`, `solution.py`, and task memory.

The following reference-only behavior was deliberately not copied because it
would change the published ERS implementation or violate repository hygiene:

- EDA and BSPM-era routing or prompt content.
- Legacy method-family summary fields in place of the paper-aligned summary.
- Hard-coded sandbox endpoints, filesystem locations other than the requested
  default task Parquet, or credentials.
- Separate branch-selection logic for Claude Code.

Therefore `codex` and `claude-code` are transport variants of one ERS
algorithm, not independent algorithms.
