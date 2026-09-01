# Experience Routed Search

Experience Routed Search (ERS) treats multi-round machine-learning engineering as search with compact memory. At round `r`, the controller observes the current best program, recent results, recent method categories, and task memory. It then chooses an action, a search intent, and one full Skill source.

## Actions and routing

| Action | Goal | Routed source |
| --- | --- | --- |
| Draft | Build or restore a strong runnable solution | Task Skill |
| Debug | Repair the latest concrete failure | Failure Skill |
| Improve | Raise the validation score | Task Skill |

Only one full Skill source is selected per round. Controller contracts may constrain the requested change, but they do not introduce another Skill source.

## Planning and implementation

The runtime uses branch-specific execution:

1. Draft receives the Task Skill in a planning call, writes `planning.md`, and generates `solution.py` from that contract.
2. Debug receives the Failure Skill directly during coding and skips the planning call.
3. Improve receives the Task Skill directly during coding and skips the planning call.
4. The sandbox executes the candidate against the internal validation split.
5. ERS records the result and updates compact task memory.

For archival consistency, debug and improve commits contain a short no-planning record instead of a generated plan.

## Branch selection

ERS begins with warmup actions, then uses execution evidence:

- Choose debug after a concrete failure.
- Choose improve after a valid scored run.
- Choose draft when establishing or restoring a runnable baseline.
- Switch an improve action from local tuning to alternative exploration when recent valid attempts fail to match the best score or repeat one method category too often.

The controller does not predict the best model directly. It selects the type of experience that should guide the next coding action.

## Compact memory

Each completed round records the four fields listed in the paper appendix:

- Method summary: model family, feature pipeline, validation split, and output writer.
- Result reflection: whether the run failed, improved, or revealed a task fact.
- Method category: a concise tag such as linear model, tree ensemble, CNN, transformer, or post-processing.
- Relative change: what changed from the current best program and what stayed fixed.

The persistent task memory keeps only facts that can influence later branch selection or implementation. It is not a full transcript.

## Evaluation protocol

Candidate programs run on an internal training/evaluation split during search. After the budget expires, the valid candidate with the best internal score is executed against the final task data. Grade is the normalized public-leaderboard rank; an unscored run receives grade `1.0`.
