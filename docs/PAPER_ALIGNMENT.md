# Paper alignment

This repository maps the public appendix artifacts from the MetaMLE paper to executable or inspectable source files.

| Paper artifact | Repository artifact | Conformance rule |
| --- | --- | --- |
| Table 17, baseline experiment prompt | `SYSTEM_PROMPT` in `prompts.py` | The runtime template is preserved; task metadata and benchmark messages are supplied dynamically. The former duplicate text file is not included. |
| Table 18, task-level Skill synthesis prompt | Not included | This prompt is unrelated to ERS execution and is currently absent from the repository. |
| Table 19, MLSP task Skill example | `skills/SKILL_mlsp-2013-birds.md` | The complete Skill contains the non-omitted appendix text plus reasonable detail in sections abbreviated by the paper. |
| Table 20, failure-prevention Skill example | `skills/SKILL_error.md` | The title, workflow, seven-contract structure, compact heuristics, and closing exclusions match the non-omitted appendix text; detailed requirements replace the paper's omission placeholders. |

The ERS runtime also follows the method text surrounding the appendix:

- Draft routes the Task Skill.
- Debug routes the Failure Skill.
- Improve routes the Task Skill.
- Each round routes one full Skill source.
- Runtime summaries use method summary, result reflection, method category, and relative change.

One runtime choice intentionally differs from Section 4.3 of the paper: only draft creates a plan; debug and improve pass their routed full Skill directly to coding. The standalone pre-planning data-inspection subsystem from the older experimental snapshot has been removed. The paper describes an optional lightweight read-only draft check but does not specify the former multi-call implementation.
