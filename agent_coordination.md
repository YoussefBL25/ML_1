# Agent coordination

Shared board for the agents (and people) working on this repository, mainly
on `report/main.tex`. **Every agent reads this file after `git pull` and
before any edit**, and updates it in the same commit as its change. Delete
this file before the final submission.

## Agents and owners

| Agent | Works for | Owns in `report/main.tex` | Owns in code |
|---|---|---|---|
| **P1** | Hassen Ben Chaabane (Person 1) | Data description, imputation and scaling | `boosting.py`, `search_boosting.py`, `tune_boosting.py` |
| **P2** | Meriem Laataoui (Person 2) | Codebook rules, protocol, logistic models, final model, results, Tables I and III, Fig. 1 | `imbalance.py`, `plots.py` |
| **P3 (referee)** | Youssef Ben Lassoued (Person 3) | Abstract, introduction, linear models, linear rows of Table II, conclusion, reproducibility | `cross_validation.py`, `metrics.py`, `tune_linear.py` |

Shared, no single owner (changes need a proposal): `preprocess.py`,
`interactions.py`, `evaluate_best.py`, `tune_models.py`,
`tuning.py`, `threshold.py`, `models.py`, `data.py`, `run.py`, `README.md`,
`config/`, the report preamble and float placement.

If the table is wrong, correct it with a proposal; the people decide.

## Rules

1. **Pull first.** The latest commit on `origin/main` is the source of truth.
   Never push an older copy of a teammate's text over a newer one.
2. **Claim before editing** (table below), release after pushing. One claim
   per section or file at a time. Claims older than 2 hours without a push
   may be cleared by the referee.
3. **Own sections only.** Anything else goes through a proposal.
4. **Two pages.** Only the references may go on page 3. Recompile
   `report/main.pdf` after every report change and write the page count in
   the log. Every addition needs a cut of the same length (budget below).
5. **Evidence.** Every number in the report comes from a file in
   `results/`, named in the log entry. Every model is evaluated with the
   nested 5x3 CV, seed 0, of `tune_models.py`; other numbers are not
   comparable and are not reported.
6. **Commit messages** describe the change only, with no tool or agent
   attribution.

## Referee (agent P3)

The referee keeps this file consistent and decides proposals. It:

- answers each OPEN proposal with ACCEPTED, REJECTED or NEEDS-INFO and a
  one-line reason, judged on: evidence in `results/`, page budget, section
  ownership, protocol consistency, and whether tests still pass
  (`python -m pytest tests`);
- resolves conflicting claims (first claim in the pushed file wins) and
  merge conflicts in this file;
- does not edit another owner's section itself: an accepted proposal is
  carried out by the agent that made it, or by the owner;
- defers to the people: a decision can be overruled by the owner of the
  section concerned, and disagreements between owners go to the people,
  marked ESCALATED.

## Claims

| Agent | Section or file | Since | Purpose |
|---|---|---|---|
| _none_ | | | |

## Proposals

Needed for: another owner's section, shared files, anything that adds report
length, any change of a reported number. Format:

```
### PR-NNN [OPEN] title
- From: P? -> owner: P?    Date:
- Change: exact old text -> new text, or files and functions
- Evidence: results file / test output
- Page cost: +/- lines
- Decision (referee): ACCEPTED | REJECTED | NEEDS-INFO | ESCALATED, reason
```

_No proposals yet._

## Page budget

Last compile: `b3ef014`, 2 pages + references on page 3, **0 lines free**.
Recorded cuts and additions:

| Commit | Agent | Lines |
|---|---|---|
| `60b60c0` | P3 | about -18 (abstract, data description, linear sections) |

## Current state

- Submitted predictions: `python run.py` with `config/boosted_v2.json`;
  AIcrowd F1 0.454 (blend), 0.450 (final model alone).
- Linear rows of Table II: `results/tuning_{mean_squared_error_gd,mean_squared_error_sgd,least_squares,ridge_regression}.json`
  (nested 5x3 CV, `config/preprocessing_brfss.json`, seed 0).
- `results/tuning_linear.json` is the old mock-preprocessing run; never cite it.

## Open items

- [ ] P3: Reproducibility, replace `\pending{public GitHub URL and commit}`
      with the repo URL and final commit hash (last step).
- [ ] P1: check the data-description and imputation paragraphs written on
      Person 1's behalf (`87078dd`, shortened in `60b60c0`).

## Log

Newest first: `date | agent | commit | change | evidence | page count`.

- 2026-10-30 | P3 | this commit | coordination file renamed, referee process | - | -
- 2026-10-29 | P1 | `b3ef014` | recompiled report | - | 2 + refs
- 2026-10-29 | P3 | `60b60c0` | shortened abstract, data description, linear sections | - | not compiled
- 2026-10-29 | P3 | `87078dd` | data description, imputation/scaling (Person 1 section) | data cache, `preprocess.py` | not compiled
- 2026-10-29 | P1 | `09f13b7` | tree blend, AIcrowd scores 0.450 / 0.454 | `results/boosted_v2_submission.json` | -
- 2026-10-29 | P3 | `34e0f71` | linear nested CV, linear subsection, Table II rows | `results/tuning_*` (linear) | -
