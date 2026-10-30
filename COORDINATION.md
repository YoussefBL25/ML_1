# Coordination notes

Shared notes for everyone (people and coding agents) editing this repository,
especially `report/main.tex`. Read this file before editing, and add an entry
to the log at the bottom after each push. Delete this file before the final
submission.

## Rules

1. `git pull` before editing. The latest commit on `origin/main` is the source
   of truth; never overwrite a teammate's newer text with an older copy.
2. Edit only the sections whose `OWNER` banner in `report/main.tex` is yours.
   To change someone else's section, write the proposal under "Open
   questions" below instead.
3. The report must stay within **2 pages**; only the references may go on
   page 3. Recompile `report/main.pdf` after every report change and record
   the page count in the log. Anything added must be paid for by a cut.
4. Every number in the report must come from a file in `results/`; name the
   file in the log entry.
5. Commit messages describe the change only; no tool or agent attribution.

## Owners

| Person | Who | Report sections |
|---|---|---|
| 1 | Hassen Ben Chaabane | Data description, imputation and scaling; boosted trees |
| 2 | Meriem Laataoui | Codebook rules, protocol, logistic models, final model, results |
| 3 | Youssef Ben Lassoued | Abstract, introduction, linear models (+ linear rows of Table II), conclusion, reproducibility |

## Current state (2026-10-29, after `b3ef014`)

- `report/main.pdf`: 2 pages + references on page 3. **No space left.**
- Linear rows of Table II: `results/tuning_{mean_squared_error_gd,mean_squared_error_sgd,least_squares,ridge_regression}.json`
  (nested 5x3 CV, `config/preprocessing_brfss.json`, seed 0).
- Submitted predictions: `python run.py` with `config/boosted_v2.json`.
- `results/tuning_linear.json` is the old mock-preprocessing run; do not cite it.

## Open items

- [ ] Reproducibility: replace `\pending{public GitHub URL and commit}` with
      the repo URL and final commit hash (last step, Person 3).
- [ ] Person 1: check the data-description and imputation paragraphs written
      on their behalf.

## Open questions

_None._ Add: `- [date] [from] -> [to]: question`.

## Log

Newest first: `date | person | commit | what changed | page count`.

- 2026-10-29 | Person 1 | `b3ef014` | recompiled report | 2 + refs
- 2026-10-29 | Person 3 | `60b60c0` | shortened abstract, data description, linear sections | not compiled
- 2026-10-29 | Person 3 | `87078dd` | data description and imputation/scaling text (Person 1 section) | not compiled
- 2026-10-29 | Person 1 | `09f13b7` | tree blend, corrected AIcrowd scores (0.450 / 0.454) | -
- 2026-10-29 | Person 3 | `34e0f71` | linear nested CV results, linear subsection, Table II rows | -
