"""Staged search for tree-heavy blends, with paired evaluation on fixed folds.

Folds: five stratified outer folds (seed 0, as in the report). In outer fold k,
the other four folds are split 3:1 (stratified, seed 1) into inner training and
inner validation rows. Every model is fitted on the inner training rows only;
its probabilities on the inner validation rows and on outer fold k are cached in
results/boosting_search/fold<k>/. Tree counts, blend weights and thresholds are
selected on inner validation rows; outer fold k only scores the selection.

  train    fit models on the given folds and cache their probabilities
  search   inner-validation table on one fold; never reads outer-fold scores
  confirm  select on inner validation, score the outer folds, paired deltas
  final    select on pooled inner validation, fit on all rows, write a submission

Model names: "logistic" is the configured interaction logistic regression.
Tree names start with "trees" and add "_"-separated tokens: d<depth>,
lr<learning rate>, n<trees>, b<bins>, leaf<min leaf>, l2-<penalty>, s<seed>,
pp-<variant>. Missing tokens keep the submitted values (d4, lr0.08, n300, b32,
leaf150, l2-10, s0, pp-base). Joining tree names with "+" averages their
probabilities, e.g. a seed ensemble.

Two operating-point rules: "f1" maximizes F1; "both" maximizes F1 among points
whose accuracy is at least the logistic model's accuracy at its own F1 optimum
on the same rows.
"""

import argparse
import hashlib
import json
import multiprocessing
from pathlib import Path
import platform
import time

import numpy as np

from boosting import HistogramBinner, HistogramGradientBoosting, validate_boosted_config
from cross_validation import stratified_k_fold
from data import DATA_PATH, ROOT, load_data
from interactions import BEST_CONFIG, load_best_config, log_loss, model_from_config
from make_preprocessing_config import NOMINAL_DROPPED, column_names
from preprocess import Preprocessor, binary_labels
from run import generate_submission

CACHE = ROOT / "results" / "boosting_search"
OUTER_SEED, INNER_SEED = 0, 1
GRID = np.round(np.arange(0.150, 0.3501, 0.002), 3)
WEIGHTS = (0.5, 0.65, 0.8, 0.9, 1.0)
CURRENT = "trees"  # Tree part of the submitted blend, which uses weight 0.5.
CURRENT_WEIGHT = 0.5
DEFAULTS = {"d": 4, "lr": 0.08, "n": 300, "b": 32, "leaf": 150, "l2": 10.0, "s": 0, "pp": "base"}
# Raw answers kept with their own codes ("refused" weight, per-week versus
# per-month reporting). Recodes of kept variables (age groups, race variants,
# HTIN4) and MMYYYY dates, which are not ordered in time, stay dropped.
RAW_MEASURES = ["WEIGHT2", "HEIGHT3", "ALCDAY5", "FRUITJU1", "FRUIT1", "FVBEANS", "FVGREEN",
                "FVORANG", "VEGETAB1", "EXEROFT1", "EXEROFT2", "EXERHMM1", "EXERHMM2",
                "STRENGTH", "BLDSUGAR", "FEETCHK2", "LONGWTCH"]
CODE_INDICATOR_MIN = 300
VARIANTS = ("raw", "nominal", "codes")


def shown(path):
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def parse_tree(name):
    """Return (canonical name, spec) for a tree model name."""
    tokens = name.split("_")
    if tokens[0] != "trees":
        raise ValueError(f"Tree model names start with 'trees': {name}")
    spec = dict(DEFAULTS)
    for token in tokens[1:]:
        for key, cast in (("l2-", float), ("leaf", int), ("lr", float), ("pp-", str),
                          ("d", int), ("n", int), ("b", int), ("s", int)):
            if token.startswith(key):
                spec[key.rstrip("-")] = cast(token[len(key):])
                break
        else:
            raise ValueError(f"Unknown token {token!r} in {name}")
    if any(part not in VARIANTS for part in spec["pp"].split(".") if spec["pp"] != "base"):
        raise ValueError(f"Unknown preprocessing variant in {name}")
    canonical = "_".join(["trees"] + [
        f"{key}-{spec[key]:g}" if key == "l2" else f"pp-{spec[key]}" if key == "pp" else
        f"{key}{spec[key]:g}" if key == "lr" else f"{key}{spec[key]}"
        for key in DEFAULTS if spec[key] != DEFAULTS[key]])
    return canonical, spec


def tree_params(spec):
    return {"n_estimators": spec["n"], "max_depth": spec["d"], "learning_rate": spec["lr"],
            "max_bins": spec["b"], "min_samples_leaf": spec["leaf"], "l2": spec["l2"],
            "subsample": 0.7, "feature_fraction": 0.8, "seed": spec["s"]}


def checkpoints(n):
    step = 100 if n <= 300 else 300
    return sorted(set(range(step, n + 1, step)) | {n})


def variant_settings(base, variant, names):
    """Binner preprocessing and options for a dotted combination of variants."""
    settings = json.loads(json.dumps(base))
    binning = {}
    index = {name: j for j, name in enumerate(names)}
    parts = set() if variant == "base" else set(variant.split("."))
    restore = set()
    if "raw" in parts:
        restore |= {index[n] for n in RAW_MEASURES}
    if "nominal" in parts:
        nominal = [index[n] for n in NOMINAL_DROPPED]
        restore |= set(nominal)
        settings["categorical_columns"] = settings.get("categorical_columns", []) + nominal
    if "codes" in parts:
        binning["code_indicator_min"] = CODE_INDICATOR_MIN
    settings["drop_columns"] = [c for c in settings.get("drop_columns", []) if c not in restore]
    return settings, binning


def fold_rows(y, fold):
    """(inner training, inner validation, outer test) row indices for a fold."""
    outer = stratified_k_fold(y, 5, OUTER_SEED)
    test = outer[fold]
    rest = np.concatenate([part for k, part in enumerate(outer) if k != fold])
    rest.sort()
    inner = stratified_k_fold(y[rest], 4, INNER_SEED)
    return rest[np.concatenate(inner[1:])], rest[inner[0]], test


# --------------------------------------------------------------------------- training

_JOBS = {}  # Filled before forking, so workers share the bins instead of pickling them.


def _fit_tree(key):
    fold, name, path = key
    _, spec = parse_tree(name)
    y_train, train, val, test = _JOBS[key]
    started = time.perf_counter()
    model = HistogramGradientBoosting(**tree_params(spec)).fit(y_train, train)
    stages = checkpoints(spec["n"])
    val_scores = dict(model.staged_predict_proba(val, stages))
    test_scores = dict(model.staged_predict_proba(test, stages))
    elapsed = time.perf_counter() - started
    np.savez_compressed(path, checkpoints=np.array(stages),
                        val=np.array([val_scores[c] for c in stages], dtype=np.float32),
                        test=np.array([test_scores[c] for c in stages], dtype=np.float32))
    save_json(path.with_suffix(".json"), {
        "model": name, "fold": fold, "params": tree_params(spec), "variant": spec["pp"],
        "features": int(train.shape[1]), "training_loss": model.loss_, "elapsed_s": elapsed})
    print(f"fold {fold} {name}: {elapsed / 60:.1f} min, training loss {model.loss_:.5f}", flush=True)
    return name


def train(args):
    X, _, y, _, _ = load_data(args.data_dir)
    y = binary_labels(y)
    names = column_names(args.data_dir)
    base = json.loads(args.preprocessing.read_text(encoding="utf-8"))
    logistic_config = load_best_config(args.logistic_config)
    models = args.models.split(",")
    for fold in args.folds:
        inner_train, val, test = fold_rows(y, fold)
        directory = CACHE / f"fold{fold}"
        directory.mkdir(parents=True, exist_ok=True)
        rows = directory / "rows.npz"
        if not rows.exists():
            np.savez_compressed(rows, train=inner_train, val=val, test=test,
                                y_val=y[val].astype(np.int8), y_test=y[test].astype(np.int8))
        if "logistic" in models and not (directory / "logistic.npz").exists():
            started = time.perf_counter()
            processor = Preprocessor(**logistic_config["preprocessing"])
            model = model_from_config(logistic_config).fit(y[inner_train],
                                                           processor.fit_transform(X[inner_train]))
            np.savez_compressed(directory / "logistic.npz", checkpoints=np.array([0]),
                                val=model.predict_proba(processor.transform(X[val]))[None].astype(np.float32),
                                test=model.predict_proba(processor.transform(X[test]))[None].astype(np.float32))
            save_json(directory / "logistic.json", {
                "model": "logistic", "fold": fold, "config": logistic_config,
                "training_loss": model.loss_, "elapsed_s": time.perf_counter() - started})
            print(f"fold {fold} logistic: training loss {model.loss_:.5f}", flush=True)
            del model, processor
    bins = {}
    for fold in args.folds:
        inner_train, val, test = fold_rows(y, fold)
        for name in models:
            if name == "logistic":
                continue
            canonical, spec = parse_tree(name)
            path = CACHE / f"fold{fold}" / f"{canonical}.npz"
            if path.exists():
                continue
            binning_key = (fold, spec["pp"], spec["b"], spec["s"])
            if binning_key not in bins:
                settings, binning = variant_settings(base, spec["pp"], names)
                binner = HistogramBinner(settings, spec["b"], spec["s"], **binning)
                bins[binning_key] = (binner.fit_transform(X[inner_train]),
                                     binner.transform(X[val]), binner.transform(X[test]))
            _JOBS[(fold, canonical, path)] = (y[inner_train], *bins[binning_key])
    del X
    if not _JOBS:
        print("All requested models are cached.")
        return
    # Longest jobs first, so the pool finishes together.
    keys = sorted(_JOBS, key=lambda k: -parse_tree(k[1])[1]["n"] * 2 ** parse_tree(k[1])[1]["d"])
    print(f"Training {len(keys)} tree models with {args.workers} workers.", flush=True)
    with multiprocessing.get_context("fork").Pool(min(args.workers, len(keys))) as pool:
        for _ in pool.imap_unordered(_fit_tree, keys):
            pass


# --------------------------------------------------------------------------- analysis

def load_scores(fold, name):
    """(checkpoints, inner-validation scores, outer-test scores); '+' averages members."""
    parts = []
    for member in name.split("+"):
        canonical = member if member == "logistic" else parse_tree(member)[0]
        with np.load(CACHE / f"fold{fold}" / f"{canonical}.npz") as cached:
            parts.append((cached["checkpoints"], cached["val"].astype(float),
                          cached["test"].astype(float)))
    stages = parts[0][0]
    if any(not np.array_equal(stages, p[0]) for p in parts):
        raise ValueError(f"Ensemble members need equal checkpoints: {name}")
    return (stages, np.mean([p[1] for p in parts], axis=0), np.mean([p[2] for p in parts], axis=0))


def fold_labels(fold):
    with np.load(CACHE / f"fold{fold}" / "rows.npz") as rows:
        return rows["y_val"].astype(float), rows["y_test"].astype(float)


def confusion_grid(y, scores, thresholds=GRID):
    """Rows of (tp, fp, fn, tn) for predictions scores >= threshold."""
    order = np.argsort(scores, kind="stable")
    positives = np.r_[0, np.cumsum(y[order])].astype(np.int64)
    starts = np.searchsorted(scores[order], thresholds, side="left")
    predicted = len(y) - starts
    tp = positives[-1] - positives[starts]
    fp = predicted - tp
    fn = positives[-1] - tp
    return np.stack((tp, fp, fn, len(y) - tp - fp - fn), axis=1)


def point(cm, **extra):
    tp, fp, fn, tn = (int(v) for v in cm)
    return dict(extra, f1=2 * tp / max(2 * tp + fp + fn, 1), accuracy=(tp + tn) / (tp + fp + fn + tn),
                precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1),
                confusion={"tp": tp, "fp": fp, "fn": fn, "tn": tn})


def f1s(cm):
    return 2 * cm[:, 0] / np.maximum(2 * cm[:, 0] + cm[:, 1] + cm[:, 2], 1)


def candidate_grids(y, model, logistic, stages, weights):
    """Yield (checkpoint, weight, blended scores, confusion grid)."""
    for c, scores in zip(stages, model):
        for w in weights:
            blended = w * scores + (1 - w) * logistic
            yield int(c), float(w), blended, confusion_grid(y, blended)


def select(grids, floor=None):
    """Best F1 over (checkpoint, weight, threshold), optionally with a correct-count floor.

    Ties keep the first candidate and the lowest threshold.
    """
    best = None
    for c, w, _, cm in grids:
        allowed = np.ones(len(GRID), bool) if floor is None else cm[:, 0] + cm[:, 3] >= floor
        if not allowed.any():
            continue
        f1 = np.where(allowed, f1s(cm), -1)
        i = int(np.argmax(f1))
        if best is None or f1[i] > best[0]:
            best = (f1[i], c, w, float(GRID[i]))
    return None if best is None else dict(zip(("f1", "checkpoint", "weight", "threshold"), best))


def evaluate(y, scores, threshold, **extra):
    cm = confusion_grid(y, scores, np.array([threshold]))[0]
    return point(cm, threshold=threshold, log_loss=log_loss(y, np.clip(scores, 0, 1)), **extra)


def logistic_reference(y, logistic):
    """Logistic operating point (F1 optimum) and its number of correct predictions."""
    cm = confusion_grid(y, logistic)
    i = int(np.argmax(f1s(cm)))
    return float(GRID[i]), int(cm[i, 0] + cm[i, 3])


def pareto(rows):
    """Rows not dominated in (F1, accuracy): no other row is at least as good in both and better in one."""
    rows = sorted(rows, key=lambda r: (-r["f1"], -r["accuracy"]))
    frontier, best_accuracy = [], -1
    for row in rows:
        if row["accuracy"] > best_accuracy:
            frontier.append(row)
            best_accuracy = row["accuracy"]
    return frontier


def search(args):
    fold = args.fold
    y_val, _ = fold_labels(fold)
    _, logistic, _ = load_scores(fold, "logistic")
    logistic = logistic[0]
    logistic_threshold, floor = logistic_reference(y_val, logistic)
    stages, current, _ = load_scores(fold, CURRENT)
    current_scores = CURRENT_WEIGHT * current[list(stages).index(300)] + (1 - CURRENT_WEIGHT) * logistic
    current_cm = confusion_grid(y_val, current_scores)
    current_i = int(np.argmax(f1s(current_cm)))
    baselines = {
        "logistic": evaluate(y_val, logistic, logistic_threshold),
        "current_blend": evaluate(y_val, current_scores, float(GRID[current_i]),
                                  weight=CURRENT_WEIGHT, checkpoint=300)}
    logistic_cm = confusion_grid(y_val, logistic)
    rows, all_points = [], [point(logistic_cm[i], model="logistic", checkpoint=0, weight=0.0,
                                  threshold=float(GRID[i])) for i in range(len(GRID))]
    for name in args.models.split(","):
        stages, model, _ = load_scores(fold, name)
        for c, w, blended, cm in candidate_grids(y_val, model, logistic, stages, WEIGHTS):
            f1 = f1s(cm)
            correct = cm[:, 0] + cm[:, 3]
            entry = {"model": name, "checkpoint": c, "weight": w,
                     "log_loss": log_loss(y_val, blended),
                     "f1_rule": point(cm[int(np.argmax(f1))], threshold=float(GRID[np.argmax(f1)]))}
            allowed = correct >= floor
            if allowed.any():
                i = int(np.argmax(np.where(allowed, f1, -1)))
                entry["both_rule"] = point(cm[i], threshold=float(GRID[i]))
            entry["beats_current_f1_and_logistic_accuracy"] = bool(
                "both_rule" in entry and entry["both_rule"]["f1"] > baselines["current_blend"]["f1"])
            rows.append(entry)
            all_points += [point(cm[i], model=name, checkpoint=c, weight=w, threshold=float(GRID[i]))
                           for i in range(len(GRID))]
    ranked = sorted(rows, key=lambda r: -r.get("both_rule", {"f1": -1})["f1"])
    result = {"protocol": "inner validation of one outer fold; outer-fold scores are not read",
              "fold": fold, "rows": len(y_val), "thresholds": GRID.tolist(), "weights": list(WEIGHTS),
              "logistic_correct_floor": floor, "baselines": baselines, "candidates": ranked,
              "pareto_frontier": pareto(all_points),
              "environment": {"python": platform.python_version(), "numpy": np.__version__}}
    save_json(args.output, result)
    print(f"Fold {fold} inner validation, {len(y_val)} rows. Baselines:")
    for key, value in baselines.items():
        print(f"  {key:14s} F1 {value['f1']:.4f}  acc {value['accuracy']:.4f}  thr {value['threshold']:.3f}"
              f"  logloss {value['log_loss']:.5f}")
    print(f"{'model':48s} {'ckpt':>4s} {'w':>4s} {'logloss':>8s} | {'F1-rule F1':>10s} {'acc':>6s} {'thr':>5s}"
          f" | {'both F1':>7s} {'acc':>6s} {'thr':>5s}")
    for r in sorted(rows, key=lambda r: (r["model"], r["checkpoint"], r["weight"])):
        b = r.get("both_rule")
        both = f"{b['f1']:7.4f} {b['accuracy']:.4f} {b['threshold']:.3f}" if b else "   none"
        star = " *" if r["beats_current_f1_and_logistic_accuracy"] else ""
        print(f"{r['model']:48s} {r['checkpoint']:4d} {r['weight']:4.2f} {r['log_loss']:8.5f} | "
              f"{r['f1_rule']['f1']:10.4f} {r['f1_rule']['accuracy']:.4f} {r['f1_rule']['threshold']:.3f} | "
              f"{both}{star}")
    print(f"Saved {args.output}")


def confirm(args):
    """Each candidate is 'model@checkpoint'; weights and thresholds are chosen per fold."""
    candidates = [c.split("@") for c in args.candidates.split(",")]
    folds, per_fold = args.folds, []
    for fold in folds:
        y_val, y_test = fold_labels(fold)
        _, (logistic_val,), (logistic_test,) = load_scores(fold, "logistic")
        logistic_threshold, floor = logistic_reference(y_val, logistic_val)
        stages, cur_val, cur_test = load_scores(fold, CURRENT)
        k = list(stages).index(300)
        current_val = CURRENT_WEIGHT * cur_val[k] + (1 - CURRENT_WEIGHT) * logistic_val
        current_threshold = select([(300, CURRENT_WEIGHT, None, confusion_grid(y_val, current_val))])["threshold"]
        current_test = CURRENT_WEIGHT * cur_test[k] + (1 - CURRENT_WEIGHT) * logistic_test
        row = {"fold": fold, "logistic": evaluate(y_test, logistic_test, logistic_threshold),
               "current_blend": evaluate(y_test, current_test, current_threshold,
                                         weight=CURRENT_WEIGHT, checkpoint=300), "candidates": {}}
        for name, checkpoint in candidates:
            stages, model_val, model_test = load_scores(fold, name)
            k = list(stages).index(int(checkpoint))
            grids = list(candidate_grids(y_val, model_val[k:k + 1], logistic_val, stages[k:k + 1], WEIGHTS))
            for rule, rule_floor in (("f1", None), ("both", floor)):
                chosen = select(grids, rule_floor)
                key = f"{name}@{checkpoint}:{rule}"
                if chosen is None:
                    row["candidates"][key] = None
                    continue
                w = chosen["weight"]
                scores = w * model_test[k] + (1 - w) * logistic_test
                row["candidates"][key] = evaluate(y_test, scores, chosen["threshold"], weight=w,
                                                  checkpoint=int(checkpoint),
                                                  validation_f1=chosen["f1"])
        per_fold.append(row)
    summary = {}
    for key in per_fold[0]["candidates"]:
        if any(r["candidates"][key] is None for r in per_fold):
            summary[key] = None
            continue
        entry = {}
        for metric in ("f1", "accuracy", "precision", "recall"):
            values = np.array([r["candidates"][key][metric] for r in per_fold])
            entry[metric] = {"mean": values.mean(), "std": values.std()}
            for base in ("logistic", "current_blend"):
                delta = values - np.array([r[base][metric] for r in per_fold])
                entry[metric][f"delta_vs_{base}"] = {"mean": delta.mean(), "std": delta.std(),
                                                     "positive_folds": int(np.sum(delta > 0)),
                                                     "per_fold": delta.tolist()}
        entry["weights"] = [r["candidates"][key]["weight"] for r in per_fold]
        entry["thresholds"] = [r["candidates"][key]["threshold"] for r in per_fold]
        summary[key] = entry
    baselines = {base: {metric: {"mean": float(np.mean([r[base][metric] for r in per_fold])),
                                 "std": float(np.std([r[base][metric] for r in per_fold]))}
                        for metric in ("f1", "accuracy", "precision", "recall")}
                 for base in ("logistic", "current_blend")}
    result = {"protocol": "select on inner validation of each outer fold; score the outer fold",
              "outer_seed": OUTER_SEED, "inner_seed": INNER_SEED, "folds": folds,
              "thresholds": GRID.tolist(), "weights": list(WEIGHTS),
              "baselines": baselines, "summary": summary, "per_fold": per_fold,
              "environment": {"python": platform.python_version(), "numpy": np.__version__}}
    save_json(args.output, result)
    print(f"Outer folds {folds}; mean over folds (paired deltas, positive folds).")
    for base, value in baselines.items():
        print(f"  {base:52s} F1 {value['f1']['mean']:.4f}  acc {value['accuracy']['mean']:.4f}")
    for key, entry in summary.items():
        if entry is None:
            print(f"  {key:52s} no feasible point in some fold")
            continue
        f, a = entry["f1"], entry["accuracy"]
        print(f"  {key:52s} F1 {f['mean']:.4f} (vs blend {f['delta_vs_current_blend']['mean']:+.4f},"
              f" {f['delta_vs_current_blend']['positive_folds']}/{len(folds)})"
              f"  acc {a['mean']:.4f} (vs logistic {a['delta_vs_logistic']['mean']:+.4f},"
              f" {a['delta_vs_logistic']['positive_folds']}/{len(folds)};"
              f" vs blend {a['delta_vs_current_blend']['mean']:+.4f},"
              f" {a['delta_vs_current_blend']['positive_folds']}/{len(folds)})")
    print(f"Saved {args.output}")


def final(args):
    """Choose weight and threshold on pooled inner validation, then fit on all rows."""
    name, checkpoint = args.candidate.split("@")
    checkpoint = int(checkpoint)
    y_parts, logistic_parts, model_parts = [], [], []
    for fold in range(5):
        y_val, _ = fold_labels(fold)
        stages, model_val, _ = load_scores(fold, name)
        y_parts.append(y_val)
        logistic_parts.append(load_scores(fold, "logistic")[1][0])
        model_parts.append(model_val[list(stages).index(checkpoint)])
    y_val, logistic_val, model_val = map(np.concatenate, (y_parts, logistic_parts, model_parts))
    _, floor = logistic_reference(y_val, logistic_val)
    grids = list(candidate_grids(y_val, [model_val], logistic_val, [checkpoint], WEIGHTS))
    chosen = select(grids, floor if args.rule == "both" else None)
    if chosen is None:
        raise ValueError("No pooled operating point satisfies the rule.")
    specs = [parse_tree(member)[1] for member in name.split("+")]
    shared = {k: v for k, v in specs[0].items() if k != "s"}
    if any({k: v for k, v in s.items() if k != "s"} != shared for s in specs):
        raise ValueError("Ensemble members must differ only in their seed.")
    base = json.loads(args.preprocessing.read_text(encoding="utf-8"))
    settings, binning = variant_settings(base, shared["pp"], column_names(args.data_dir))
    params = tree_params(dict(specs[0], n=checkpoint))
    config = validate_boosted_config({
        "model": "histogram_boosted_blend", "params": params, "seeds": [s["s"] for s in specs],
        "preprocessing": settings, "binning": binning, "tree_weight": chosen["weight"],
        "threshold": chosen["threshold"], "logistic_config": load_best_config(args.logistic_config),
        "status": f"selected by search_boosting.py ({args.rule} rule, pooled inner validation)"})
    save_json(args.config_output, config)
    print(f"Pooled inner validation selection: {chosen}; saved {args.config_output}", flush=True)
    started = time.perf_counter()
    predictions = generate_submission(config, args.data_dir, args.output, workers=args.workers)
    written = np.genfromtxt(args.output, delimiter=",", skip_header=1)
    _, x_test, _, _, test_ids = load_data(args.data_dir)
    if (len(written) != len(x_test) or not np.array_equal(written[:, 0], test_ids) or
            not np.all(np.isin(written[:, 1], [-1, 1]))):
        raise ValueError("The submission must keep every test Id in order, with labels -1 and 1.")
    changed = {}
    for previous in ("submission_boosted.csv", "submission_rerun.csv"):
        path = ROOT / previous
        if path.exists():
            old = np.genfromtxt(path, delimiter=",", skip_header=1)
            if old.shape == written.shape and np.array_equal(old[:, 0], written[:, 0]):
                changed[previous] = int(np.sum(old[:, 1] != written[:, 1]))
    metadata = {"config_file": shown(args.config_output), "submission": shown(args.output),
                "candidate": args.candidate, "rule": args.rule, "pooled_selection": chosen,
                "pooled_validation_rows": len(y_val),
                "rows": len(predictions), "positive_predictions": int(np.sum(predictions == 1)),
                "negative_predictions": int(np.sum(predictions == -1)),
                "changed_predictions": changed,
                "csv_sha256": hashlib.sha256(Path(args.output).read_bytes()).hexdigest(),
                "elapsed_s": time.perf_counter() - started, "aicrowd_score": None}
    save_json(args.metadata_output, metadata)
    print(json.dumps(metadata, indent=2))


def main(argv=None):
    global CACHE
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DATA_PATH)
    parser.add_argument("--logistic-config", type=Path, default=BEST_CONFIG)
    parser.add_argument("--preprocessing", type=Path, default=ROOT / "config/preprocessing_brfss.json")
    parser.add_argument("--cache-dir", type=Path, default=CACHE)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("train")
    p.add_argument("--folds", type=lambda s: [int(v) for v in s.split(",")], default=[0])
    p.add_argument("--models", required=True)
    p.add_argument("--workers", type=int, default=7)
    p = sub.add_parser("search")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--models", required=True)
    p.add_argument("--output", type=Path, default=ROOT / "results/boosting_search_fold0.json")
    p = sub.add_parser("confirm")
    p.add_argument("--folds", type=lambda s: [int(v) for v in s.split(",")], default=[0, 1, 2, 3, 4])
    p.add_argument("--candidates", required=True)
    p.add_argument("--output", type=Path, default=ROOT / "results/boosting_search_confirmation.json")
    p = sub.add_parser("final")
    p.add_argument("--candidate", required=True)
    p.add_argument("--rule", choices=("f1", "both"), default="both")
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--config-output", type=Path, default=ROOT / "config/boosted_v2.json")
    p.add_argument("--output", type=Path, default=ROOT / "submission_boosted_v2.csv")
    p.add_argument("--metadata-output", type=Path, default=ROOT / "results/boosted_v2_submission.json")
    args = parser.parse_args(argv)
    CACHE = args.cache_dir
    {"train": train, "search": search, "confirm": confirm, "final": final}[args.command](args)


if __name__ == "__main__":
    main()
