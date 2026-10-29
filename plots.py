"""Report figures, drawn from the saved results.

tuning.pdf: heatmap of the cross-validated tuning grid of tune_models.py,
inner-CV F1 for every (gamma, lambda_) combination; lambda_ = 0 is plain
logistic regression. Selected models are outlined.
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt

from data import ROOT

INK, MUTED, SURFACE, NEUTRAL, ACCENT = "#0b0b0b", "#52514e", "#ffffff", "#e4e3df", "#eb6834"
BLUES = LinearSegmentedColormap.from_list("blues", [
    "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
F1_RANGE = (0.41, 0.432)
GAP = 0.25


def load(results, name):
    with (results / f"tuning_{name}.json").open(encoding="utf-8") as stream:
        return json.load(stream)


def tuning_figure(results, output):
    plain, regularized = load(results, "logistic_regression"), load(results, "reg_logistic_regression")
    cells = {(r["params"]["gamma"], 0.0): r["inner_f1"] for r in plain["tuning_grid"]}
    cells.update({(r["params"]["gamma"], r["params"]["lambda_"]): r["inner_f1"]
                  for r in regularized["tuning_grid"]})
    selected = {(plain["best"]["config"]["params"]["gamma"], 0.0),
                tuple(regularized["best"]["config"]["params"][k] for k in ("gamma", "lambda_"))}
    gammas = sorted({g for g, _ in cells})
    lambdas = sorted({l for _, l in cells})
    norm = Normalize(*F1_RANGE)

    fig, ax = plt.subplots(figsize=(3.4, 1.75))
    for i, gamma in enumerate(gammas):
        for j, lambda_ in enumerate(lambdas):
            x, f1 = j + (GAP if j else 0.0), cells[(gamma, lambda_)]
            inside = f1 >= F1_RANGE[0]
            ax.add_patch(Rectangle((x, i), 1, 1, facecolor=BLUES(norm(f1)) if inside else NEUTRAL,
                                   edgecolor=SURFACE, linewidth=1.2))
            dark = inside and norm(f1) > 0.55
            ax.text(x + 0.5, i + 0.5, f"{f1:.4f}"[1:] if inside else f"{f1:.2f}",
                    ha="center", va="center", fontsize=7,
                    color=SURFACE if dark else INK)
            if (gamma, lambda_) in selected:
                ax.add_patch(Rectangle((x + 0.07, i + 0.07), 0.86, 0.86, fill=False,
                                       edgecolor=ACCENT, linewidth=1.6))

    centers = [j + 0.5 + (GAP if j else 0.0) for j in range(len(lambdas))]
    ax.set_xticks(centers)
    ax.set_xticklabels(["0"] + [f"$10^{{{round(math.log10(l))}}}$" for l in lambdas[1:]])
    ax.set_yticks([i + 0.5 for i in range(len(gammas))])
    ax.set_yticklabels([f"{g:g}" for g in gammas])
    ax.set_xlim(0, len(lambdas) + GAP)
    ax.set_ylim(len(gammas), 0)
    ax.set_xlabel("$\\lambda$", fontsize=9, color=MUTED, labelpad=1)
    ax.set_ylabel("$\\gamma$", fontsize=9, color=MUTED, rotation=0, labelpad=6)
    ax.tick_params(labelsize=7.5, colors=MUTED, length=0, pad=2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(0.5, -0.25, "Logistic", ha="center", va="bottom", fontsize=8, color=INK)
    ax.text(1 + GAP + (len(lambdas) - 1) / 2, -0.25, "Regularized logistic",
            ha="center", va="bottom", fontsize=8, color=INK)

    bar = fig.colorbar(ScalarMappable(norm=norm, cmap=BLUES), ax=ax, fraction=0.04, pad=0.02)
    bar.set_ticks([0.41, 0.42, 0.43])
    bar.set_label("inner-CV F1", fontsize=8, color=MUTED)
    bar.ax.tick_params(labelsize=7, colors=MUTED, length=2)
    bar.outline.set_visible(False)

    fig.tight_layout(pad=0.3)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    print(f"Saved {output}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--figures", type=Path, default=ROOT / "report" / "figures")
    args = parser.parse_args(argv)
    tuning_figure(args.results, args.figures / "tuning.pdf")


if __name__ == "__main__":
    main()
