"""Report figures, drawn from the saved results.

tuning.pdf: cross-validated tuning grids of tune_models.py: F1 against gamma
for logistic regression, and against lambda_ at the selected gamma for
regularized logistic regression.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import ROOT

SERIES, INK, MUTED, GRID = "#2a78d6", "#0b0b0b", "#52514e", "#e4e3df"


def style(ax, title, xlabel):
    ax.set_title(title, fontsize=8, color=INK, loc="left")
    ax.set_xlabel(xlabel, fontsize=7, color=MUTED)
    ax.set_xscale("log")
    ax.tick_params(labelsize=6.5, colors=MUTED, length=2)
    ax.minorticks_off()
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
        ax.spines[side].set_linewidth(0.6)


def draw(ax, rows, x_of, selected):
    rows = sorted(rows, key=x_of)
    xs = [x_of(r) for r in rows]
    ax.errorbar(xs, [r["inner_f1"] for r in rows], yerr=[r["inner_fold_std"] for r in rows],
                color=SERIES, linewidth=1.5, elinewidth=0.8, capsize=2, zorder=2)
    for x, r in zip(xs, rows):
        ax.plot(x, r["inner_f1"], "o", markersize=4.5, markeredgewidth=1.2,
                markeredgecolor=SERIES, zorder=3,
                markerfacecolor=SERIES if r["params"] == selected else "white")


def tuning_figure(results, output):
    def load(name):
        with (results / f"tuning_{name}.json").open(encoding="utf-8") as stream:
            return json.load(stream)

    plain, regularized = load("logistic_regression"), load("reg_logistic_regression")
    best_plain = plain["best"]["config"]["params"]
    best_reg = regularized["best"]["config"]["params"]
    zero = min(r["params"]["lambda_"] for r in regularized["tuning_grid"]) / 10
    at_gamma = [dict(r, params=dict(r["params"], lambda_=0.0)) for r in plain["tuning_grid"]
                if r["params"]["gamma"] == best_reg["gamma"]]
    at_gamma += [r for r in regularized["tuning_grid"] if r["params"]["gamma"] == best_reg["gamma"]]

    fig, (left, right) = plt.subplots(1, 2, figsize=(3.4, 1.8), sharey=True)
    draw(left, plain["tuning_grid"], lambda r: r["params"]["gamma"], best_plain)
    style(left, "Logistic: F1 vs $\\gamma$", "$\\gamma$")
    left.set_ylabel("inner-CV F1", fontsize=7, color=MUTED)
    draw(right, at_gamma, lambda r: r["params"]["lambda_"] or zero, best_reg)
    style(right, f"Reg. logistic: F1 vs $\\lambda$ ($\\gamma={best_reg['gamma']:g}$)", "$\\lambda$")
    right.set_xticks([zero] + [10.0 ** e for e in range(-5, 0, 2)])
    right.set_xticklabels(["0"] + [f"$10^{{{e}}}$" for e in range(-5, 0, 2)])
    fig.tight_layout(pad=0.4, w_pad=0.8)
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
