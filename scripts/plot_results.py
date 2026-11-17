"""Figures and tables for the report from results/results_<split>.csv.

    python scripts/plot_results.py [--split test]
Outputs (results/figures/): map_vs_eps.png, recall_by_class.png, recall_by_scale.png, latency.png,
and results/summary_<split>.csv / .tex (mean +- std over PGD seeds).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from uavadv.config import VISDRONE_NAMES, base_parser, load_config  # noqa: E402


def label(r):
    s = f"{r.model} | {r.attack}"
    if r.defense != "none":
        s += f" + {r.defense}" + (" (adaptive)" if r.adaptive else "")
    return s


def main():
    p = base_parser(__doc__ or "")
    p.add_argument("--split", default="test")
    args = p.parse_args()
    cfg = load_config(args)
    out = Path(cfg["paths"]["out_dir"])
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(out / f"results_{args.split}.csv")
    df["adaptive"] = df["adaptive"].astype(bool)
    df["series"] = df.apply(label, axis=1)

    keys = ["model", "attack", "defense", "adaptive", "eps255"]
    metrics = ["map50", "map50_95", "op_recall", "hidden_rate", "delta_map50_vs_clean_same_defense"]
    agg = df.groupby(keys, dropna=False)[[m for m in metrics if m in df]].agg(["mean", "std", "count"]).reset_index()
    agg.columns = ["_".join(c).rstrip("_") for c in agg.columns]
    agg.to_csv(out / f"summary_{args.split}.csv", index=False)
    agg.to_latex(out / f"summary_{args.split}.tex", index=False, float_format="%.3f")

    # 1) mAP vs epsilon -- clean value plotted at eps=0 for every series of the same model/defense
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, m in zip(axes, ["map50", "map50_95"]):
        for (model, attack, defense, adaptive), g in df[df.attack != "none"].groupby(["model", "attack", "defense", "adaptive"]):
            clean = df[(df.model == model) & (df.attack == "none") & (df.defense == defense)][m].mean()
            s = g.groupby("eps255")[m].agg(["mean", "std"]).reset_index()
            x = np.r_[0, s.eps255]
            y = np.r_[clean, s["mean"]]
            err = np.r_[0, s["std"].fillna(0)]
            ax.errorbar(x, y, yerr=err, marker="o", capsize=3, label=label(g.iloc[0]))
        ax.set_xlabel("L∞ budget ε (×1/255)")
        ax.set_ylabel(m.replace("map", "mAP@").replace("50_95", "0.5:0.95").replace("50", "0.5"))
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figs / "map_vs_eps.png", dpi=200)

    # 2) per-class / per-scale operating-point recall at the largest eps, PGD, no defense vs defense
    eps_max = df.eps255.max()
    sel = df[(df.attack.isin(["none", "pgd"])) & (df.eps255.isin([0, eps_max]))]
    for prefix, cols, fname in [("opR_", [f"opR_{n}" for n in VISDRONE_NAMES.values()], "recall_by_class.png"),
                                ("opR_", ["opR_small", "opR_medium", "opR_large"], "recall_by_scale.png")]:
        cols = [c for c in cols if c in sel]
        tab = sel.groupby("series")[cols].mean().T
        ax = tab.plot.bar(figsize=(max(8, len(cols) * 0.9), 4))
        ax.set_ylabel(f"recall (conf ≥ {cfg['eval']['match_conf']}, IoU ≥ {cfg['eval']['match_iou']})")
        ax.set_xticklabels([c.removeprefix(prefix) for c in cols], rotation=30, ha="right")
        ax.set_title(f"clean vs PGD ε={eps_max}/255")
        ax.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(figs / fname, dpi=200)
        plt.close()

    # 3) latency
    lat = df.dropna(subset=["latency_ms_median"]) if "latency_ms_median" in df else pd.DataFrame()
    if len(lat):
        ax = lat.plot.bar(x="series", y="latency_ms_median", legend=False, figsize=(6, 3.5))
        ax.set_ylabel("median ms / image (batch 1)")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(figs / "latency.png", dpi=200)
    print(f"[plot] wrote {figs} and summary_{args.split}.csv/.tex")


if __name__ == "__main__":
    main()
