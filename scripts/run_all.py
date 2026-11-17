"""Phases 2-3: every Table I comparison on the held-out TEST split, written to results/results_test.csv.

  clean / no defense                      -> baseline quality + latency
  FGSM, PGD x eps x seeds / no defense    -> attack sensitivity
  clean / chosen defense                  -> accuracy + latency cost of the defense
  FGSM, PGD / chosen defense              -> robustness recovered against a non-adaptive attacker
  PGD adaptive (BPDA / through-blur)      -> robustness against an attacker who knows the defense

    python scripts/run_all.py
    python scripts/run_all.py --weights runs/advtrain/weights/best.pt --tag advtrain   # adversarially trained model
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from uavadv.config import base_parser, load_config, run_metadata, write_json  # noqa: E402
from uavadv.pipeline import Experiment  # noqa: E402


def main():
    p = base_parser(__doc__ or "")
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--weights")
    p.add_argument("--tag")
    p.add_argument("--no-defense", action="store_true", help="skip defense rows (e.g. before calibration)")
    args = p.parse_args()
    cfg = load_config(args)
    out = Path(cfg["paths"]["out_dir"])
    a = cfg["attack"]
    exp = Experiment(cfg, args.split, args.weights, args.tag)

    defense = None
    if not args.no_defense:
        defense = cfg["defense"]["chosen"]
        chosen = out / "chosen_defense.json"
        if defense is None and chosen.exists():
            defense = json.loads(chosen.read_text())["defense"]
        if defense is None:
            print("[run_all] no chosen defense - run scripts/calibrate_defense.py first; continuing without defense")

    adv_sets = [(m, e, s, exp.attacked(m, e, s)) for m in a["attacks"] for e in a["eps"]
                for s in (a["seeds"] if m == "pgd" else [a["seeds"][0]])]

    rows = []
    for d in [None] + ([defense] if defense else []):
        rows.append(exp.evaluate(exp.clean_dir, d, with_latency=True, attack="none", eps255=0, seed=None, adaptive=False))
        for m, e, s, cond in adv_sets:
            rows.append(exp.evaluate(cond, d, attack=m, eps255=e, seed=s, adaptive=False))
    if defense and cfg["defense"]["adaptive"]:
        for e in a["eps"]:
            for s in a["seeds"]:
                cond = exp.attacked("pgd", e, s, defense=defense)
                rows.append(exp.evaluate(cond, defense, attack="pgd", eps255=e, seed=s, adaptive=True))

    df = pd.DataFrame(rows)
    clean = df[df.attack == "none"].set_index("defense")["map50"]
    df["delta_map50_vs_clean_same_defense"] = df.apply(lambda r: clean[r.defense] - r.map50, axis=1)
    path = out / f"results_{args.split}.csv"
    if path.exists():
        old = pd.read_csv(path)
        df = pd.concat([old[old.model != exp.tag], df], ignore_index=True)
    df.to_csv(path, index=False)
    write_json(out / f"run_meta_{args.split}_{exp.tag}.json", {**run_metadata(cfg), "weights": exp.weights, "defense": defense})
    print(f"[run_all] wrote {path}")


if __name__ == "__main__":
    main()
