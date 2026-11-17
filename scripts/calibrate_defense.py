"""Phase 3a: choose the preprocessing defense on the VALIDATION split (never on test).

For every candidate in defense.candidates: clean mAP and mAP under PGD at defense.calibration_eps
(generated against the undefended model). Picks the candidate with the highest attacked mAP@0.5
among those whose clean mAP@0.5 drop is <= defense.max_clean_drop.
Writes results/defense_calibration.csv and results/chosen_defense.json.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from uavadv import defenses  # noqa: E402
from uavadv.config import base_parser, load_config, run_metadata, write_json  # noqa: E402
from uavadv.pipeline import Experiment  # noqa: E402


def main():
    cfg = load_config(base_parser(__doc__ or "").parse_args())
    out = Path(cfg["paths"]["out_dir"])
    exp = Experiment(cfg, "val")
    eps = cfg["defense"]["calibration_eps"]
    adv = exp.attacked("pgd", eps, cfg["attack"]["seeds"][0])
    rows = []
    for d in [None] + cfg["defense"]["candidates"]:
        clean = exp.evaluate(exp.clean_dir, d)
        attacked = exp.evaluate(adv, d)
        rows.append({"defense": defenses.name_of(d), "spec": json.dumps(d), "clean_map50": clean["map50"],
                     "clean_map50_95": clean["map50_95"], f"pgd{eps}_map50": attacked["map50"],
                     f"pgd{eps}_map50_95": attacked["map50_95"]})
    df = pd.DataFrame(rows)
    base = df.loc[df.defense == "none", "clean_map50"].item()
    df["clean_drop"] = base - df["clean_map50"]
    df["eligible"] = (df["clean_drop"] <= cfg["defense"]["max_clean_drop"]) & (df.defense != "none")
    df.to_csv(out / "defense_calibration.csv", index=False)
    print(df.to_string(index=False))
    if not df.eligible.any():
        print("[calibrate] no candidate meets max_clean_drop; relax it or add milder candidates")
        return
    best = df[df.eligible].sort_values(f"pgd{eps}_map50", ascending=False).iloc[0]
    write_json(out / "chosen_defense.json", {"defense": json.loads(best.spec), "selected_on": "val",
                                             "criterion": f"max PGD eps={eps} mAP50 s.t. clean drop <= {cfg['defense']['max_clean_drop']}",
                                             "meta": run_metadata(cfg)})
    print(f"[calibrate] chosen: {best.defense}")


if __name__ == "__main__":
    main()
