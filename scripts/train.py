"""Phase 1b / 3b: fine-tune the YOLO baseline on VisDrone, or adversarially train it.

    python scripts/train.py                                    # clean baseline -> runs/baseline/weights/best.pt
    python scripts/train.py --set adv_train.enabled=true       # adversarial training -> runs/advtrain/...
Model selection uses the VisDrone *val* split only; the test split is never touched here.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultralytics import YOLO  # noqa: E402

from uavadv.advtrain import make_adv_trainer  # noqa: E402
from uavadv.config import base_parser, load_config, resolve_device, run_metadata, write_json  # noqa: E402


def main():
    p = base_parser(__doc__)
    p.add_argument("--init", help="override starting weights (default model.init_weights; adv training defaults to the baseline)")
    args = p.parse_args()
    cfg = load_config(args)
    t, a = cfg["train"], cfg["adv_train"]
    data_yaml = t["data"] or Path(cfg["paths"]["data_dir"]) / "visdrone.yaml"
    kw = dict(data=str(data_yaml), imgsz=t["imgsz"], epochs=t["epochs"], batch=t["batch"], patience=t["patience"],
              workers=t["workers"], seed=cfg["seed"], deterministic=True, device=resolve_device(cfg["device"]),
              project=cfg["paths"]["runs_dir"], name=t["name"], exist_ok=True, val=True)
    if a["enabled"]:
        init = args.init or cfg["model"]["weights"]
        kw.update(name=a["name"], epochs=a["epochs"],
                  trainer=make_adv_trainer(a["attack"], a["eps"], a["ratio"], a["pgd_steps"]))
    else:
        init = args.init or cfg["model"]["init_weights"]
    print(f"[train] init={init} -> {cfg['paths']['runs_dir']}/{kw['name']}")
    YOLO(init).train(**kw)
    write_json(Path(cfg["paths"]["runs_dir"]) / kw["name"] / "run_meta.json",
               {**run_metadata(cfg), "init_weights": init, "train_kwargs": {k: str(v) for k, v in kw.items()}})


if __name__ == "__main__":
    main()
