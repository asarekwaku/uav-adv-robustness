"""Phase 1a: download VisDrone-DET2019, convert to YOLO format, build fixed letterboxed eval sets.

    python scripts/prepare_data.py                         # all splits
    python scripts/prepare_data.py --set data.splits=[val] --set data.limit=40   # local smoke test
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uavadv.config import base_parser, load_config, write_json  # noqa: E402
from uavadv.data import convert_split, download_split, make_eval_set, write_dataset_yaml  # noqa: E402


def main():
    cfg = load_config(base_parser(__doc__).parse_args())
    root, eval_dir = Path(cfg["paths"]["data_dir"]), Path(cfg["paths"]["eval_dir"])
    stats = []
    for split in cfg["data"]["splits"]:
        download_split(root, split)
        stats.append(convert_split(root, split))
        print(f"[data] {stats[-1]}")
    yaml_path = write_dataset_yaml(root)
    for split in [s for s in cfg["data"]["splits"] if s in ("val", "test")]:
        out = eval_dir / f"{split}_{cfg['eval']['imgsz']}"
        make_eval_set(root, split, out, cfg["eval"]["imgsz"], cfg["data"]["limit"])
        print(f"[data] eval set ready: {out}")
    write_json(Path(cfg["paths"]["out_dir"]) / "data_conversion.json", {"dataset_yaml": str(yaml_path), "splits": stats})


if __name__ == "__main__":
    main()
