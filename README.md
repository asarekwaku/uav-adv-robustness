# Adversarial robustness of a YOLO detector on UAV imagery (VisDrone)

Code for the term project *"Adversarial Attacks Against Machine Learning Models in UAV-Based Aerial
Surveillance: A Proposed Robustness Evaluation"*. Every script implements one box of the proposal's Fig. 1.

| Proposal | Script | Output |
|---|---|---|
| Phase 1: data, fixed splits | `scripts/prepare_data.py` | `data/VisDrone` (YOLO format), `data/eval_sets/{val,test}_640/clean`, `results/data_conversion.json` |
| Phase 1: clean baseline | `scripts/train.py` | `runs/baseline/weights/best.pt`, `run_meta.json` |
| Phase 3a: defense chosen on **val** | `scripts/calibrate_defense.py` | `results/defense_calibration.csv`, `results/chosen_defense.json` |
| Phase 2 + 3: Table I on **test** | `scripts/run_all.py` | `results/results_test.csv`, `run_meta_*.json` |
| Phase 3b: adversarial training | `scripts/train.py --set adv_train.enabled=true` | `runs/advtrain/...` |
| Sec. V figures/tables | `scripts/plot_results.py` | `results/figures/*.png`, `summary_test.{csv,tex}` |

## Where to run

* **Colab:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/asarekwaku/uav-adv-robustness/blob/main/notebooks/colab_pipeline.ipynb) — or open `notebooks/colab_pipeline.ipynb`. It uses `configs/colab.yaml` (dataset on local disk, results on Drive, resumable).
* **Lab cluster:** edit the paths in `configs/lab.yaml`, then `sbatch slurm/run_pipeline.sbatch`.
* **Laptop smoke test** (tests the code only, not a valid experiment, because it trains on val):
  ```bash
  python scripts/prepare_data.py --config configs/smoke.yaml
  python scripts/train.py --config configs/smoke.yaml
  python scripts/calibrate_defense.py --config configs/smoke.yaml
  python scripts/run_all.py --config configs/smoke.yaml --split val
  python scripts/plot_results.py --config configs/smoke.yaml --split val
  ```
Any setting can be overridden with `--set key=value`, e.g. `--set attack.eps=[2,4] --set attack.seeds=[0,1,2]`.

## Methodology decisions (cite these in the report)

**Data.** VisDrone2019-DET train / val / test-dev → `train` / `val` / `test`. Category 0 (ignored regions)
and 11 (others) are dropped; 1–10 → classes 0–9. `val` is used for model selection and defense calibration;
`test` is only read by `run_all.py`.

**Identical preprocessing.** Each val/test image is letterboxed once to 640×640 and stored as lossless PNG.
Clean, attacked and defended conditions are all evaluated from those pixels with the same validator settings
(`conf=0.001`, NMS IoU 0.7). Model input = stored image, so the threat model's ‖δ‖∞ is measured exactly.

**Threat model / attacks.** White-box, digital, L∞ ≤ ε with ε ∈ {1, 2, 4, 8}/255.
* FGSM: Eq. (1), one step of size ε.
* PGD: uniform random start, 10 steps, α = 2.5ε/steps, projection after every step; repeated over seeds.
* Loss: the Ultralytics YOLOv8/11 training loss (box + cls + DFL) against ground truth, maximised (**untargeted**).
  `attack.objective=vanish` instead minimises every anchor's max class confidence (**object hiding**), reported separately.
* The perturbation is masked to the real image (not letterbox padding) and rounded to 8 bits; `attack.json` in
  each set records the measured ‖δ‖∞, and the code asserts it never exceeds ε.

**Defenses.** JPEG, median, Gaussian blur and bit-depth reduction are swept on val. The chosen one has the
best PGD-ε=4 mAP@0.5 among candidates losing at most 0.02 clean mAP@0.5. On test it is evaluated against
(a) attacks crafted on the undefended model and (b) an **adaptive** PGD that differentiates through the defense
(exact gradient for Gaussian blur, BPDA identity gradient for JPEG/median/bit-depth). Report (b) as the honest number.
Adversarial training mixes 50% FGSM-with-random-start (ε=4/255) images into each batch, fine-tuned from the baseline.

**Metrics.** mAP@0.5, mAP@0.5:0.95, per-class P/R/AP (Ultralytics); ΔmAP = clean − attacked under the same
defense; operating-point recall (conf ≥ 0.25, same-class IoU ≥ 0.5) per class and per COCO size bucket;
**hidden rate** = objects detected on clean but missed after attack ÷ objects detected on clean; latency =
median ms/image at batch 1 for defense + full predict on the same hardware.

## Reproducibility
`ultralytics==8.3.214` is pinned. Every run writes git commit, library versions, hardware and full config to
`run_meta*.json`. Seeds come from `seed` and `attack.seeds`.
