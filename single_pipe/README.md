# single_pipe — SinBMI body-only pipeline

A staged pipeline in the style of [DigitalScale](https://github.com/im-ethz/DigitalScale)
implementing [SinBMI (Technion SIPL, 2025)](https://sipl.ece.technion.ac.il/wp-content/uploads/2025/07/2025-SinBMI-Estimating-BMI-from-a-Single-Image.pdf):
**EfficientNet-B2 + a 7-layer FC+GELU MLP (→512→256→128→64→32→16→1), MSE loss,
Adam + plateau LR scheduling, ~40 epochs**, augmentation = horizontal flip,
Gaussian noise, rotation ≤10°. Body image only — **no face branch, no gender
input** (deliberately different from the two-branch model in the parent repo).
Paper reference numbers on these same two datasets: MAE 2.92, MAPE 9.38%.

Datasets are referenced in place (paths in `datasets.json`) — never copied.
All derived artifacts live here, under `work/` and `runs/`.

| step | script | output |
|---|---|---|
| 1. Clean VisualBodyToBMI | `step1_clean_visualbmi.py` | `work/visualbmi_clean.csv` |
| 2. Clean 2DImage2BMI | `step2_clean_image2bmi.py` | `work/image2bmi_clean.csv` |
| 3. Common CSV format | `step3_make_common_csv.py` | `work/common.csv` |
| 4. Crop/pad/resize uniformly | `step4_preprocess.py` | `work/common_preprocessed.csv` + `work/cache_260/` |
| 5. Split by subject ID | `step5_split.py` | `work/common_split.csv` |
| 6. Train on VisualBodyToBMI | `step6_train_visualbmi.py` | `runs/step6_visualbmi/` |
| 7. Train on combined | `step7_train_combined.py` | `runs/step7_combined/` + `sinbmi_body.tflite` |
| 8. Evaluate both datasets | `step8_evaluate.py` | metrics tables + `test_predictions.csv` |
| 9. Collect own app images | `step9_collect_own.py` | `own_images/` + `work/own_clean.csv` |
| 10. Fine-tune on own images | `step10_finetune_own.py` | `runs/step10_own/` + final `.tflite` |

Steps 1–5 need only pillow/numpy/pandas; steps 6–10 use TensorFlow.
`bash run_pipeline.sh` runs 1–8 and auto-picks the python: it prefers
`~/.venvs/sinbmi312` (a local venv outside iCloud — the in-repo `venv312`
lives in iCloud Drive and gets evicted under disk pressure, after which TF
import stalls re-downloading files; the local venv avoids that, and has
`tensorflow-metal` for GPU training). Delete `~/.venvs/sinbmi312` any time
to reclaim ~2.5 GB; `run_pipeline.sh` then falls back to `venv312`.

## Model contract (for the app — you adapt the app side)

- Input: `[1, 260, 260, 3]` float32, RGB, 0..1 — the body region cropped,
  padded to square with black, resized to 260×260 (exactly
  `sinbmi_lib.crop_pad_resize`).
- Output: `[1, 1]` float32 predicted BMI.

## Notes

- Split is person-level 70/15/15 over the combined data; the 2DImage2BMI
  release's own folders leak 462 person IDs between train and test and are
  ignored. Step 5 asserts zero subject leakage.
- `work/cache_260/` (~200 MB) only exists to make training fast; delete it
  any time and re-run step 4 to rebuild. `step4_preprocess.py --no-cache`
  skips it entirely (training then crops on the fly from the originals).
- BMI outside 10–60 is dropped; stored BMI is cross-checked against
  weight/height where both exist.
