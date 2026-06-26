# BMI Predictor

This project contains scripts for preparing image datasets and training BMI prediction models from body and face images.

## Where To Add Images

Put the raw dataset images in a folder named `visual_bmi/` at the project root:

```text
bmi_predictor/
  visual_bmi/
    visual_bmi_annotations.csv
    bodyface_1to17/
      person_001.jpg
      person_002.jpg
```

The training scripts look for a CSV inside the dataset folder. Preferred CSV name:

```text
visual_bmi/visual_bmi_annotations.csv
```

The CSV should include an image filename/path column and either:

- a BMI column, for example `BMI` or `bmi`
- or height and weight columns, for example `height_in` and `weight_lb`

Supported image formats include `.jpg`, `.jpeg`, `.png`, `.webp`, and `.bmp`.

Image and generated output folders are ignored by Git through `.gitignore`, so dataset images can stay local without being committed.

## Setup

Create and activate a Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the Python dependencies used by the scripts:

```bash
pip install tensorflow pandas numpy pillow scikit-learn opencv-python mediapipe shap matplotlib
```

## Prepare Cropped Images

To create body and face crops from the raw dataset:

```bash
python body_crop.py --in_dir visual_bmi --out_dir visual_bmi_cropped
```

This writes:

```text
visual_bmi_cropped/body/
visual_bmi_cropped/face/
```

The script also copies CSV files into the cropped output folders so each cropped dataset is self-contained.

## Optional Data Curation

Run the curation step to extract pose landmarks, estimate person size in the frame, and cluster image postures:

```bash
python data_curation.py --data_dir visual_bmi --k 5 --preview
```

This writes:

```text
visual_bmi/curation.csv
visual_bmi/curation_clusters_preview.jpg
```

Open the preview image, decide which clusters contain good full-body examples, then pass those cluster numbers during training.

## Train A Model

Train on the raw dataset:

```bash
python bmi_model.py --data_dir visual_bmi --output_dir outputs
```

Train on cropped body/face images:

```bash
python train_two_branch.py --data_dir visual_bmi_cropped --output_dir outputs_v3
```

Train with curation filters:

```bash
python train_two_branch.py \
  --data_dir visual_bmi_cropped \
  --output_dir outputs_v4 \
  --curation_csv visual_bmi/curation.csv \
  --min_bbox_area_ratio 0.15 \
  --keep_clusters 2,3
```

## Outputs

Training writes model files, splits, and metadata into the selected output folder, such as:

```text
outputs/
outputs_v3/
outputs_v4/
```

Common generated files include:

```text
best_bmi_model.keras
bmi_model.tflite
train_split.csv
val_split.csv
test_split.csv
model_metadata.txt
```

These output folders are ignored by Git.

## Notes

- Keep raw image data in `visual_bmi/`.
- Keep generated crops in `visual_bmi_cropped/`.
- Do not commit datasets, generated model files, or virtual environments.
- If an image path in the CSV does not resolve, the loader searches under the dataset folder and common nested paths such as `bodyface_1to17/`.
