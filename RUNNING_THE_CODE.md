# Running The Code

These commands assume you are running from the root of this GitHub repository.

## 1. Create A Python Environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The model-validation scripts require PyTorch. Install the PyTorch build that matches your CUDA/CPU environment if the default `pip install -r requirements.txt` command does not provide the right build.

## 2. Download The Dataset

Download the dataset archive from the institutional drive folder:

https://kuacae-my.sharepoint.com/:f:/g/personal/iyyakutti_ganapathi_ku_ac_ae/IgA50zXBVJdqS7uqbg0ZaHjCAdMg9QA4KlIg2OfNmAzytnc?e=tgNy7Q

For normal use, download and extract:

```text
blueberry_ripeness_real_curated_dataset_v1.zip
```

Recommended extraction location:

```text
outputs/scientific_data_release/
```

After extraction, the real dataset should contain:

```text
outputs/scientific_data_release/
  dataset/images/
  dataset/masks_binary/
  dataset/masks_semantic/
  dataset/masks_overall/
  metadata/
  splits/
  reports/
```

## 3. Use The Already-Curated Dataset

Most users do not need to rerun preprocessing. Use the extracted release directly:

- RGB images: `outputs/scientific_data_release/dataset/images/`
- Semantic masks: `outputs/scientific_data_release/dataset/masks_semantic/`
- Binary masks: `outputs/scientific_data_release/dataset/masks_binary/<class>/`
- Splits: `outputs/scientific_data_release/splits/train.txt`, `val.txt`, `test.txt`
- Manifest: `outputs/scientific_data_release/metadata/dataset_manifest.csv`

Semantic mask labels:

```text
0   background
1   green_immature
2   pale_pink
3   pink_turns_purple
4   fully_ripe
5   over_ripe
255 overlap/ignore
```

## 4. Rebuild The Curated Release From Raw Source Folders

Only run this step if you have the original raw folder structure.

Expected raw layout:

```text
data/
  images/
  green_immature/
  pale_pink/
  pink_turns_purple/
  fully_ripe/
  over_ripe/
  Overall/
  Image Wise Classname Count.xlsx
```

Command:

```bash
python scripts/prepare_scientific_data_release.py \
  --data-root data \
  --output-root outputs/scientific_data_release \
  --threshold 127 \
  --force
```

Main outputs:

```text
outputs/scientific_data_release/dataset/
outputs/scientific_data_release/metadata/
outputs/scientific_data_release/splits/
outputs/scientific_data_release/reports/
outputs/scientific_data_release/figures/
```

## 5. Optional: Create Natural Synthetic Training Augmentation

Synthetic data should be used only as training augmentation. Do not report it as additional independent real observations.

```bash
python scripts/create_natural_synthetic_dataset.py \
  --release-root outputs/scientific_data_release \
  --output-root outputs/synthetic_balanced_natural_augmentation \
  --short-side 1024 \
  --seed 2026 \
  --force
```

Expected output:

```text
outputs/synthetic_balanced_natural_augmentation/
```

## 6. Smoke-Test Model Validation

This runs a tiny real-only segmentation test to confirm the environment and paths work.

```bash
python scripts/run_natural_synthetic_segmentation_validation.py \
  --config configs/segmentation_validation.yaml \
  --release-root outputs/scientific_data_release \
  --output-root outputs/synthetic_model_validation_smoke \
  --variants real_only \
  --epochs 1 \
  --batch-size 2 \
  --limit 8 \
  --no-pretrained
```

## 7. Reproduce Full Segmentation Validation

Real-only baseline:

```bash
python scripts/run_natural_synthetic_segmentation_validation.py \
  --config configs/segmentation_validation.yaml \
  --release-root outputs/scientific_data_release \
  --output-root outputs/synthetic_model_validation \
  --variants real_only \
  --epochs 60 \
  --batch-size 4 \
  --device cuda
```

Synthetic comparison, after creating or downloading the optional synthetic package:

```bash
python scripts/run_natural_synthetic_segmentation_validation.py \
  --config configs/segmentation_validation.yaml \
  --release-root outputs/scientific_data_release \
  --synthetic-root outputs/synthetic_balanced_natural_augmentation \
  --output-root outputs/synthetic_model_validation \
  --variants real_only synthetic_only real_plus_synthetic \
  --epochs 60 \
  --batch-size 4 \
  --device cuda
```

Class-balanced cross-entropy comparison:

```bash
python scripts/run_natural_synthetic_segmentation_validation.py \
  --config configs/segmentation_validation.yaml \
  --release-root outputs/scientific_data_release \
  --synthetic-root outputs/synthetic_balanced_natural_augmentation \
  --output-root outputs/synthetic_model_validation_weighted \
  --variants real_only real_plus_synthetic \
  --loss-mode balanced_ce \
  --epochs 60 \
  --batch-size 4 \
  --device cuda
```

Main result tables are written to:

```text
<output-root>/results/tables/
```

## 8. Verify Downloaded Archives With SHA256

The data repository includes `SHA256SUMS.txt`. To verify a downloaded archive on Windows:

```powershell
Get-FileHash "blueberry_ripeness_real_curated_dataset_v1.zip" -Algorithm SHA256
```

The hash should match the corresponding line in `SHA256SUMS.txt`.
