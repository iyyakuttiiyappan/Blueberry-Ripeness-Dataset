# Running The Code

These instructions assume you are working from the root of this GitHub repository.

## 1. Install Dependencies

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

Install a PyTorch build that matches your GPU/CPU environment if needed.

## 2. Download The Dataset

Temporary institutional review folder:

https://kuacae-my.sharepoint.com/:f:/g/personal/iyyakutti_ganapathi_ku_ac_ae/IgA50zXBVJdqS7uqbg0ZaHjCAdMg9QA4KlIg2OfNmAzytnc?e=tgNy7Q

Recommended extraction location:

```text
outputs/scientific_data_release_514/
```

The extracted curated dataset should contain:

```text
outputs/scientific_data_release_514/
  dataset/images/
  dataset/masks_binary/
  dataset/masks_semantic/
  dataset/masks_overall/
  metadata/
  splits/
  reports/
```

## 3. Use The Curated Dataset Directly

For semantic segmentation:

- inputs: `outputs/scientific_data_release_514/dataset/images/`
- targets: `outputs/scientific_data_release_514/dataset/masks_semantic/`
- ignore label: `255`

For per-class binary masks:

```text
outputs/scientific_data_release_514/dataset/masks_binary/<class_name>/
```

For counts:

```text
outputs/scientific_data_release_514/metadata/image_level_counts.csv
outputs/scientific_data_release_514/metadata/dataset_manifest.csv
```

## 4. Rebuild The Curated Release From Raw/Combined Folders

Only run this step if you have the original raw/combined working layout:

```text
data/combined_514/
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
  --data-root data/combined_514 \
  --output-root outputs/scientific_data_release_514 \
  --threshold 127 \
  --force
```

## 5. Reproduce Annotation Preparation For Benchmark Runs

The benchmark code uses the raw/combined working layout because it builds crops and task-specific annotation files.

Edit `configs/fresh_benchmark_514.yaml` if your paths differ, then run:

```bash
python scripts/fresh_prepare_annotations.py --config configs/fresh_benchmark_514.yaml --rebuild
```

## 6. Reproduce The Main Technical-Validation Baselines

Semantic segmentation baseline:

```bash
python scripts/fresh_run_task.py \
  --config configs/fresh_benchmark_514.yaml \
  --task segmentation \
  --method fpn_convnextv2_tiny \
  --seed 42
```

Counting baseline:

```bash
python scripts/fresh_run_task.py \
  --config configs/fresh_benchmark_514.yaml \
  --task counting \
  --method count_efficientnetv2_s \
  --seed 42
```

Summarize benchmark outputs:

```bash
python scripts/fresh_summarize.py --config configs/fresh_benchmark_514.yaml
```

For a quick path check, add `--epochs 1 --limit 12` to `fresh_run_task.py`.
