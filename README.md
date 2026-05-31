# Blueberry Ripeness Dataset

This repository contains the code and documentation for a high-resolution greenhouse blueberry ripeness dataset with five ripeness-stage mask annotations.

## Dataset

The image and mask data are hosted separately because the curated release is larger than a normal GitHub repository.

Institutional drive download/review folder:

https://kuacae-my.sharepoint.com/:f:/g/personal/iyyakutti_ganapathi_ku_ac_ae/IgA50zXBVJdqS7uqbg0ZaHjCAUUEo0-4_MDTx5-sVKXWIkI

Primary dataset archive:

- `blueberry_ripeness_real_curated_dataset_v1.zip`

Optional training-only synthetic augmentation archive:

- `blueberry_ripeness_training_only_natural_synthetic_augmentation_v1.zip`

Use the real curated dataset as the primary data record. The synthetic archive is only for optional augmentation experiments and should not be counted as additional independent real observations.

## Dataset Summary

- 424 high-resolution real greenhouse RGB images.
- 13,909 annotated berry instances.
- Five ripeness classes: `green_immature`, `pale_pink`, `pink_turns_purple`, `fully_ripe`, and `over_ripe`.
- Per-class binary masks, semantic masks, overall masks, image-level count metadata, and fixed train/validation/test splits.
- Optional natural synthetic augmentation package for training-only imbalance experiments.

## Repository Contents

```text
configs/                 Minimal configuration for segmentation validation
docs/                    Preprocessing and validation documentation
figures/                 Small paper-ready overview figures
metadata_examples/       Small CSV/JSON metadata examples from the release
scripts/                 Reproducibility scripts
src/blueberry_multitask/ Shared model-validation utilities
RUNNING_THE_CODE.md      Step-by-step commands
DATASET_CARD.md          Dataset card
requirements.txt         Python dependencies
```

## Quick Start

See [RUNNING_THE_CODE.md](RUNNING_THE_CODE.md) for environment setup, dataset extraction, preprocessing, synthetic augmentation, and model-validation commands.

## Citation

Please cite the accompanying Scientific Data Data Descriptor after publication. A permanent dataset DOI will be added after repository publication.
