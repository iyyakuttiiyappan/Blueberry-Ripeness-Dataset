# Dataset Card

## Dataset Name

Blueberry Ripeness Dataset

## Data Access

Institutional drive download/review folder:

https://kuacae-my.sharepoint.com/:f:/g/personal/iyyakutti_ganapathi_ku_ac_ae/IgA50zXBVJdqS7uqbg0ZaHjCAdMg9QA4KlIg2OfNmAzytnc?e=tgNy7Q

## Summary

This dataset contains 424 high-resolution real greenhouse RGB blueberry images with five-stage ripeness mask annotations and berry count metadata.

## Classes

- `green_immature`
- `pale_pink`
- `pink_turns_purple`
- `fully_ripe`
- `over_ripe`

## Primary Data Products

- EXIF-oriented RGB images.
- Per-class binary PNG masks.
- Semantic PNG masks with labels 0-5 and 255 ignore.
- Overall berry masks.
- Per-image count metadata.
- Recommended train/validation/test splits.

## Optional Supplement

The natural synthetic augmentation package is provided only for training experiments. It should not be interpreted as additional real observations.

## Intended Uses

- Ripeness-stage semantic segmentation.
- Berry counting and class-distribution estimation.
- Class-imbalance benchmarking.
- Controlled-environment crop phenotyping research.

## Limitations

The dataset was collected from greenhouse imagery and should be evaluated carefully before deployment in outdoor orchards, different cultivars, or different imaging systems.
