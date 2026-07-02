# Dataset Card

## Dataset Name

AerialYield-B2D: Greenhouse Blueberry Ripeness Dataset

`B2D` means BlueBerry Dataset.

## Summary

The 514-image release contains RGB greenhouse blueberry imagery with five-stage ripeness masks and image-level berry counts. It combines close-range smartphone images with video-frame/DJI Fly video-frame samples to support evaluation across close inspection and wider plant-scale scouting views.

## Classes

- `green_immature`
- `pale_pink`
- `pink_turns_purple`
- `fully_ripe`
- `over_ripe`

## Primary Data Products

- RGB JPEG images after orientation handling.
- Per-class binary PNG masks.
- Semantic PNG masks with labels 0-5 and 255 ignore.
- Overall berry masks.
- Per-image count metadata.
- Recommended train/validation/test splits.
- Validation reports, quality metrics and checksums.

## Intended Uses

- Ripeness-stage semantic segmentation.
- Berry counting and maturity-distribution estimation.
- Benchmarking robustness across smartphone and drone/video-frame acquisition sources.
- Controlled-environment crop phenotyping research.

## Limitations

The dataset was collected in greenhouse conditions. 
