# Technical Validation

## Completeness And File Integrity

- RGB image-mask groups checked: 424
- Every RGB image has one overall mask and five class-specific masks.
- Unique release image SHA-256 hashes: 424
- Unique filenames: 424
- No missing image-mask pairs were found during preprocessing.

## Image Geometry And Orientation

Images were EXIF-transposed before release so the RGB pixels align directly with the masks.

|   width |   height |   images |
|--------:|---------:|---------:|
|    3000 |     4000 |      423 |
|    4000 |     3000 |        1 |

|   exif_orientation |   images |
|-------------------:|---------:|
|                  6 |      423 |
|                  1 |        1 |

## Annotation Counts

| class_name        |   objects |   object_percent |   images_with_objects |   mask_pixels |   mask_area_percent |
|:------------------|----------:|-----------------:|----------------------:|--------------:|--------------------:|
| green_immature    |      8797 |        63.2468   |                   421 |      82440183 |           1.62029   |
| pale_pink         |       443 |         3.18499  |                   259 |      10719889 |           0.21069   |
| pink_turns_purple |       440 |         3.16342  |                   233 |      10663786 |           0.209587  |
| fully_ripe        |      4091 |        29.4126   |                   403 |      74136995 |           1.4571    |
| over_ripe         |       138 |         0.992163 |                    45 |       2500635 |           0.0491477 |

## Empty-Mask Audit

Empty masks are expected for rare classes when a ripeness stage is absent from an image.

| class_name        |   empty_masks |   non_empty_masks |
|:------------------|--------------:|------------------:|
| green_immature    |             3 |               421 |
| pale_pink         |           165 |               259 |
| pink_turns_purple |           191 |               233 |
| fully_ripe        |            21 |               403 |
| over_ripe         |           379 |                45 |

## Mask Conversion Audit

The source masks are JPEG files and therefore contain antialiasing/compression edge values. The release converts them to thresholded binary PNG masks using threshold > 127.

- Source mask ambiguous edge/compression pixels across class and overall masks: 32,768,752
- Images with any class-overlap pixels after thresholding: 4
- Maximum class-overlap pixels in one image: 12,732
- Mean class-overlap pixels per image: 87.99
- Images with any class-union vs overall-mask mismatch pixels: 237
- Maximum union mismatch pixels in one image: 1,108
- Mean union mismatch pixels per image: 13.69

Overlap pixels are encoded as value 255 in `masks_semantic` and should be treated as ignore pixels for semantic segmentation training. The original per-class binary masks remain available for multilabel segmentation.

## Image Quality Screening

Brightness, contrast, saturation, and Laplacian sharpness were measured on downsampled EXIF-oriented images to screen for gross acquisition problems.

| index   |   brightness_mean |   brightness_std |   sharpness_laplacian_var |   saturation_mean |
|:--------|------------------:|-----------------:|--------------------------:|------------------:|
| count   |         424       |        424       |                   424     |          424      |
| mean    |         109.456   |         51.895   |                  1027.15  |          127.622  |
| std     |           6.32773 |          5.36847 |                   579.897 |           21.4185 |
| min     |          85.8831  |         35.5888  |                   126.554 |           72.2939 |
| 25%     |         105.217   |         47.9431  |                   603.897 |          113.046  |
| 50%     |         109.147   |         51.2897  |                   905.421 |          131.272  |
| 75%     |         113.225   |         56.2184  |                  1315.5   |          143.174  |
| max     |         134.289   |         67.022   |                  3527.84  |          175.985  |

## Recommended Benchmark Split

A deterministic 296/64/64 train/validation/test split was generated to balance image counts and approximate object-count balance across ripeness stages.

| split   |   images |   total_objects |   green_immature_objects |   pale_pink_objects |   pink_turns_purple_objects |   fully_ripe_objects |   over_ripe_objects |
|:--------|---------:|----------------:|-------------------------:|--------------------:|----------------------------:|---------------------:|--------------------:|
| train   |      296 |            9710 |                     6141 |                 309 |                         308 |                 2856 |                  96 |
| val     |       64 |            2099 |                     1328 |                  67 |                          66 |                  617 |                  21 |
| test    |       64 |            2100 |                     1328 |                  67 |                          66 |                  618 |                  21 |

## Validation Figures

- `figures/class_object_distribution.png`
- `figures/split_object_distribution.png`
- `figures/image_quality_distribution.png`
- `figures/overlays/contact_sheet_overlays.jpg`
