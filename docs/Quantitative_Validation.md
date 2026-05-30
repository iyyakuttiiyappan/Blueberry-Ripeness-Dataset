# Quantitative Validation

## Dataset Scale

- Images: 424
- Annotated berry instances: 13,909
- Release files: 3,422
- Release size: 1.47 GiB

## Class Distribution

|   class_id | class_name        |   objects |   object_percent |   images_with_objects |   mask_pixels |   mask_area_percent |   connected_components |
|-----------:|:------------------|----------:|-----------------:|----------------------:|--------------:|--------------------:|-----------------------:|
|          1 | green_immature    |      8797 |          63.2468 |                   421 |      82440183 |              1.6203 |                   9216 |
|          2 | pale_pink         |       443 |           3.185  |                   259 |      10719889 |              0.2107 |                    458 |
|          3 | pink_turns_purple |       440 |           3.1634 |                   233 |      10663786 |              0.2096 |                    477 |
|          4 | fully_ripe        |      4091 |          29.4126 |                   403 |      74136995 |              1.4571 |                   4422 |
|          5 | over_ripe         |       138 |           0.9922 |                    45 |       2500635 |              0.0491 |                    140 |

## Split Summary

| split   |   images |   total_objects |   green_immature_objects |   pale_pink_objects |   pink_turns_purple_objects |   fully_ripe_objects |   over_ripe_objects |
|:--------|---------:|----------------:|-------------------------:|--------------------:|----------------------------:|---------------------:|--------------------:|
| train   |      296 |            9710 |                     6141 |                 309 |                         308 |                 2856 |                  96 |
| val     |       64 |            2099 |                     1328 |                  67 |                          66 |                  617 |                  21 |
| test    |       64 |            2100 |                     1328 |                  67 |                          66 |                  618 |                  21 |

## Baseline Segmentation Results

Standard cross-entropy:

| variant             |   train_images |   best_val_metric |   miou_foreground |   dice_foreground |   pixel_accuracy |
|:--------------------|---------------:|------------------:|------------------:|------------------:|-----------------:|
| real_only           |            296 |            0.4824 |            0.4737 |            0.5904 |           0.9875 |
| synthetic_only      |            368 |            0.2874 |            0.287  |            0.3915 |           0.975  |
| real_plus_synthetic |            664 |            0.4978 |            0.4463 |            0.5661 |           0.9873 |

Class-balanced cross-entropy:

| variant             |   best_val_metric |   miou_foreground |   dice_foreground |   pixel_accuracy |
|:--------------------|------------------:|------------------:|------------------:|-----------------:|
| real_only           |            0.4422 |            0.4571 |            0.577  |           0.9852 |
| real_plus_synthetic |            0.4464 |            0.4209 |            0.5455 |           0.9836 |

Interpretation: the real-only release-split baseline is the main benchmark. Synthetic augmentation experiments should be reported as optional training studies and not as part of the primary dataset record.
