# AerialYield-B2D

**AerialYield-B2D** is a greenhouse blueberry ripeness dataset with RGB images, five-stage ripeness masks and image-level berry-count annotations. In this release, **B2D means BlueBerry Dataset**.

The dataset name is kept for continuity with the project, but the repository documents the acquisition sources explicitly: the 514-image release contains close-range smartphone images and added video-frame/DJI Fly video-frame samples. This mixture is useful because models can be tested across close inspection views and wider plant-scale greenhouse scouting views.

## Dataset Summary

- RGB images: 514
- Annotated berry instances: 30,195
- Ripeness stages: `green_immature`, `pale_pink`, `pink_turns_purple`, `fully_ripe`, `over_ripe`
- Source modalities: {'smartphone': 424, 'video_frame': 67, 'drone_video_frame': 23}
- Fixed split: 360 train, 77 validation, 77 test
- Primary tasks: semantic segmentation, ripeness-stage analysis, berry counting and maturity-distribution estimation

## Data Access

The full image/mask dataset is hosted separately and should be cited through the final data DOI.

- Temporary institutional review folder: https://kuacae-my.sharepoint.com/:f:/g/personal/iyyakutti_ganapathi_ku_ac_ae/IgA50zXBVJdqS7uqbg0ZaHjCAUUEo0-4_MDTx5-sVKXWIkI
- Final dataset DOI: [Zenodo / Figshare]

This GitHub repository contains code, documentation, metadata examples and paper/support figures. 

## Repository Contents

```text
configs/              Reproducibility config for the 514-image validation runs
docs/                 Data access, structure, preprocessing and validation notes
figures/              Small manuscript/support figures
metadata_examples/    Schema examples and compact metadata tables
scripts/              Dataset preparation and validation entry points
src/                  Python utilities used by the validation scripts
RUNNING_THE_CODE.md   Step-by-step instructions
DATASET_CARD.md       Dataset card
CITATION.cff          Citation metadata placeholder
requirements.txt      Python dependencies
```

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Then download the dataset archive from the data record and follow:

```text
RUNNING_THE_CODE.md
```

## Citation

Please cite the accompanying Scientific Data Data Descriptor and the final dataset DOI after publication.


