# Configs

`fresh_benchmark_514.yaml` is the benchmark configuration used for the 514-image AerialYield-B2D validation runs.

Before running experiments, edit:

- `data_root`: path to the raw/combined working dataset layout, usually `data/combined_514`.
- `paths.fixed_split_manifest`: path to the curated release manifest, usually `outputs/scientific_data_release_514/metadata/dataset_manifest.csv`.
- training settings such as `epochs`, `batch_size`, and `num_workers` for your hardware.

The config is included for reproducibility; 
