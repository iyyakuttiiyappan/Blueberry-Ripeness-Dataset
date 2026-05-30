# Preprocessing Pipeline

1. Raw RGB images and mask folders are scanned.
2. RGB images are EXIF-transposed to align with portrait masks.
3. Source JPEG masks are converted to thresholded binary PNG masks using threshold > 127.
4. Per-class binary masks are preserved in `dataset/masks_binary`.
5. Overall masks are written to `dataset/masks_overall`.
6. Semantic masks are generated with label IDs 0-5 and ignore value 255 for class-overlap pixels.
7. Per-image counts, mask pixels, connected components, image geometry, EXIF fields, image hashes, and quality metrics are computed.
8. A deterministic train/validation/test split is generated.
9. Technical validation reports, figures, and overlay contact sheets are written.

Primary script: `scripts/prepare_scientific_data_release.py`.
