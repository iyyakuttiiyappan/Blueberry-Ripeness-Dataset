from __future__ import annotations

import json
import re
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
from openpyxl import Workbook


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OLD_DATA = PROJECT_ROOT / "data"
ONEDRIVE_ROOT = OLD_DATA / "OneDrive_1_29-6-2026"
NEW_RGB_DIR = ONEDRIVE_ROOT / "frames_set_1_Updated_17042026-001" / "frames_set_1" / "All"
OUTPUT_ROOT = OLD_DATA / "combined_514"

CLASS_NAMES = [
    "green_immature",
    "pale_pink",
    "pink_turns_purple",
    "fully_ripe",
    "over_ripe",
]

COCO_CATEGORY_TO_CLASS = {
    1: "fully_ripe",
    2: "green_immature",
    3: "over_ripe",
    4: "pale_pink",
    5: "pink_turns_purple",
}


def reset_output() -> None:
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
    (OUTPUT_ROOT / "images").mkdir(parents=True)
    (OUTPUT_ROOT / "Overall").mkdir(parents=True)
    for class_name in CLASS_NAMES:
        (OUTPUT_ROOT / class_name).mkdir(parents=True)
    (OUTPUT_ROOT / "metadata").mkdir(parents=True)


def copy_old_424() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    old_images = sorted((OLD_DATA / "images").glob("*.jpg"))
    for image_path in old_images:
        shutil.copy2(image_path, OUTPUT_ROOT / "images" / image_path.name)
        shutil.copy2(OLD_DATA / "Overall" / image_path.name, OUTPUT_ROOT / "Overall" / image_path.name)
        for class_name in CLASS_NAMES:
            shutil.copy2(OLD_DATA / class_name / image_path.name, OUTPUT_ROOT / class_name / image_path.name)
        rows.append(
            {
                "filename": image_path.name,
                "stem": image_path.stem,
                "source_set": "Data16march2026",
                "source_modality": "smartphone",
                "metadata_source": "embedded_exif",
            }
        )
    return rows


def find_zip(name: str) -> Path:
    matches = sorted(path for path in ONEDRIVE_ROOT.rglob(name) if path.is_file())
    if not matches:
        raise FileNotFoundError(f"Could not find {name} under {ONEDRIVE_ROOT}")
    # Prefer the standalone Frames_Set package; duplicate zips exist in the combined export.
    for path in matches:
        if "Frames_Set_1_Updated_05042026-20260626T073315Z-3-001" in str(path):
            return path
    return matches[0]


def read_new_coco() -> dict:
    zip_path = find_zip("Annotation-Json.zip")
    with ZipFile(zip_path) as zf:
        json_names = [name for name in zf.namelist() if name.lower().endswith(".json")]
        if len(json_names) != 1:
            raise ValueError(f"Expected one JSON in {zip_path}, found {json_names}")
        return json.loads(zf.read(json_names[0]).decode("utf-8"))


def new_counts_from_coco(coco: dict) -> pd.DataFrame:
    image_by_id = {image["id"]: image["file_name"] for image in coco["images"]}
    counts: dict[str, Counter] = {filename: Counter() for filename in image_by_id.values()}
    for ann in coco["annotations"]:
        filename = image_by_id[ann["image_id"]]
        class_name = COCO_CATEGORY_TO_CLASS[int(ann["category_id"])]
        counts[filename][class_name] += 1

    rows = []
    for filename in sorted(counts):
        row = {"Image Name": filename}
        for class_name in CLASS_NAMES:
            row[class_name] = int(counts[filename][class_name])
        row["Total"] = sum(row[class_name] for class_name in CLASS_NAMES)
        rows.append(row)
    return pd.DataFrame(rows)


def parse_filename_datetime(filename: str) -> str:
    patterns = [
        r"_(\d{8})_(\d{6})_",
        r"dji_fly_(\d{8})_(\d{6})",
    ]
    for pattern in patterns:
        match = re.search(pattern, filename, flags=re.IGNORECASE)
        if match:
            dt = datetime.strptime(" ".join(match.groups()), "%Y%m%d %H%M%S")
            return dt.strftime("%Y:%m:%d %H:%M:%S")
    return ""


def source_modality_for(filename: str) -> str:
    return "drone_video_frame" if "dji_fly" in filename.lower() else "video_frame"


def extract_new_images_and_masks() -> list[dict[str, str]]:
    image_zip = find_zip("Images.zip")
    mask_zip = find_zip("Binary_Mask.zip")
    rows: list[dict[str, str]] = []

    with ZipFile(image_zip) as image_zf:
        image_names = sorted(name for name in image_zf.namelist() if name.lower().endswith(".jpg"))
        for name in image_names:
            filename = Path(name).name
            (OUTPUT_ROOT / "images" / filename).write_bytes(image_zf.read(name))
            rows.append(
                {
                    "filename": filename,
                    "stem": Path(filename).stem,
                    "source_set": "frames_set_1",
                    "source_modality": source_modality_for(filename),
                    "metadata_source": "filename_derived",
                    "filename_datetime": parse_filename_datetime(filename),
                }
            )

    folder_map = {"all": "Overall", **{class_name: class_name for class_name in CLASS_NAMES}}
    with ZipFile(mask_zip) as mask_zf:
        mask_names = sorted(name for name in mask_zf.namelist() if name.lower().endswith(".png"))
        for name in mask_names:
            parts = Path(name).parts
            if len(parts) < 3:
                continue
            source_folder = parts[-2]
            target_folder = folder_map.get(source_folder)
            if target_folder is None:
                continue
            filename = Path(name).name
            (OUTPUT_ROOT / target_folder / filename).write_bytes(mask_zf.read(name))
    return rows


def write_count_workbook(old_count_path: Path, new_counts: pd.DataFrame) -> None:
    old_counts = pd.read_excel(old_count_path)
    old_counts = old_counts[old_counts["Image Name"].astype(str).str.lower() != "total"].copy()
    keep_cols = ["Image Name", *CLASS_NAMES, "Total"]
    old_counts = old_counts[keep_cols]
    combined = pd.concat([old_counts, new_counts[keep_cols]], ignore_index=True)

    workbook = Workbook()
    ws = workbook.active
    ws.title = "Counts"
    ws.append(keep_cols)
    for row in combined[keep_cols].itertuples(index=False):
        ws.append(list(row))
    ws.append(["Total", *[int(combined[class_name].sum()) for class_name in CLASS_NAMES], int(combined["Total"].sum())])
    workbook.save(OUTPUT_ROOT / "Image Wise Classname Count.xlsx")
    combined.to_csv(OUTPUT_ROOT / "metadata" / "image_level_counts_combined.csv", index=False)


def validate_stage() -> None:
    image_files = sorted((OUTPUT_ROOT / "images").glob("*.jpg"))
    if len(image_files) != 514:
        raise RuntimeError(f"Expected 514 staged RGB images, found {len(image_files)}")
    for image_path in image_files:
        for folder in ["Overall", *CLASS_NAMES]:
            candidates = list((OUTPUT_ROOT / folder).glob(f"{image_path.stem}.*"))
            if not candidates:
                raise FileNotFoundError(f"Missing {folder} mask for {image_path.name}")


def main() -> None:
    reset_output()
    source_rows = copy_old_424()
    coco = read_new_coco()
    new_counts = new_counts_from_coco(coco)
    source_rows.extend(extract_new_images_and_masks())
    write_count_workbook(OLD_DATA / "Image Wise Classname Count.xlsx", new_counts)

    source_df = pd.DataFrame(source_rows).sort_values("filename")
    source_df.to_csv(OUTPUT_ROOT / "metadata" / "source_metadata.csv", index=False)
    validate_stage()

    counts_df = pd.read_excel(OUTPUT_ROOT / "Image Wise Classname Count.xlsx")
    counts_df = counts_df[counts_df["Image Name"].astype(str).str.lower() != "total"]
    summary = {
        "images": int(len(counts_df)),
        "annotated_berries": int(counts_df["Total"].sum()),
        "class_counts": {class_name: int(counts_df[class_name].sum()) for class_name in CLASS_NAMES},
        "source_modality_counts": source_df["source_modality"].value_counts().to_dict(),
        "source_set_counts": source_df["source_set"].value_counts().to_dict(),
    }
    (OUTPUT_ROOT / "metadata" / "combined_514_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Prepared combined data root: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
