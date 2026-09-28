"""Create train-only visual QA pairs; this does not certify actual plate coverage."""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from reid.infer.plate_mask import DEFAULT_PLATE_MASK, plate_mask_box
from reid.data.preprocess import clip_bbox
from scripts.diagnose_c3_gap import train_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("dataset"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("refusing to overwrite QA output")
    query, _ = train_rows(args.data_root, Path("splits"), 0)
    selected = query[::max(1, len(query) // 8)][:8]  # Fixed spaced train sample, no error selection.
    args.out.mkdir(parents=True)
    records = []
    for page in range(2):
        canvas = Image.new("RGB", (960, 760), "white")
        draw = ImageDraw.Draw(canvas)
        for index, row in enumerate(selected[page * 4:(page + 1) * 4]):
            with Image.open(row.image_path) as source:
                source = source.convert("RGB")
                box = clip_bbox(*row.bbox, *source.size)
                mask_box = plate_mask_box(row.bbox, *source.size)
                masked = source.copy()
                masked.paste((DEFAULT_PLATE_MASK.fill,) * 3, mask_box)
                for column, frame in enumerate((source, masked)):
                    crop = ImageOps.contain(frame.crop(box), (475, 164))
                    canvas.paste(crop, (column * 480, index * 190 + 23))
                draw.text((4, index * 190 + 4), f"{row.image_id} | original / geometric control", fill="black")
                records.append({"image_id": row.image_id, "bbox": row.bbox, "mask_box": mask_box})
        canvas.save(args.out / f"page{page + 1}.png")
    (args.out / "selection.json").write_text(json.dumps({
        "scope": "8 regularly spaced train fold0 queries; qualitative coverage inspection only",
        "mask": DEFAULT_PLATE_MASK.as_dict(), "images": records,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
