"""Download a few shards of DeepPatent and build contact sheets for hand-picking.

DeepPatent (Voxel51/DeepPatent on Hugging Face) stores US design-patent drawing
sheets as PNGs in ~352 shards (data/data_0 ... data/data_351), each ~1,000 images.
File names look like USD0806350-20180102-D00000.png:
    USD0806350 = design patent D806350, 20180102 = issue date, D00000 = sheet 0.

Usage (laptop or a Colab CPU runtime, no GPU needed):
    python scripts/fetch_candidates.py --shards 0 120 240 --out dataset/candidates

Then open dataset/candidates/sheets/*.png, note the IDs you like, and list them
(one file name per line) in dataset/selection.txt.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

from PIL import Image, ImageDraw

REPO_ID = "Voxel51/DeepPatent"


def download_shards(shards: list[int], out_dir: Path) -> list[Path]:
    from huggingface_hub import snapshot_download  # imported lazily: only needed here

    patterns = [f"data/data_{s}/*.png" for s in shards]
    local = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        allow_patterns=patterns,
        local_dir=out_dir / "raw",
    )
    files = sorted(Path(local).glob("data/data_*/*.png"))
    print(f"Downloaded {len(files)} drawing sheets from shards {shards}")
    return files


def make_contact_sheets(files: list[Path], out_dir: Path, per_sheet: int = 48,
                        thumb: int = 220, cols: int = 8) -> list[Path]:
    """Tile thumbnails with their file stem printed underneath, for fast browsing."""
    out_dir.mkdir(parents=True, exist_ok=True)
    label_h = 18
    rows = math.ceil(per_sheet / cols)
    sheets = []
    for start in range(0, len(files), per_sheet):
        batch = files[start:start + per_sheet]
        canvas = Image.new("RGB", (cols * thumb, rows * (thumb + label_h)), "white")
        draw = ImageDraw.Draw(canvas)
        for i, f in enumerate(batch):
            try:
                im = Image.open(f).convert("L")
            except OSError:
                continue
            im.thumbnail((thumb - 8, thumb - 8))
            x, y = (i % cols) * thumb, (i // cols) * (thumb + label_h)
            canvas.paste(im.convert("RGB"), (x + (thumb - im.width) // 2, y + (thumb - im.height) // 2))
            draw.rectangle([x, y, x + thumb - 1, y + thumb + label_h - 1], outline=(200, 200, 200))
            draw.text((x + 4, y + thumb + 2), f.stem, fill=(0, 0, 0))
        path = out_dir / f"sheet_{start // per_sheet:03d}.png"
        canvas.save(path)
        sheets.append(path)
    print(f"Wrote {len(sheets)} contact sheets to {out_dir}")
    return sheets


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", type=int, nargs="+", default=[0, 120, 240],
                    help="Shard numbers to download (0-351). Spread them out for variety.")
    ap.add_argument("--out", type=Path, default=Path("dataset/candidates"))
    ap.add_argument("--skip-download", action="store_true",
                    help="Only rebuild contact sheets from files already in --out/raw.")
    args = ap.parse_args()

    if args.skip_download:
        files = sorted((args.out / "raw").glob("data/data_*/*.png"))
    else:
        files = download_shards(args.shards, args.out)
    make_contact_sheets(files, args.out / "sheets")


if __name__ == "__main__":
    main()
