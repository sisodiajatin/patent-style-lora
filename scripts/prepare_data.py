"""Turn hand-picked patent drawing sheets into clean 1024x1024 training images.

For each file listed in dataset/selection.txt this script:
  1. converts to grayscale and binarizes (patent sheets are 1-bit black on white),
  2. finds the main figure by merging nearby ink into blobs and keeping the largest,
     which drops "FIG. 1" labels, sheet numbers and smaller extra figures,
  3. crops to that figure, pads it to a white square so it fills ~80% of the frame,
  4. resizes to 1024x1024 RGB and writes it to dataset/images/,
  5. writes dataset/manifest.csv with patent number, date, sheet and source URL.

Images where the second-largest blob is big (likely a second figure that was
dropped) are flagged in the console and in the manifest's `review` column.
Always eyeball the output folder before training.

Usage:
    python scripts/prepare_data.py --selection dataset/selection.txt
"""
from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

FILENAME_RE = re.compile(r"^USD0*(\d+)-(\d{8})-D(\d+)$")


@dataclass
class PatentRef:
    patent: str      # e.g. "D806350"
    issue_date: str  # e.g. "2018-01-02"
    sheet: int

    @property
    def url(self) -> str:
        return f"https://patents.google.com/patent/US{self.patent}S"


def parse_name(stem: str) -> PatentRef | None:
    m = FILENAME_RE.match(stem)
    if not m:
        return None
    num, date, sheet = m.groups()
    return PatentRef(f"D{num}", f"{date[:4]}-{date[4:6]}-{date[6:]}", int(sheet))


def binarize(gray: np.ndarray, threshold: int = 160) -> np.ndarray:
    """True where there is ink."""
    return gray < threshold


def main_figure_mask(ink: np.ndarray, merge_px: int) -> tuple[np.ndarray, float]:
    """Return (mask of ink belonging to the largest figure, second/first blob size ratio)."""
    merged = ndimage.binary_dilation(ink, iterations=merge_px)
    labels, n = ndimage.label(merged)
    if n == 0:
        return ink, 0.0
    sizes = ndimage.sum(ink, labels, index=range(1, n + 1))  # ink pixels per blob
    order = np.argsort(sizes)[::-1]
    keep = order[0] + 1
    second_ratio = float(sizes[order[1]] / sizes[order[0]]) if n > 1 else 0.0
    return ink & (labels == keep), second_ratio


def process(path: Path, size: int = 1024, fill: float = 0.8,
            merge_frac: float = 0.012) -> tuple[Image.Image, float]:
    gray = np.asarray(Image.open(path).convert("L"))
    ink = binarize(gray)
    if not ink.any():
        raise ValueError(f"{path.name}: no ink found")
    merge_px = max(2, int(round(merge_frac * max(gray.shape))))
    mask, second_ratio = main_figure_mask(ink, merge_px)

    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    # Keep original gray values inside the figure's box, but whiten ink outside the main blob.
    crop = gray[y0:y1, x0:x1].copy()
    crop[ink[y0:y1, x0:x1] & ~mask[y0:y1, x0:x1]] = 255

    h, w = crop.shape
    side = int(round(max(h, w) / fill))
    canvas = np.full((side, side), 255, dtype=np.uint8)
    oy, ox = (side - h) // 2, (side - w) // 2
    canvas[oy:oy + h, ox:ox + w] = crop

    out = Image.fromarray(canvas).resize((size, size), Image.LANCZOS).convert("RGB")
    return out, second_ratio


def find_source(name: str, raw_dir: Path) -> Path | None:
    stem = Path(name.strip()).stem
    hits = list(raw_dir.rglob(f"{stem}.png"))
    return hits[0] if hits else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selection", type=Path, default=Path("dataset/selection.txt"))
    ap.add_argument("--raw", type=Path, default=Path("dataset/candidates/raw"))
    ap.add_argument("--out", type=Path, default=Path("dataset/images"))
    ap.add_argument("--manifest", type=Path, default=Path("dataset/manifest.csv"))
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--fill", type=float, default=0.8, help="Share of the frame the figure fills.")
    ap.add_argument("--review-ratio", type=float, default=0.25,
                    help="Flag images whose 2nd-largest blob is at least this share of the largest.")
    args = ap.parse_args()

    names = [n for n in args.selection.read_text().splitlines() if n.strip() and not n.startswith("#")]
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in names:
        src = find_source(name, args.raw)
        if src is None:
            print(f"MISSING  {name} (not found under {args.raw})")
            continue
        img, ratio = process(src, args.size, args.fill)
        dst = args.out / f"{src.stem}.png"
        img.save(dst)
        ref = parse_name(src.stem)
        review = "second figure dropped?" if ratio >= args.review_ratio else ""
        flag = "REVIEW" if review else "ok    "
        print(f"{flag}  {src.stem}  (2nd blob ratio {ratio:.2f})")
        rows.append({
            "file_name": dst.name,
            "patent": ref.patent if ref else "",
            "issue_date": ref.issue_date if ref else "",
            "sheet": ref.sheet if ref else "",
            "source_url": ref.url if ref else "",
            "copyright_notice_checked": "",  # fill "no notice" after checking the patent text
            "review": review,
        })

    with args.manifest.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["file_name"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n{len(rows)} images -> {args.out}, manifest -> {args.manifest}")


if __name__ == "__main__":
    main()
