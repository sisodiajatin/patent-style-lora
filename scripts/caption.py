"""Draft and build captions for the training images.

Two steps, with a hand edit in between:

  1. draft  - runs BLIP on every image in dataset/images and writes
              dataset/captions.csv with columns: file_name, object, view.
              BLIP's style words ("a drawing of", "black and white") are stripped,
              because the trigger word should carry the style, not the caption.
  2. (you)  - open captions.csv and fix every row: correct the object and set
              the view to one of: perspective view, front view, side view,
              top view, rear view, bottom view, exploded view.
  3. build  - writes dataset/images/metadata.jsonl in the format the Diffusers
              trainer reads:  {"file_name": "...png", "text": "ptnt_style, front view of a kettle"}

Usage:
    python scripts/caption.py draft
    python scripts/caption.py build
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

TRIGGER = "ptnt_style"
VIEWS = {"perspective view", "front view", "side view", "top view",
         "rear view", "bottom view", "exploded view"}

# Style words the captioner adds that must NOT end up in captions.
_STYLE_PATTERNS = [
    r"^arafed\s+", r"^there is\s+", r"^this is\s+",
    r"\b((a|an)\s+)?(black and white\s+|line\s+|technical\s+|simple\s+)*"
    r"(drawing|sketch|illustration|diagram|image|picture|outline|line art)\s+of\s+",
    r"\bblack and white\b", r"\bline (drawing|art)\b", r"\bin (a|the) style of [\w\s]+",
    r"\bon a white background\b", r"\bwith a white background\b",
    r"\b(patent|technical) (drawing|illustration)\b",
]


def clean_caption(text: str) -> str:
    """Strip style words so only the object description is left."""
    t = text.strip().lower()
    for pat in _STYLE_PATTERNS:
        t = re.sub(pat, "", t).strip()
    t = re.sub(r"\s{2,}", " ", t).strip(" ,.")
    if t and not re.match(r"^(a|an|the|two|three|pair|set)\b", t):
        t = ("an " if t[0] in "aeiou" else "a ") + t
    return t


def build_text(obj: str, view: str) -> str:
    view = (view or "perspective view").strip().lower()
    if view not in VIEWS:
        raise ValueError(f"unknown view '{view}', use one of {sorted(VIEWS)}")
    return f"{TRIGGER}, {view} of {obj.strip()}"


def cmd_draft(images: Path, out_csv: Path, model: str) -> None:
    import torch
    from PIL import Image
    from transformers import pipeline

    device = 0 if torch.cuda.is_available() else -1
    captioner = pipeline("image-to-text", model=model, device=device)
    files = sorted(p for p in images.glob("*.png"))
    rows = []
    for f in files:
        raw = captioner(Image.open(f).convert("RGB"), max_new_tokens=30)[0]["generated_text"]
        rows.append({"file_name": f.name, "object": clean_caption(raw), "view": "perspective view",
                     "blip_raw": raw})
        print(f"{f.name}: {rows[-1]['object']}   (raw: {raw})")
    with out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["file_name", "object", "view", "blip_raw"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nDrafted {len(rows)} captions -> {out_csv}. Edit object + view for every row, then run `build`.")


def cmd_build(images: Path, in_csv: Path) -> None:
    with in_csv.open() as fh:
        rows = list(csv.DictReader(fh))
    missing = [r["file_name"] for r in rows if not (images / r["file_name"]).exists()]
    if missing:
        raise SystemExit(f"captions.csv lists files not in {images}: {missing}")
    uncaptioned = sorted({p.name for p in images.glob("*.png")} - {r["file_name"] for r in rows})
    if uncaptioned:
        raise SystemExit(f"images without a caption row: {uncaptioned}")
    out = images / "metadata.jsonl"
    with out.open("w") as fh:
        for r in rows:
            if not r["object"].strip():
                raise SystemExit(f"empty object for {r['file_name']}")
            fh.write(json.dumps({"file_name": r["file_name"], "text": build_text(r["object"], r["view"])}) + "\n")
    print(f"Wrote {len(rows)} captions -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["draft", "build"])
    ap.add_argument("--images", type=Path, default=Path("dataset/images"))
    ap.add_argument("--csv", type=Path, default=Path("dataset/captions.csv"))
    ap.add_argument("--model", default="Salesforce/blip-image-captioning-large")
    args = ap.parse_args()
    if args.step == "draft":
        cmd_draft(args.images, args.csv, args.model)
    else:
        cmd_build(args.images, args.csv)


if __name__ == "__main__":
    main()
