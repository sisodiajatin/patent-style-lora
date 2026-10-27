"""Notebook widgets for building the dataset inside Google Colab (or Jupyter).

  picker(...)          browse drawing sheets as thumbnails, tick the ones you want;
                       every tick is saved to selection.txt immediately.
  review(...)          look over the cleaned 1024px images and remove bad ones.
  caption_editor(...)  fix the object + view for every image, then save, which writes
                       captions.csv and the trainer's metadata.jsonl.

All three save to files on Google Drive, so nothing is lost if Colab disconnects.
"""
from __future__ import annotations

import csv
import io
import sys
from functools import lru_cache
from pathlib import Path

import ipywidgets as W
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from caption import VIEWS, cmd_build  # noqa: E402

VIEW_OPTIONS = ["perspective view", "front view", "side view", "top view",
                "rear view", "bottom view", "exploded view"]
assert set(VIEW_OPTIONS) == VIEWS


@lru_cache(maxsize=512)
def _thumb_bytes(path: str, size: int) -> bytes:
    im = Image.open(path).convert("L")
    im.thumbnail((size, size))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def _read_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [l.strip() for l in path.read_text().splitlines() if l.strip() and not l.startswith("#")]


def _write_selection(path: Path, stems: list[str]) -> None:
    header = "# Picked in the Colab picker. One drawing-sheet ID per line.\n"
    path.write_text(header + "".join(f"{s}\n" for s in stems))


def picker(raw_dir: Path, selection_file: Path, per_page: int = 24, thumb: int = 170,
           cols: int = 6) -> W.VBox:
    """Paged thumbnail grid with a checkbox per sheet. Ticks are saved to selection_file at once."""
    files = sorted(Path(raw_dir).rglob("*.png"))
    if not files:
        raise FileNotFoundError(f"no PNGs under {raw_dir}; run the download cell first")
    selected = _read_lines(selection_file)  # keeps pick order
    pages = max(1, -(-len(files) // per_page))
    state = {"page": 0}

    count = W.HTML()
    page_label = W.Label()
    grid = W.GridBox(layout=W.Layout(grid_template_columns=f"repeat({cols}, {thumb + 16}px)",
                                     grid_gap="6px"))
    prev_b, next_b = W.Button(description="◀ Prev"), W.Button(description="Next ▶")
    jump = W.BoundedIntText(value=1, min=1, max=pages, description="Page", layout=W.Layout(width="160px"))

    def update_count():
        n = len(selected)
        color = "green" if 20 <= n <= 30 else ("orange" if n else "gray")
        count.value = f"<b style='color:{color}'>{n} picked</b> (aim for about 25, keep the best 20)"

    def on_toggle(stem):
        def handler(change):
            if change["new"] and stem not in selected:
                selected.append(stem)
            elif not change["new"] and stem in selected:
                selected.remove(stem)
            _write_selection(selection_file, selected)
            update_count()
        return handler

    def render():
        p = state["page"]
        tiles = []
        for f in files[p * per_page:(p + 1) * per_page]:
            img = W.Image(value=_thumb_bytes(str(f), thumb), format="png",
                          layout=W.Layout(width=f"{thumb}px", height=f"{thumb}px", object_fit="contain"))
            cb = W.Checkbox(value=f.stem in selected, description=f.stem[3:10], indent=False,
                            layout=W.Layout(width=f"{thumb}px"))
            cb.observe(on_toggle(f.stem), names="value")
            tiles.append(W.VBox([img, cb], layout=W.Layout(border="1px solid #ddd", padding="4px")))
        grid.children = tiles
        page_label.value = f"page {p + 1} of {pages} · {len(files)} sheets"
        jump.value = p + 1

    def go(delta=None, to=None):
        state["page"] = (to if to is not None else state["page"] + delta) % pages
        render()

    prev_b.on_click(lambda _: go(-1))
    next_b.on_click(lambda _: go(+1))
    jump.observe(lambda c: go(to=c["new"] - 1) if c["new"] - 1 != state["page"] else None, names="value")

    update_count()
    render()
    tip = W.HTML("<small>Tick drawings with <b>one clear object</b>; mix object types and views; "
                 "skip sheets that are mostly text or many small figures.</small>")
    return W.VBox([W.HBox([prev_b, next_b, jump, page_label]), count, tip, grid])


def review(images_dir: Path, selection_file: Path, manifest: Path, thumb: int = 220,
           cols: int = 5) -> W.VBox:
    """Grid of cleaned images; tick 'remove' on bad ones, then Apply."""
    images_dir = Path(images_dir)
    files = sorted(images_dir.glob("*.png"))
    flagged = set()
    if Path(manifest).exists():
        with open(manifest) as fh:
            flagged = {r["file_name"] for r in csv.DictReader(fh) if r.get("review")}
    boxes = {}
    tiles = []
    for f in files:
        img = W.Image(value=_thumb_bytes(str(f), thumb), format="png",
                      layout=W.Layout(width=f"{thumb}px", height=f"{thumb}px"))
        note = " ⚠ check: 2nd figure dropped" if f.name in flagged else ""
        cb = W.Checkbox(value=False, description=f"remove{note}", indent=False)
        boxes[f] = cb
        tiles.append(W.VBox([W.Label(f.stem[3:10]), img, cb],
                            layout=W.Layout(border="1px solid #ddd", padding="4px")))
    grid = W.GridBox(tiles, layout=W.Layout(grid_template_columns=f"repeat({cols}, {thumb + 16}px)",
                                            grid_gap="6px"))
    out = W.HTML(f"{len(files)} cleaned images. Remove any with leftover labels, cut-off parts, or two objects.")
    apply_b = W.Button(description="Apply removals", button_style="warning")

    def apply(_):
        gone = [f for f, cb in boxes.items() if cb.value]
        for f in gone:
            f.unlink(missing_ok=True)
        stems = {f.stem for f in gone}
        _write_selection(selection_file, [s for s in _read_lines(selection_file) if s not in stems])
        if Path(manifest).exists():
            with open(manifest) as fh:
                rows = list(csv.DictReader(fh))
            keep = [r for r in rows if Path(r["file_name"]).stem not in stems]
            with open(manifest, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(keep)
        left = len(list(images_dir.glob("*.png")))
        for f in gone:
            boxes[f].disabled = True
            boxes[f].description = "removed"
        out.value = f"Removed {len(gone)}. <b>{left} images left</b>" + (
            "" if left >= 20 else " (fewer than 20: pick a few more and re-run the cleaning cell)")

    apply_b.on_click(apply)
    return W.VBox([out, grid, apply_b])


def caption_editor(images_dir: Path, captions_csv: Path, thumb: int = 200) -> W.VBox:
    """One row per image: thumbnail, object text, view dropdown. Save writes CSV + metadata.jsonl."""
    images_dir, captions_csv = Path(images_dir), Path(captions_csv)
    drafts = {}
    if captions_csv.exists():
        with captions_csv.open() as fh:
            drafts = {r["file_name"]: r for r in csv.DictReader(fh)}
    rows = []
    for f in sorted(images_dir.glob("*.png")):
        d = drafts.get(f.name, {})
        obj = W.Text(value=d.get("object", ""), placeholder="e.g. a cordless power drill",
                     description="object", layout=W.Layout(width="420px"))
        view = W.Dropdown(options=VIEW_OPTIONS,
                          value=d.get("view") if d.get("view") in VIEW_OPTIONS else "perspective view",
                          description="view", layout=W.Layout(width="260px"))
        hint = W.HTML(f"<small style='color:#777'>BLIP said: {d.get('blip_raw', '-')}</small>")
        rows.append((f, obj, view, d.get("blip_raw", "")))
        img = W.Image(value=_thumb_bytes(str(f), thumb), format="png",
                      layout=W.Layout(width=f"{thumb}px", height=f"{thumb}px"))
        rows[-1] = rows[-1] + (W.HBox([img, W.VBox([W.Label(f.name), obj, view, hint])],
                                      layout=W.Layout(border="1px solid #ddd", padding="4px")),)
    status = W.HTML("Describe the object and view only. No style words like 'line art' or 'black and white'.")
    save_b = W.Button(description="Save captions", button_style="success")

    def save(_):
        empty = [f.name for f, obj, *_ in rows if not obj.value.strip()]
        if empty:
            status.value = f"<b style='color:red'>Fill in the object for: {', '.join(empty)}</b>"
            return
        with captions_csv.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["file_name", "object", "view", "blip_raw"])
            w.writeheader()
            for f, obj, view, raw, _ in rows:
                w.writerow({"file_name": f.name, "object": obj.value.strip(), "view": view.value, "blip_raw": raw})
        try:
            cmd_build(images_dir, captions_csv)
        except SystemExit as e:  # cmd_build reports problems via SystemExit
            status.value = f"<b style='color:red'>{e}</b>"
            return
        status.value = (f"<b style='color:green'>Saved {len(rows)} captions</b> to {captions_csv.name} "
                        f"and metadata.jsonl. Next: the training notebook.")

    save_b.on_click(save)
    return W.VBox([status, *[r[-1] for r in rows], save_b])
