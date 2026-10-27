"""CPU tests for data prep, captions, eval helpers and pixel metrics. Run: pytest -q"""
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from caption import build_text, clean_caption  # noqa: E402
from eval import conditions, make_grid, prompt_for  # noqa: E402
from pixel_metrics import all_metrics  # noqa: E402
from prepare_data import parse_name, process  # noqa: E402


def fake_sheet(path: Path, second_figure: bool = False) -> Path:
    """A drawing sheet like DeepPatent's: one hatched object, a 'FIG. 1' label, a sheet number."""
    im = Image.new("L", (1700, 2200), 255)
    d = ImageDraw.Draw(im)
    d.rectangle([450, 500, 1250, 1300], outline=0, width=6)          # body
    d.ellipse([650, 700, 1050, 1100], outline=0, width=6)            # dial
    for x in range(470, 640, 18):                                     # hatching
        d.line([x, 520, x + 120, 1280], fill=0, width=3)
    d.line([1250, 900, 1450, 900], fill=0, width=6)                   # handle
    font = ImageFont.load_default(size=70)
    d.text((760, 1700), "FIG. 1", fill=0, font=font)                  # label, far below the object
    d.text((780, 80), "1/3", fill=0, font=font)                       # sheet number
    if second_figure:
        d.rectangle([300, 1850, 1100, 2150], outline=0, width=6)      # big second figure
        for x in range(320, 1080, 25):
            d.line([x, 1860, x, 2140], fill=0, width=3)
    im.save(path)
    return path


def test_process_removes_labels_and_frames_object(tmp_path):
    out, ratio = process(fake_sheet(tmp_path / "USD0806350-20180102-D00000.png"))
    assert out.size == (1024, 1024) and out.mode == "RGB"
    assert ratio < 0.25, "labels are small, so the 2nd blob should be small"
    a = np.asarray(out.convert("L"))
    ys, xs = np.nonzero(a < 128)
    h, w = ys.max() - ys.min(), xs.max() - xs.min()
    assert 0.7 * 1024 <= max(h, w) <= 0.85 * 1024, "figure should fill ~80% of the frame"
    # Object box in the source is 1000 px wide (450..1450) and 800 tall: wider than tall after cropping
    assert w > h
    m = all_metrics(out)
    assert m["border_white"] > 0.999 and m["color_leak"] == 0.0


def test_process_flags_second_figure(tmp_path):
    _, ratio = process(fake_sheet(tmp_path / "USD0000001-20200101-D00001.png", second_figure=True))
    assert ratio >= 0.25


def test_parse_name():
    ref = parse_name("USD0806350-20180102-D00003")
    assert ref.patent == "D806350" and ref.issue_date == "2018-01-02" and ref.sheet == 3
    assert ref.url == "https://patents.google.com/patent/USD806350S"
    assert parse_name("random_file") is None


@pytest.mark.parametrize("raw,expected", [
    ("arafed drawing of a chair with a cushion", "a chair with a cushion"),
    ("a black and white drawing of a kettle", "a kettle"),
    ("a line drawing of an electric toothbrush on a white background", "an electric toothbrush"),
    ("a sketch of a pair of headphones", "a pair of headphones"),
    ("there is a technical drawing of a lamp", "a lamp"),
    ("umbrella", "an umbrella"),
])
def test_clean_caption(raw, expected):
    assert clean_caption(raw) == expected


def test_build_text():
    assert build_text("a kettle", "Side View ") == "ptnt_style, side view of a kettle"
    assert build_text("a kettle", "") == "ptnt_style, perspective view of a kettle"
    with pytest.raises(ValueError):
        build_text("a kettle", "isometric")


def test_eval_conditions_and_prompts():
    assert [c for c, _ in conditions([0.4, 1.0], baselines_only=True)] == ["A_base", "B_prompt"]
    full = conditions([0.4, 0.7, 1.0], baselines_only=False)
    assert [c for c, _ in full] == ["A_base", "B_prompt", "C_lora@0.4", "C_lora@0.7", "C_lora@1", "D_lora+words"]
    assert prompt_for("A_base", "a drone", "top view") == "top view of a drone"
    assert prompt_for("C_lora@0.7", "a drone", "top view") == "ptnt_style, top view of a drone"
    assert prompt_for("D_lora+words", "a drone", "top view").startswith("ptnt_style, patent drawing")
    assert "ptnt_style" not in prompt_for("B_prompt", "a drone", "top view")


def test_make_grid_size():
    cells = {(0, "A"): Image.new("RGB", (64, 64), "red"), (1, "B"): Image.new("RGB", (64, 64), "blue")}
    g = make_grid(cells, ["r0", "r1"], ["A", "B"], thumb=100)
    assert g.size == (170 + 200, 28 + 200)


def test_pixel_metrics():
    white = Image.new("RGB", (200, 200), "white")
    lines = white.copy()
    ImageDraw.Draw(lines).line([50, 100, 150, 100], fill="black", width=3)
    m = all_metrics(lines)
    assert m["color_leak"] == 0 and m["border_white"] == 1 and m["gray_share"] < 0.02

    colored = Image.new("RGB", (200, 200), (200, 60, 40))
    assert all_metrics(colored)["color_leak"] == 1.0 and all_metrics(colored)["border_white"] == 0.0

    gray = Image.new("RGB", (200, 200), (128, 128, 128))
    assert all_metrics(gray)["gray_share"] == 1.0 and all_metrics(gray)["color_leak"] == 0.0
