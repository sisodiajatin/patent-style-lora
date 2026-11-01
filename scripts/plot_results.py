"""Draw the README results chart (light + dark SVG) from the run-02 evaluation.

Numbers are the unseen-object means from eval/results/lora_run02_ckpt1500/summary.csv
(6 objects x 3 seeds). Three metrics on different scales, so three small panels, never one
dual axis. Baselines are neutral gray; only the LoRA bar carries color.

Usage:  python scripts/plot_results.py      -> docs/results-light.svg, docs/results-dark.svg
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CONDITIONS = ["Plain prompt", '"Patent drawing"\nprompt', "LoRA @ 0.7"]
PANELS = [
    # title, subtitle, values, formatter, axis max
    ("White page  ↑", "share of edge pixels near-white", [0.001, 0.440, 0.998], lambda v: f"{v:.1%}", 1.0),
    ("Gray shading  ↓", "share of mid-gray pixels", [0.746, 0.241, 0.038], lambda v: f"{v:.1%}", 1.0),
    ("Subject kept  ↑", "CLIP score vs. object name", [0.284, 0.301, 0.316], lambda v: f"{v:.3f}", 0.35),
]

THEMES = {
    "light": dict(surface="#fcfcfb", text="#0b0b0b", text2="#52514e", muted="#898781",
                  baseline="#c3c2b7", accent="#2a78d6", neutral="#898781"),
    "dark": dict(surface="#1a1a19", text="#f0efec", text2="#c3c2b7", muted="#898781",
                 baseline="#383835", accent="#3987e5", neutral="#6f6e68"),
}


def draw(theme: str, out: Path) -> None:
    t = THEMES[theme]
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "path"})
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.1), sharey=True)
    fig.patch.set_facecolor(t["surface"])
    colors = [t["neutral"], t["neutral"], t["accent"]]
    ys = range(len(CONDITIONS))[::-1]

    for i, (ax, (title, sub, vals, fmt, vmax)) in enumerate(zip(axes, PANELS)):
        ax.set_facecolor(t["surface"])
        ax.barh(list(ys), vals, height=0.56, color=colors, edgecolor=t["surface"], linewidth=2)
        for y, v, c in zip(ys, vals, colors):
            ax.text(v + vmax * 0.02, y, fmt(v), va="center", ha="left", fontsize=10.5,
                    color=t["text"], fontweight="bold" if c == t["accent"] else "normal")
        ax.set_xlim(0, vmax * 1.28)
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=t["text"], pad=18)
        ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=9, color=t["text2"], va="bottom")
        ax.set_xticks([])
        for side in ("top", "right", "bottom"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_color(t["baseline"])
        ax.tick_params(axis="y", length=0, labelsize=10, colors=t["text"], pad=6)
        if i == 0:
            ax.set_yticks(list(ys), CONDITIONS)

    fig.suptitle("The LoRA gives a white page and clean lines without losing the subject",
                 x=0.012, ha="left", fontsize=13.5, fontweight="bold", color=t["text"], y=1.06)
    fig.text(0.012, 0.925, "Unseen objects (drone, smartwatch, VR headset, scooter, robot arm, foldable phone), "
             "3 seeds each, same settings for every condition",
             fontsize=9.5, color=t["text2"], ha="left")
    fig.subplots_adjust(left=0.14, right=0.99, top=0.70, bottom=0.04, wspace=0.12)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=t["surface"], bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1] / "docs"
    for theme in THEMES:
        draw(theme, root / f"results-{theme}.svg")
        print("wrote", root / f"results-{theme}.svg")
