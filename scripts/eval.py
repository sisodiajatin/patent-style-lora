"""Fixed-seed evaluation: base vs. prompt-only vs. LoRA (strength sweep) vs. LoRA + style words.

Conditions (see the project plan, Evaluation section):
  A  base         "{view} of {object}"                                   LoRA off
  B  prompt-only  "patent drawing, {view} of {object}, black and white line art"   LoRA off
  C  lora@s       "ptnt_style, {view} of {object}"                       LoRA at each strength s
  D  lora+words   B's prompt with the trigger prepended                  LoRA at 0.7

Everything else (seed, steps, CFG, negative prompt, resolution) is held fixed
and read from eval/prompts.json, so runs are comparable.

Usage:
  # Phase 3, before training: baselines only (A and B)
  python scripts/eval.py --run baselines --baselines-only
  # After training: full eval with a checkpoint (local folder or Hub repo id)
  python scripts/eval.py --run lora_ckpt1000 --lora /content/drive/MyDrive/patent-lora/output/checkpoint-1000
  # On the 8GB laptop GPU add --offload
  # Add --clip for CLIP subject scores and the nearest-training-image check

Outputs in eval/results/<run>/: images/, grid_seed<N>.png, scores.csv, summary.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pixel_metrics import all_metrics  # noqa: E402

TRIGGER = "ptnt_style"
BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
FP16_VAE = "madebyollin/sdxl-vae-fp16-fix"


def conditions(strengths: list[float], baselines_only: bool) -> list[tuple[str, float]]:
    conds = [("A_base", 0.0), ("B_prompt", 0.0)]
    if not baselines_only:
        conds += [(f"C_lora@{s:g}", s) for s in strengths] + [("D_lora+words", 0.7)]
    return conds


def prompt_for(cond: str, obj: str, view: str) -> str:
    plain = f"{view} of {obj}"
    words = f"patent drawing, {plain}, black and white line art"
    if cond.startswith("A_"):
        return plain
    if cond.startswith("B_"):
        return words
    if cond.startswith("C_"):
        return f"{TRIGGER}, {plain}"
    if cond.startswith("D_"):
        return f"{TRIGGER}, {words}"
    raise ValueError(cond)


def make_grid(cells: dict[tuple[int, str], Image.Image], row_labels: list[str],
              col_labels: list[str], thumb: int = 256) -> Image.Image:
    """Rows = subjects, columns = conditions. Missing cells stay blank."""
    head, side = 28, 170
    grid = Image.new("RGB", (side + thumb * len(col_labels), head + thumb * len(row_labels)), "white")
    d = ImageDraw.Draw(grid)
    for j, c in enumerate(col_labels):
        d.text((side + j * thumb + 6, 8), c, fill=(0, 0, 0))
    for i, r in enumerate(row_labels):
        d.text((6, head + i * thumb + thumb // 2 - 6), r[:26], fill=(0, 0, 0))
        for j, c in enumerate(col_labels):
            im = cells.get((i, c))
            if im is not None:
                grid.paste(im.resize((thumb, thumb)), (side + j * thumb, head + i * thumb))
            d.rectangle([side + j * thumb, head + i * thumb, side + (j + 1) * thumb - 1,
                         head + (i + 1) * thumb - 1], outline=(220, 220, 220))
    return grid


def load_pipe(lora: str | None, offload: bool):
    import torch
    from diffusers import AutoencoderKL, StableDiffusionXLPipeline

    vae = AutoencoderKL.from_pretrained(FP16_VAE, torch_dtype=torch.float16)
    pipe = StableDiffusionXLPipeline.from_pretrained(BASE_MODEL, vae=vae, torch_dtype=torch.float16,
                                                     variant="fp16", use_safetensors=True)
    if offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    if lora:
        p = Path(lora)
        if p.is_dir():
            pipe.load_lora_weights(str(p), weight_name="pytorch_lora_weights.safetensors", adapter_name="patent")
        elif p.is_file():
            pipe.load_lora_weights(str(p.parent), weight_name=p.name, adapter_name="patent")
        else:  # Hub repo id
            pipe.load_lora_weights(lora, adapter_name="patent")
    return pipe


class ClipScorer:
    """Subject score (image vs. "a photo/drawing of <object>") and nearest training image similarity."""

    def __init__(self, train_dir: Path):
        import open_clip
        import torch

        self.torch = torch
        self.model, _, self.pre = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
        self.tok = open_clip.get_tokenizer("ViT-B-32")
        self.model.eval()
        files = sorted(train_dir.glob("*.png"))
        self.train = self._img_emb([Image.open(f).convert("RGB") for f in files]) if files else None

    def _img_emb(self, ims):
        with self.torch.no_grad():
            e = self.model.encode_image(self.torch.stack([self.pre(i) for i in ims]))
        return e / e.norm(dim=-1, keepdim=True)

    def score(self, im: Image.Image, obj: str) -> dict[str, float]:
        with self.torch.no_grad():
            t = self.model.encode_text(self.tok([obj]))
        t = t / t.norm(dim=-1, keepdim=True)
        e = self._img_emb([im])
        out = {"clip_subject": float((e @ t.T).item())}
        if self.train is not None:
            out["nn_train_sim"] = float((e @ self.train.T).max().item())
        return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="Name of this eval run (folder under eval/results).")
    ap.add_argument("--lora", help="Checkpoint folder, .safetensors file, or Hub repo id.")
    ap.add_argument("--baselines-only", action="store_true", help="Only conditions A and B (no LoRA needed).")
    ap.add_argument("--config", type=Path, default=Path("eval/prompts.json"))
    ap.add_argument("--out", type=Path, default=Path("eval/results"))
    ap.add_argument("--train-dir", type=Path, default=Path("dataset/images"))
    ap.add_argument("--offload", action="store_true", help="CPU offload for 8GB GPUs (slower).")
    ap.add_argument("--clip", action="store_true", help="Also compute CLIP scores (needs open_clip_torch).")
    args = ap.parse_args()

    if not args.baselines_only and not args.lora:
        ap.error("--lora is required unless --baselines-only")

    import torch

    cfg = json.loads(args.config.read_text())
    conds = conditions(cfg["lora_strengths"], args.baselines_only)
    run_dir = args.out / args.run
    (run_dir / "images").mkdir(parents=True, exist_ok=True)

    pipe = load_pipe(None if args.baselines_only else args.lora, args.offload)
    clip = ClipScorer(args.train_dir) if args.clip else None
    subjects = cfg["subjects"]
    rows = []

    for seed in cfg["seeds"]:
        cells = {}
        for i, s in enumerate(subjects):
            for cond, strength in conds:
                if not args.baselines_only:
                    pipe.set_adapters(["patent"], adapter_weights=[strength])
                prompt = prompt_for(cond, s["object"], s["view"])
                g = torch.Generator("cpu").manual_seed(seed)
                im = pipe(prompt, negative_prompt=cfg["negative_prompt"], num_inference_steps=cfg["steps"],
                          guidance_scale=cfg["guidance_scale"], height=cfg["resolution"],
                          width=cfg["resolution"], generator=g).images[0]
                safe = cond.replace("@", "_").replace("+", "_")
                im.save(run_dir / "images" / f"s{i:02d}_{safe}_seed{seed}.png")
                cells[(i, cond)] = im
                row = {"seed": seed, "subject": s["object"], "view": s["view"],
                       "seen_type": s["seen_type"], "condition": cond, "strength": strength,
                       "prompt": prompt, **all_metrics(im)}
                if clip:
                    row.update(clip.score(im, s["object"]))
                rows.append(row)
                print(f"seed {seed} | {s['object']:<22} | {cond:<14} | leak {row['color_leak']:.3f} "
                      f"border {row['border_white']:.3f} gray {row['gray_share']:.3f}")
        make_grid(cells, [f"{s['object']} ({s['view']})" for s in subjects],
                  [c for c, _ in conds]).save(run_dir / f"grid_seed{seed}.png")

    keys = list(rows[0].keys())
    with (run_dir / "scores.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    write_summary(rows, run_dir / "summary.csv")
    print(f"\nDone -> {run_dir}")


def write_summary(rows: list[dict], path: Path) -> None:
    """Mean of each numeric metric per condition, split by seen vs. unseen subjects."""
    metrics = [k for k in ("color_leak", "border_white", "gray_share", "clip_subject", "nn_train_sim") if k in rows[0]]
    groups: dict[tuple[str, bool], list[dict]] = {}
    for r in rows:
        groups.setdefault((r["condition"], r["seen_type"]), []).append(r)
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["condition", "subjects", "n", *metrics])
        for (cond, seen), rs in sorted(groups.items()):
            w.writerow([cond, "seen-type" if seen else "unseen", len(rs),
                        *[f"{sum(r[m] for r in rs) / len(rs):.4f}" for m in metrics]])


if __name__ == "__main__":
    main()
