# Patent-Drawing Style LoRA

Train a small LoRA on about 20 US patent drawings so that adding `ptnt_style` to a prompt turns any
gadget into a black-and-white patent figure. Then evaluate it honestly and host a demo.

> Results grid goes here once you have it: base vs. prompt-only vs. LoRA at fixed seeds.

## Workflow

| Step | Where | Command / file | Time |
| --- | --- | --- | --- |
| 1. Set up | Laptop | see *Setup* below | 20 min |
| 2. Fetch candidates | Laptop | `python scripts/fetch_candidates.py --shards 0 120 240` | 10 min |
| 3. Hand-pick 20 | Laptop | browse `dataset/candidates/sheets/`, list picks in `dataset/selection.txt` | 1-2 hrs |
| 4. Clean images | Laptop | `python scripts/prepare_data.py` | 1 min |
| 5. Draft captions | Laptop | `python scripts/caption.py draft` | 5 min |
| 6. Fix captions | Laptop | edit `dataset/captions.csv` by hand | 30 min |
| 7. Build metadata | Laptop | `python scripts/caption.py build` | seconds |
| 8. Push to GitHub | Laptop | `git add -A && git commit && git push` | 1 min |
| 9. Baselines | Colab L4 | `notebooks/03_eval_colab.ipynb`, baselines section | 15 min |
| 10. Train | Colab L4 | `notebooks/02_train_colab.ipynb` | 1-2 hrs |
| 11. Pick checkpoint | Colab L4 | last section of notebook 02 | 10 min |
| 12. Full eval | Colab L4 | `notebooks/03_eval_colab.ipynb` | 30-40 min |
| 13. Publish + demo | Colab + HF | Hub upload cell in notebook 03, then `demo/` to a ZeroGPU Space | 30 min |

Laptop steps need no GPU (BLIP captioning runs on CPU in a few minutes for 20 images).
Colab steps need the L4 runtime and use roughly 5-8 compute units in total for one run.

## Setup (Windows laptop)

```powershell
cd patent-drawing-lora
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-local.txt
# Only if you want GPU on the RTX 5060 (Blackwell needs CUDA 12.8+ builds):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
pytest -q   # data-prep and metric tests, CPU only
```

## Picking images (step 3)

Open the contact sheets and pick drawings that are:

- one clear object per sheet (multi-figure sheets get the largest figure kept, but check)
- varied: tools, electronics, furniture, kitchenware, wearables, vehicles; at most 2 per category
- a mix of views (about half perspective) and shading styles (hatching, stippling, plain outline)

Write the picked file stems in `dataset/selection.txt`, one per line, for example
`USD0806350-20180102-D00000`. Pick ~25 so you can drop the ones that clean up badly.

After `prepare_data.py`, look at every image in `dataset/images/`. Delete any that still show labels,
lost parts of the object, or were flagged `REVIEW`, and remove their line from `selection.txt`.

**Rights check:** open each patent's `source_url` from `dataset/manifest.csv` and confirm the text has
no copyright notice (37 CFR 1.71(d)); then write `no notice` in `copyright_notice_checked`.

## Captions (steps 5-7)

`captions.csv` has one row per image: `object` (e.g. `a cordless power drill`) and `view`
(perspective / front / side / top / rear / bottom / exploded view). Describe the object and the view only.
Never add style words such as "line art" or "black and white": the trigger word must carry the style.

`build` produces lines like `ptnt_style, side view of a cordless power drill` in `dataset/images/metadata.jsonl`.

## Repo layout

```
configs/train_sdxl.yaml       every training hyperparameter (copied next to each run's checkpoints)
dataset/selection.txt         your hand-picked sheet IDs
dataset/images/               20 cleaned 1024px PNGs + metadata.jsonl (public domain, committed)
dataset/manifest.csv          patent number, date, sheet, source URL, rights check
dataset/captions.csv          editable captions (object + view)
scripts/fetch_candidates.py   download DeepPatent shards + contact sheets
scripts/prepare_data.py       isolate main figure, strip labels, pad, resize
scripts/caption.py            BLIP draft captions -> metadata.jsonl
scripts/eval.py               fixed-seed eval: 4 conditions, grids, scores
scripts/pixel_metrics.py      color leakage, border whiteness, gray share
notebooks/02_train_colab.ipynb   training on a Colab L4, resumable from Drive
notebooks/03_eval_colab.ipynb    baselines, full eval, Hub upload
demo/                         Gradio app for a ZeroGPU Space + model card
tests/                        CPU tests for data prep and metrics
```

## Credits

Drawings from US design patents via [DeepPatent](https://openaccess.thecvf.com/content/WACV2022/html/Kucer_DeepPatent_Large_Scale_Patent_Drawing_Recognition_and_Retrieval_WACV_2022_paper.html)
(Kucer et al., WACV 2022). Training uses the Hugging Face Diffusers SDXL LoRA example script.
