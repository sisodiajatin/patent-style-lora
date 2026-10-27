# patent-style-lora

Train a small SDXL LoRA on 24 US design-patent drawings so that adding `ptnt_style` to a prompt turns any
gadget into a black-and-white patent figure. Then evaluate it honestly against plain prompting and host a demo.

## Status

Work in progress. Dataset built, baselines rendered, first training run done.

**Baseline finding.** SDXL already draws decent line art when prompted with "patent drawing … black and white
line art", but gets the page wrong: beige "old paper" backgrounds, fake text and title blocks, several objects
per image, scenery. The LoRA's job is the page: pure white, one centered object, no text.

**Run 01** (rank 16, LR 1e-4, 1,500 steps, UNet only): partial. Kettle and smartwatch prompts switched to single
centered line drawings only in the last checkpoints, still on peach/pink backgrounds; desk lamp stayed a photo.
Raising LoRA strength added hatching texture but never a white page. Diagnosis: undertrained, weak trigger
binding, and the mid-tone drift diffusion models show on mostly-white images.

**Run 02** (in progress): adds noise offset 0.1, text-encoder training, and 3,000 steps.
Full notes: [`notes/learning-log.md`](notes/learning-log.md). Configs for each run are in [`configs/`](configs/).

## Run it entirely in Google Colab

**One-time setup (2 minutes):** in Google Drive, create a folder named `patent-lora` in My Drive and upload
`patent-drawing-lora.zip` into it. Every notebook unpacks it to `MyDrive/patent-lora/repo` on first run and works
from there, so your picks, captions, checkpoints and results all stay on Drive.

Open each notebook in Colab with **File → Upload notebook** (or open it from `MyDrive/patent-lora/repo/notebooks/`).

| # | Notebook | Runtime | What you do | Time |
| --- | --- | --- | --- | --- |
| 1 | `01_dataset_colab.ipynb` | **CPU** | download sheets, tick ~25 in the picker, clean, review, fix captions | 1.5-3 hrs (mostly picking) |
| 2 | `03_eval_colab.ipynb`, *Baselines* section only | **L4** | render the no-LoRA baselines before training | 15 min |
| 3 | `02_train_colab.ipynb` | **L4** | train, then compare checkpoints and pick one | 1-2 hrs |
| 4 | `03_eval_colab.ipynb`, rest | **L4** | full eval with your checkpoint, publish to the Hub | 40 min |
| 5 | `demo/` | Hugging Face Space | upload `app.py`, `requirements.txt`, `SPACE_README.md` (as `README.md`) to a ZeroGPU Space | 20 min |

### Using the Colab extension in VS Code

The notebooks also run from VS Code with Google's Colab extension:

1. Open this project folder in VS Code and open a notebook from `notebooks/`.
2. **Select Kernel → Colab → New Colab Server**, and choose the machine type (CPU for notebook 01, L4 for 02 and 03).
3. Run the cells as usual. Drive mounting and the picker/caption widgets work. The code runs on Colab's machine,
   not your laptop, so the project still has to be on Drive (the zip in `MyDrive/patent-lora/`).

Differences from the browser: the Hugging Face upload cell asks you to paste your token instead of reading
Colab Secrets, and if the last "disconnect" cell can't stop the runtime, use **Command Palette → Colab: Remove Server**.

Expect roughly 5-8 compute units for one full run. Each GPU notebook ends with a cell that disconnects
the runtime; run it so an idle GPU doesn't keep using units.

## Optional: run the data steps on a laptop instead

| Step | Command |
| --- | --- |
| Fetch candidates | `python scripts/fetch_candidates.py --shards 0 120 240` (contact sheets in `dataset/candidates/sheets/`) |
| Pick | list file stems in `dataset/selection.txt` |
| Clean | `python scripts/prepare_data.py` |
| Draft captions | `python scripts/caption.py draft`, then edit `dataset/captions.csv` |
| Build metadata | `python scripts/caption.py build` |

Then copy the project folder to `MyDrive/patent-lora/repo` and continue with the training notebook.

### Laptop setup (Windows)

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

## Picking images

In the picker (or the contact sheets on a laptop), pick drawings that are:

- one clear object per sheet (multi-figure sheets get the largest figure kept, but check)
- varied: tools, electronics, furniture, kitchenware, wearables, vehicles; at most 2 per category
- a mix of views (about half perspective) and shading styles (hatching, stippling, plain outline)

Pick ~25 so you can drop the ones that clean up badly. Picks are stored in `dataset/selection.txt`.

After cleaning, review every image (notebook 01, step 4). Remove any that still show labels,
lost parts of the object, or were flagged as possibly having a second figure.

**Rights check:** open each patent's `source_url` from `dataset/manifest.csv` and confirm the text has
no copyright notice (37 CFR 1.71(d)); then write `no notice` in `copyright_notice_checked`.

## Captions

The caption editor (notebook 01, step 7) stores one row per image in `captions.csv`: `object` (e.g. `a cordless power drill`) and `view`
(perspective / front / side / top / rear / bottom / exploded view). Describe the object and the view only.
Never add style words such as "line art" or "black and white": the trigger word must carry the style.

Saving produces lines like `ptnt_style, side view of a cordless power drill` in `dataset/images/metadata.jsonl`.

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
scripts/colab_ui.py           notebook widgets: picker, review, caption editor
notebooks/01_dataset_colab.ipynb build the dataset in Colab (CPU runtime)
notebooks/02_train_colab.ipynb   training on a Colab L4, resumable from Drive
notebooks/03_eval_colab.ipynb    baselines, full eval, Hub upload
demo/                         Gradio app for a ZeroGPU Space + model card
tests/                        CPU tests for data prep and metrics
```

## Credits

Drawings from US design patents via [DeepPatent](https://openaccess.thecvf.com/content/WACV2022/html/Kucer_DeepPatent_Large_Scale_Patent_Drawing_Recognition_and_Retrieval_WACV_2022_paper.html)
(Kucer et al., WACV 2022). Training uses the Hugging Face Diffusers SDXL LoRA example script.
