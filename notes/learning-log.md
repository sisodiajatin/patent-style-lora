# Learning log

One entry per work session: what you did, what you learned, what surprised you.

## Phase 1: concepts
- [ ] Diffusion: timesteps, noise schedule, sampler
- [ ] Latent diffusion and the VAE (why thin lines are hard)
- [ ] Text encoders, UNet, cross-attention
- [ ] Classifier-free guidance
- [ ] LoRA math: W + (alpha / rank) × B × A
- [ ] Overfitting with 20 images

## Sessions

## Run 01 (2026-10-27): r16, lr 1e-4, 1500 steps, UNet only
- Kettle and smartwatch switched to single centered line drawings only at steps 1250-1500: still improving when training stopped.
- Desk lamp stayed a photo at every checkpoint; drone stayed a wallpaper pattern.
- Strength test (0.8 / 1.2 / 1.6): higher strength adds hatching texture everywhere but never produces a white page; 1.6 breaks down.
- Diagnosis: (1) undertrained, (2) trigger weakly bound (UNet-only), (3) mid-tone drift: backgrounds peach/pink, a known
  diffusion limitation for mostly-white images.
- Run 02 changes: noise_offset 0.1, train_text_encoder, 3000 steps.

## Run 02 (2026-10-27): r16, lr 1e-4, 3000 steps, + text encoder, + noise_offset 0.1
- Every probe prompt (drone, kettle, smartwatch, desk lamp) is a clean single-object line drawing on a pure white page
  from step 500 on. Drone, smartwatch and desk lamp were not in the training set: the style generalizes.
- Picked up the design-patent convention of dashed lines for unclaimed parts (smartwatch band).
- Steps 1000-1500: best balance of crisp lines and correct subjects.
- Steps 2500-3000: overfitting. The quadcopter grows six arms, the watch turns almost entirely dashed, lamp lines go faint.
- Takeaway: noise offset fixed the background; text-encoder training fixed trigger binding. Picked checkpoint-1500 for eval.

## Eval (2026-10-27): run02 checkpoint-1500, 10 subjects x 3 seeds
- LoRA @ 0.7 vs prompt-only (B), unseen subjects: white border 0.440 -> 0.998, gray share 0.241 -> 0.038,
  CLIP subject 0.301 -> 0.316. Seen-type subjects: border 0.151 -> 0.999, gray 0.186 -> 0.023.
- Unseen gadgets score the same as seen object types: the style generalizes.
- Strength 1.0 whitens slightly more but CLIP subject drops to 0.308: 0.7 is the default.
- Trigger + style words (D) ~= trigger alone (C@0.7).
- Similarity to training set 0.60 -> 0.68: consistent with style learning; still check grids for copying,
  fake text and extra objects, which these metrics don't measure.
