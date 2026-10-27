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
