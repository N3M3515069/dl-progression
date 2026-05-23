# WGAN-GP — MNIST

A from-scratch implementation of Wasserstein GAN with Gradient Penalty (WGAN-GP) trained on MNIST using PyTorch, following the architecture from [Gulrajani et al. 2017](https://arxiv.org/abs/1704.00028).

Built as a direct follow-up to DCGAN to replace BCE loss with a theoretically grounded distance metric and eliminate the need for careful loss balancing.

---

## Results

### Early training (Step 37) — already recognizable digits
By step 37, G was producing diverse, readable digits across multiple classes with no mode collapse.

![Early training](assets/early_training.png)

### Final output (Step 89)
Sharp, varied digits visually comparable to real MNIST samples.

![Final output](assets/final_output.png)

---

## DCGAN vs WGAN-GP — Comparison

|  | DCGAN | WGAN-GP |
|---|---|---|
| Loss function | BCELoss | Wasserstein distance + gradient penalty |
| Output activation (D/C) | Sigmoid | None (unbounded critic scores) |
| Discriminator role | Binary classifier | Critic estimating Wasserstein distance |
| Normalization | BatchNorm | InstanceNorm |
| Training ratio | 1:1 (D:G) | 5:1 (critic:G) |
| Gradient constraint | None | Gradient penalty (λ=10) on interpolated samples |
| Optimizer betas | β1=0.5, β2=0.999 | β1=0.0, β2=0.9 |

---

## What WGAN-GP fixes over DCGAN

- **Meaningful loss metric** — critic loss approximates Wasserstein distance, so loss curves are actually interpretable. Negative critic loss = positive W-distance = critic successfully separating real from fake
- **No sigmoid saturation** — BCE gradients vanish when D is too confident; Wasserstein loss provides useful gradients even when the critic is strong
- **Gradient penalty over weight clipping** — original WGAN enforced the Lipschitz constraint via weight clipping, which causes capacity underuse. GP directly penalizes the gradient norm of the critic on interpolated samples instead
- **InstanceNorm replaces BatchNorm** — BatchNorm creates dependencies between samples in a batch, which corrupts the per-sample gradient computation that GP requires

---

## Training observations

Critic loss started at -111 (epoch 0) and stabilized to the -5 to -9 range from epoch 2 onwards — magnitude shrinking indicates the critic and generator settled into equilibrium rather than the critic running away.

Generator loss trended upward from ~63 to ~77 across 10 epochs, reflecting steady improvement in fooling the critic throughout training.

No mode collapse observed at any point. Compare to Vanilla GAN where D_loss crashed to ~0.09 while G_loss stayed high, confirming D dominance and permanent collapse.

---

## Architecture

### Critic
Input: `(N, 1, 64, 64)` grayscale image

Conv2d(1, 16, 4, 2, 1)       → (N, 16, 32, 32)    LeakyReLU(0.2)
Conv2d(16, 32, 4, 2, 1)      → (N, 32, 16, 16)    InstanceNorm → LeakyReLU(0.2)
Conv2d(32, 64, 4, 2, 1)      → (N, 64, 8, 8)      InstanceNorm → LeakyReLU(0.2)
Conv2d(64, 128, 4, 2, 1)     → (N, 128, 4, 4)     InstanceNorm → LeakyReLU(0.2)
Conv2d(128, 1, 4, 2, 0)      → (N, 1, 1, 1)       (no activation)

### Generator
Input: `(N, 100, 1, 1)` noise vector

ConvTranspose2d(100, 256, 4, 1, 0)  → (N, 256, 4, 4)    BatchNorm → ReLU
ConvTranspose2d(256, 128, 4, 2, 1)  → (N, 128, 8, 8)    BatchNorm → ReLU
ConvTranspose2d(128, 64, 4, 2, 1)   → (N, 64, 16, 16)   BatchNorm → ReLU
ConvTranspose2d(64, 32, 4, 2, 1)    → (N, 32, 32, 32)   BatchNorm → ReLU
ConvTranspose2d(32, 1, 4, 2, 1)     → (N, 1, 64, 64)    Tanh

---

## Training

| Hyperparameter | Value |
|---|---|
| Epochs | 10 |
| Batch size | 64 |
| Noise dim (z) | 100 |
| Learning rate | 1e-4 |
| Optimizer | Adam (β1=0.0, β2=0.9) |
| Critic iterations per G step | 5 |
| Gradient penalty λ | 10 |
| Image size | 64×64 |
| Weight init | Normal(0, 0.02) |

---

## Stack
Python, PyTorch, torchvision, TensorBoard, Matplotlib
