# CNN — FashionMNIST

Second stage of `dl-progression` — a from-scratch convolutional neural network 
classifying clothing items from FashionMNIST, building on the fully-connected 
approach from `Vanilla_NN_MNIST`.

## What this is

A CNN with two convolutional blocks (each with two Conv2d layers, ReLU, and 
MaxPool) followed by a flattening classifier head — architecture modeled on a 
mini VGG-style block structure. Trained on FashionMNIST, a harder 10-class 
classification task than plain MNIST (clothing items instead of digits, more 
intra-class visual variation).

## Architecture

- Input: 1×28×28 (single-channel grayscale)
- **Block 1:** Conv2d(1→10, k=3, pad=1) → ReLU → Conv2d(10→10, k=3, pad=1) → ReLU → MaxPool(2)
- **Block 2:** Conv2d(10→10, k=3, pad=1) → ReLU → Conv2d(10→10, k=3, pad=1) → ReLU → MaxPool(2)
- **Classifier:** Flatten → Linear(10×7×7 → 10)
- Two MaxPool(2) layers downsample 28×28 → 14×14 → 7×7, hence the 7×7 spatial 
  dimension feeding the classifier

## Training

- Loss: CrossEntropyLoss
- Optimizer: Adam, lr = 0.001
- Epochs: 5
- Batch size: 32, via `DataLoader` with shuffling on train set
- Proper mini-batch training (unlike the full-batch approach in `Vanilla_NN_MNIST`)

## Results

| Epoch | Train Loss | Train Acc | Test Loss | Test Acc |
|---|---|---|---|---|
| 0 | 0.537 | 0.804 | 0.404 | 0.855 |
| 1 | 0.366 | 0.868 | 0.378 | 0.864 |
| 2 | 0.331 | 0.881 | 0.339 | 0.878 |
| 3 | 0.309 | 0.889 | 0.327 | 0.885 |
| 4 | 0.293 | 0.893 | 0.314 | 0.887 |

Untrained baseline accuracy: 9.99% (≈ random guessing across 10 classes)
Final test accuracy: **88.7%**
Reloaded checkpoint accuracy (`FashionMNISTModel.pth`): **88.72%** — confirms 
`state_dict()` save/load round-trips correctly with no accuracy drift.

Predictions were visualized on 9 test samples before and after training — 
untrained predictions were essentially random, post-training predictions 
correctly classified nearly all samples shown (color-coded green/red for 
correct/incorrect).

## Limitations (expected at this stage)

- Only 5 epochs — accuracy was still climbing at the final epoch, more training 
  would likely push it higher
- No data augmentation, batch norm, or dropout — kept minimal to isolate what 
  convolution alone contributes over the vanilla NN baseline
- Fixed 10 hidden channels throughout — no channel expansion across blocks, 
  which is a common CNN design pattern this doesn't yet explore

## Next

→ `Vanilla_GAN_MNIST/` — shifts from classification to generative modeling, 
introducing adversarial training on MNIST digits.
