# Vanilla NN — MNIST

First stage of `dl-progression` — a from-scratch feedforward neural network 
classifying handwritten digits from MNIST.

## What this is

A simple multi-layer perceptron (MLP) with two hidden layers, trained end-to-end 
on flattened 28×28 MNIST images to classify digits 0–9.

## Architecture

- Input: 784 (flattened 28×28 grayscale pixel values, normalized to [0,1])
- Hidden layers: 2 × Linear(64) with ReLU activation
- Output: 10 (logits over digit classes 0–9)
- No convolutions, no regularization — pure fully-connected layers

## Training

- Loss: CrossEntropyLoss
- Optimizer: Adam, lr = 0.001
- Epochs: 300
- **Full-batch gradient descent** — the entire training set is passed through in 
  one forward/backward pass per epoch, no mini-batching or DataLoader. This keeps 
  the loop simple for a first-stage model, at the cost of noisier/slower convergence 
  than mini-batch SGD would give.
- Train/test split: 80/20 (`random_state=42`)

## Results

| Epoch | Train Loss | Train Acc | Test Loss | Test Acc |
|---|---|---|---|---|
| 0   | 2.30 | 0.10 | 2.28 | 0.15 |
| 50  | 0.49 | 0.87 | 0.48 | 0.87 |
| 100 | 0.29 | 0.92 | 0.30 | 0.92 |
| 150 | 0.23 | 0.93 | 0.25 | 0.93 |
| 200 | 0.19 | 0.95 | 0.21 | 0.94 |
| 250 | 0.15 | 0.96 | 0.18 | 0.95 |

Final test accuracy: **~95%**

Predictions were visualized before and after training on a sample of 10 test 
images — pre-training predictions were essentially random guesses, post-training 
predictions matched true labels on nearly all samples shown.

## Limitations (expected at this stage)

- No convolutional structure — the model treats pixels as an unordered flat 
  vector, discarding spatial relationships entirely
- Full-batch training doesn't scale to larger datasets
- No regularization (dropout, weight decay) — fine at this scale, but would 
  overfit on harder tasks

## Next

→ `CNN_FashionMNIST/` — introduces convolutional layers to exploit spatial 
structure in images, tested on the harder FashionMNIST dataset.
