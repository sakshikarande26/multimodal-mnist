# HW6 Q3 — Multimodal MNIST Classification

All numbers below come from one deterministic run of `hw6_q3_template.py`
(seed = 42, batch size = 32, 10 epochs, MPS device on Apple Silicon).
The task is to predict the product `y = d1 * d2` of two MNIST digits, so
the classifier has 82 output classes (the products fall in the integer set
{0, 1, ..., 81}).

Data shapes I confirmed from the supplied `.npy` files:

- `train_data_merged.npy`: `(10000, 2, 784)` `float64`, range `[0, 1]`
- `val_data_merged.npy`:   `(2000, 2, 784)`  `float64`, range `[0, 1]`
- `test_data_merged.npy`:  `(2000, 2, 784)`  `float64`, range `[0, 1]`
- `*_labels_merged.npy`:   `(N, 3)` `int64`. Cols 0 and 1 are the two digit
  values (`0..9`); col 2 is overwritten in the loader as `label[:,0] *
  label[:,1]` so the multiplication target lives in `0..81`.

---

## 1. Model architecture

### EarlyFusion (Table 2 of HW6)

The two images are placed *side-by-side* along the width axis, then fed
through a single CNN.

The runtime shape check in the code prints (the script prints these from
`logs/early_fusion_<opt>.log`):

```
inputs:       x1=(1,1,28,28)  x2=(1,1,28,28)
after concat: (1,1,28,56)
after conv1:  (1,2,26,54)
after conv2:  (1,4,24,52)
after pool:   (1,4,12,26)
after flatten:(1,1248)
```

| Layer | Type | Output | Parameters |
|-------|------|--------|-----------:|
| concat (`dim=3`) | side-by-side | (1, 28, 56) | 0 |
| `conv1` | `Conv2d(1 → 2,  k=3, s=1, p=0)` + ReLU | (2, 26, 54) | 20 |
| `conv2` | `Conv2d(2 → 4,  k=3, s=1, p=0)` + ReLU | (4, 24, 52) | 76 |
| `pool`  | `MaxPool2d(k=2, s=2)`                  | (4, 12, 26) | 0 |
| dropout | `Dropout(p=0.25)`                       | (4, 12, 26) | 0 |
| flatten | -                                       | (1248,)     | 0 |
| `fc1`   | `Linear(1248 → 128)` + ReLU             | (128,)      | 159,872 |
| `fc2`   | `Linear(128 → 82)`                       | (82,)       | 10,578 |
| **Total** | | | **170,546** |

### LateFusion (unchanged)

Each image passes through the *same* (parameter-shared) conv stack; the two
flattened feature maps are concatenated and the FC head classifies the
joint vector.

| Layer | Type | Output | Parameters |
|-------|------|--------|-----------:|
| `conv1` | `Conv2d(1 → 16, k=3, p=1)` + ReLU + MaxPool(2) | (16, 14, 14) | 160 |
| `conv2` | `Conv2d(16 → 32, k=3, p=1)` + ReLU + MaxPool(2) | (32, 7, 7)   | 4,640 |
| `fc1`   | `Linear(3136 → 128)` + ReLU                     | (128,)       | 401,536 |
| `fc2`   | `Linear(128 → 82)`                              | (82,)        | 10,578 |
| **Total** | | | **416,914** |

---

## 2. Parameter analysis: conv vs fully-connected

| Model       | Conv params | FC params | Total   | Conv fraction | FC fraction |
|-------------|------------:|----------:|--------:|--------------:|------------:|
| EarlyFusion |          96 |   170,450 | 170,546 |  0.0006 (0.06%) |  0.9994 (99.94%) |
| LateFusion  |       4,800 |   412,114 | 416,914 |  0.0115 (1.15%) |  0.9885 (98.85%) |

**Why so lopsided?** Conv weights are `Cout · Cin · k · k`, and EarlyFusion
uses width 2 / 4 with 3×3 kernels — together that is just 96 trainable
parameters. Meanwhile `fc1` has `(C · H · W) · 128 = 4 · 12 · 26 · 128 =
159,744` weights (plus 128 biases = 159,872). One single hidden FC layer
carries roughly 1,665× as many parameters as the entire conv stack.

LateFusion is more balanced (1.15% conv) but still overwhelmingly FC-heavy
because `fc1` ingests the concatenated 3,136-dim vector from two
independently-extracted feature maps.

**Practical takeaway.** If you wanted to shrink either model the cheap win
is narrowing `fc1` (e.g. a smaller hidden width, or a global-average-pool
before the FC), not adding/removing conv channels.

---

## 3. Optimizer comparison (EarlyFusion)

Three optimizers were trained for 10 epochs each on the EarlyFusion
architecture above with the same batch size and seed.

| # | Optimizer | Hyperparameters |
|---|-----------|-----------------|
| 1 | `torch.optim.Adam`    | `lr=0.001, betas=(0.999, 0.999)` |
| 2 | `torch.optim.RMSprop` | `lr=0.001, alpha=0.9, centered=False, weight_decay=0.001` |
| 3 | `torch.optim.AdamW`   | `lr=0.001, betas=(0.99, 0.999), weight_decay=0.01` |

### 3.1 Final accuracies

`final_*_acc` is at the last epoch (epoch 10). `best_val_epoch` is the
val-selected checkpoint and `best_ckpt_test_acc` is its test accuracy.

| Optimizer | Final train | Final val | Final test | Best-val epoch | Test @ best ckpt |
|-----------|------------:|----------:|-----------:|---------------:|-----------------:|
| Adam      |  0.7354 |  0.6770 |  0.6730 |  9 |  0.6730 |
| RMSprop   |  0.9120 |  0.8170 |  **0.7975** |  9 |  0.7975 |
| AdamW     |  0.8894 |  0.7970 |  0.7805 |  9 |  0.7805 |

### 3.2 Per-epoch test accuracy (EarlyFusion)

| Epoch | Adam | RMSprop | AdamW |
|------:|-----:|--------:|------:|
|  1 | 0.204 | 0.400 | 0.252 |
|  2 | 0.331 | 0.552 | 0.429 |
|  3 | 0.404 | 0.630 | 0.558 |
|  4 | 0.433 | 0.703 | 0.631 |
|  5 | 0.487 | 0.726 | 0.711 |
|  6 | 0.527 | 0.749 | 0.741 |
|  7 | 0.584 | 0.749 | 0.739 |
|  8 | 0.622 | 0.777 | 0.763 |
|  9 | 0.653 | 0.770 | 0.773 |
| 10 | 0.673 | 0.797 | 0.780 |

### 3.3 Plots

- `early_fusion_train_accuracy.png` — training accuracy vs epoch for the 3
  optimizers.
- `early_fusion_val_accuracy.png`   — validation accuracy vs epoch for the
  3 optimizers.
- `early_fusion_accuracy_comparison.png` — bonus: train (solid) and test
  (dashed) overlay for the 3 optimizers.

### 3.4 Discussion of optimizer behavior

**Adam with `betas=(0.999, 0.999)` is the slowest and worst.**
At step *t* the first-moment estimate effectively averages over a window of
about `1/(1−β₁) = 1000` minibatches. With only ~313 steps per epoch
(10000/32), the running mean barely warms up inside an epoch, so each
gradient step is heavily shrunk. This Adam config trails badly throughout
training (test 0.204 after epoch 1 vs 0.400 RMSprop / 0.252 AdamW) and never
catches up — final test accuracy 0.673 is ~12 pp behind RMSprop.

**RMSprop is the fastest learner and the best on EarlyFusion.**
Without momentum and with `α=0.9`, it adapts the per-parameter learning
rate on a much shorter horizon (~10 minibatches), so the first few epochs
are unmistakably faster: by epoch 3 it is at 0.630 test accuracy, ahead of
both Adam variants. The mild `weight_decay=0.001` keeps the train accuracy
in check (final train 0.912 vs AdamW 0.889). RMSprop tops the table at
epoch 10 (test 0.7975, val 0.8170). Curves are slightly noisier than AdamW
(epoch-7 test plateau at 0.749) but the trend is monotonic enough that no
early stopping was needed — the val-selected checkpoint is at epoch 9.

**AdamW is competitive but slightly behind RMSprop here.**
With `betas=(0.99, 0.999)` the first moment uses a more reasonable
averaging window (~100 minibatches) and the decoupled `weight_decay=0.01`
is substantially stronger than RMSprop's L2 penalty. AdamW starts slower
than RMSprop (test 0.252 vs 0.400 at epoch 1) and overtakes Adam by epoch
2, but doesn't catch RMSprop. Stronger decay also keeps its train accuracy
below RMSprop's. Final test 0.780.

**Headline takeaway.** The choice of `betas` matters more than the
algorithm name. The pathological `β₁=0.999` cripples Adam. Between the
sensible choices, RMSprop — the simpler optimizer — actually wins on this
problem at this scale; AdamW's heavier decay would likely matter more on a
larger network or with longer training.

### 3.5 Effect of fusion type

LateFusion still outperforms the new EarlyFusion under every optimizer
(LateFusion final test: Adam 0.857, RMSprop 0.854, AdamW 0.882, vs
EarlyFusion 0.673 / 0.798 / 0.780). The gap is now large because the new
EarlyFusion has only 170 K parameters (vs 417 K for LateFusion) and only
4-channel feature maps after the conv stack — the bottleneck is capacity
in the convolutional encoder, not in the fusion strategy. LateFusion gets
twice the conv capacity (one pass per image, parameter-shared) and 2.6× the
total parameters.

---

## 4. Reproducibility

The script seeds `torch`, `torch.cuda`, `numpy`, and Python `random` to 42
and sets `cudnn.deterministic=True`, `cudnn.benchmark=False`. On the
Apple-MPS backend a few operators are still non-deterministic, so re-running
may shift accuracies by a couple of tenths of a percent; the qualitative
ordering (RMSprop > AdamW > Adam on EarlyFusion) is stable.

---

## 5. Files written by the script

- `final_summary.json` — machine-readable summary of all six runs and the
  per-layer parameter splits.
- `early_fusion_train_accuracy.png`, `early_fusion_val_accuracy.png` — the
  two EarlyFusion plots required by Q3.
- `early_fusion_accuracy_comparison.png`,
  `late_fusion_accuracy_comparison.png` — bonus train/test overlay plots
  for both fusion models.
- `<fusion>_<optimizer>_report/` — per-run directory with
  `train_accuracies.npy`, `val_accuracies.npy`, `test_accuracies.npy`,
  `mnist_model.pt` (best-val checkpoint), and `test_summary.json`.
