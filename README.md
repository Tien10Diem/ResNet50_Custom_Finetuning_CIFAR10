# Custom ResNet50 for CIFAR-10 Image Classification

This project builds and trains ResNet50 **entirely from scratch using pure PyTorch** — no `torchvision.models` or any pre-built implementation. Every component, from the Bottleneck block and residual connections to the weight-loading logic, is defined manually to gain a deep understanding of how ResNet actually works rather than just calling an API.

## Model Architecture

The model is built bottom-up through 3 abstraction layers: `BotNeck` → `Layer` → `ResNet50`.

- **`BotNeck`**: A custom bottleneck block implementing the Conv 1×1 → 3×3 → 1×1 sequence. The shortcut connection is handled manually: `Identity` if shapes match, Conv 1×1 if adjustment is needed.
- **`Layer`**: Stacks multiple `BotNeck` blocks together. The first block may downsample (stride=2); subsequent blocks use stride=1.
- **`ResNet50`**: Assembles everything following the original ResNet50 configuration.

### Architecture Diagram

![Architecture Diagram](architecture.png)

Overview:

1. **Stem**: Conv 7×7, stride=2 → BatchNorm → ReLU → MaxPool
2. **Residual Layers**:
   - Layer 1: 3 Bottleneck blocks (64 → 256)
   - Layer 2: 4 Bottleneck blocks (256 → 512)
   - Layer 3: 6 Bottleneck blocks (512 → 1024)
   - Layer 4: 3 Bottleneck blocks (1024 → 2048)
3. **Head**: AdaptiveAvgPool → FC(2048 → 10)

## Installation

```bash
git clone https://github.com/Ta-Quang-Huy/ResNet50_Custom_Finetuning_CIFAR10.git
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Download the CIFAR-10 dataset by running the first cells in the notebook.

## Training & Evaluation

**Optimizer:** Adam `lr=1e-4`, `weight_decay=1e-4` | **Batch size:** 32 | **Epochs:** 3

### Training Results

| Epoch | Train Loss | Val Loss | Val Accuracy |
|:---:|:---:|:---:|:---:|
| 1 | 0.3448 | 0.1398 | 95.24% |
| 2 | 0.1139 | 0.1497 | 95.42% |
| 3 | 0.0714 | 0.1423 | 95.96% |

### Evaluation Results — Test Accuracy: **95.17%**

| Class | Precision | Recall | F1-score | Support |
|:---|:---:|:---:|:---:|:---:|
| airplane | 0.95 | 0.95 | 0.95 | 1000 |
| automobile | 0.97 | 0.98 | 0.98 | 1000 |
| bird | 0.92 | 0.96 | 0.94 | 1000 |
| cat | 0.91 | 0.90 | 0.90 | 1000 |
| deer | 0.97 | 0.95 | 0.96 | 1000 |
| dog | 0.93 | 0.90 | 0.92 | 1000 |
| frog | 0.98 | 0.97 | 0.97 | 1000 |
| horse | 0.97 | 0.96 | 0.97 | 1000 |
| ship | 0.95 | 0.99 | 0.97 | 1000 |
| truck | 0.99 | 0.96 | 0.97 | 1000 |
| **avg** | **0.95** | **0.95** | **0.95** | **10000** |

## Inference

```bash
python inferences.py -i <path_to_image> -w <path_to_weight>
# Example:
python inferences.py -i R.jpg -w best_model.pth
```
