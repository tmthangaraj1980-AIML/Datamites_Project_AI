# PRAICP-1009-101 — Food-101 Image Classification

**DataMites AI Capstone Project | Computer Vision | CNN and Transfer Learning**

## Overview
Classify RGB food photographs into **101 Food-101 categories**. This project compares a custom CNN baseline with ImageNet-pretrained architectures and records progressive fine-tuning experiments.

## Dataset and preprocessing
- **Food-101:** 101 classes and 101,000 images.
- Target input: **224 × 224 RGB** images.
- Original experiments used a nominal **70% training / 15% validation / 15% testing** split.
- TensorFlow `tf.data` input pipeline with decoding, resizing, batching and prefetching.
- Preprocessing differs by architecture; see the notebook for the exact input scaling.
- Data augmentation was used in the **EfficientNetV2B0** and **MobileNetV2** experiments. The original **EfficientNetB0** training did **not** use the same explicit augmentation pipeline.

## Experiments and recorded results

| Architecture / stage | Best recorded validation accuracy |
|---|---:|
| Custom CNN baseline | **29.76%** (final validation) |
| EfficientNetB0, initial frozen backbone (Colab) | 67.62% |
| EfficientNetB0, continued frozen backbone (Kaggle) | 73.64% |
| EfficientNetB0, upper 20 layers fine-tuned | 77.38% |
| EfficientNetB0, extended fine-tuning | 77.87% |
| **EfficientNetB0, upper 50 layers fine-tuned** | **78.17%** |
| EfficientNetV2B0, frozen backbone with augmentation | 67.13% |
| MobileNetV2, frozen backbone with augmentation | 55.01% |

**Selected architecture:** EfficientNetB0 based on the highest recorded validation accuracy (**78.17%**). In the final stage, upper backbone layers were fine-tuned with a very low learning rate while keeping BatchNormalization layers frozen.

### Evaluation caveat
The original experiment recorded **78.00% test accuracy** and approximately **78.00% macro F1** on a reconstructed Kaggle split. Because the original Colab and Kaggle data ordering/splits were not guaranteed identical, the reconstructed test set may overlap with earlier training images. **Do not interpret 78.00% as independently verified holdout accuracy.** The reported model comparisons also reflect different training/fine-tuning stages, not an identical experimental budget.

## Repository contents
Files visible in the local project folder include:
- `PRAICP_1009_101_Food_Classification.ipynb`
- `Food101_Classification_Final.ipynb`
- `Food101-efficientnetb0-training.ipynb`
- `food101_baseline_cnn.keras`
- `best_efficientnet.keras`

Other filenames may vary; inspect the notebook and model files before assuming they are interchangeable. The highest-performing **Stage 3** checkpoint was named `best_finetuned_stage3.keras` in the original Kaggle runtime and was **not confirmed recovered** in the consolidated notebook. The file `best_efficientnet.keras` should **not** be claimed to contain those Stage 3 weights without checking.

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook PRAICP_1009_101_Food_Classification.ipynb
```

The experiments were performed in **Google Colab and Kaggle**. Attach/download Food-101 and update environment-specific dataset and checkpoint paths before running. Training all models requires significant compute; GPU acceleration is recommended.

## Model loading example

```python
import tensorflow as tf
model = tf.keras.models.load_model('best_efficientnet.keras', compile=False)
```

**Important:** This loads the named local checkpoint only; it does not establish that the model is the final 78.17% Stage 3 model. Use the exact architecture-specific preprocessing and the original ordered 101-class label mapping for predictions.

## Limitations and future work
- Recreate a single fixed, verified, independent split before retraining and comparing all architectures.
- Re-run all architectures with comparable training budgets and consistent augmentation where appropriate.
- Report macro precision/recall/F1 and inference speed on a truly independent test set.
- Preserve checkpoint provenance, model version, and the ordered class-label mapping.

## Status
Baseline CNN, EfficientNetB0 fine-tuning, EfficientNetV2B0 and MobileNetV2 frozen-backbone experiments are documented. The consolidated notebook includes opt-in training sections rather than automatically rerunning expensive experiments.
