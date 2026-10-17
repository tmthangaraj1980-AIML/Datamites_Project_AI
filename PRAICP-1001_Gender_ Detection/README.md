# PRAICP-1001 — Gender Detection - Image Classification 

**DataMites AI Capstone Project | Computer Vision | TensorFlow / Keras**

## Overview
A custom convolutional neural network (CNN) trained to classify face photographs into the two labels supplied by the dataset: `man` (0) and `woman` (1). These are **dataset categories**, not a reliable determination of an individual's gender identity.

## Dataset
- Total: **2,307** RGB images; `man`: **1,173**, `woman`: **1,134**.
- Folder structure: `gender_dataset_face/man/` and `gender_dataset_face/woman/`.
- Stratified split: **1,614 training**, **346 validation**, **347 test** images (approximately 70/15/15).
- All 2,307 image files passed the image readability check.
- Original image dimensions vary; inputs are resized to **224 × 224** and normalized to **[0, 1]**.
- Training augmentation: horizontal flip, slight rotation and zoom.

## Model
Custom Keras `Sequential` CNN:

1. Input `(224, 224, 3)` and training-time augmentation
2. `Conv2D(32, 3, relu)` → `MaxPooling2D`
3. `Conv2D(64, 3, relu)` → `MaxPooling2D`
4. `Conv2D(128, 3, relu)` → `MaxPooling2D`
5. `Flatten` → `Dense(128, relu)` → `Dropout(0.2)`
6. `Dense(1, sigmoid)`

Trained with Adam, binary cross-entropy, early stopping and a best-validation-loss checkpoint.

## Results

| Metric | Result |
|---|---:|
| Best validation accuracy | **92.49%** |
| Best validation loss | **0.1935** |
| Saved-model test accuracy | **94.52%** |
| Saved-model test loss | **0.1330** |
| Correct / total test predictions | **328 / 347** |

Test confusion matrix (rows = true label; columns = predicted label):

| | Predicted man | Predicted woman |
|---|---:|---:|
| True man | 168 | 8 |
| True woman | 11 | 160 |

Class-specific precision / recall / F1 (rounded): `man`: 0.94 / 0.95 / 0.95; `woman`: 0.95 / 0.94 / 0.94.

Two external example images produced predicted labels `woman` (96.16% model score) and `man` (98.58% model score). Two examples do not establish generalization; model scores are not calibrated probabilities of gender identity.

## Repository contents
- `PRAICP_1001_Gender_Detection.ipynb` — project notebook.
- `best_gender_cnn.keras` — saved Keras model.
- `README.md` — this documentation.
- `requirements.txt` — Python dependencies.

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook PRAICP_1001_Gender_Detection.ipynb
```

The original notebook was developed in Google Colab. Update its Google Drive dataset paths when running locally; mount Drive in Colab if necessary. Obtain the dataset from the course-provided source; do not commit a large image dataset unless redistribution is permitted.

## Load saved model

```python
import tensorflow as tf
model = tf.keras.models.load_model('best_gender_cnn.keras')
```

For inference, resize to `(224, 224)`, convert to RGB, scale pixel values by `/255.0`, add a batch dimension and interpret sigmoid output below 0.5 as dataset label `man` and 0.5 or above as `woman`.

## Limitations and responsible use
- Small dataset; results may not transfer to different image sources, lighting or populations.
- Label accuracy and demographic representation have not been independently audited.
- Facial appearance cannot reliably determine gender identity; do not use this educational classifier for decisions about people.

## Status
Completed: EDA, preprocessing, training, checkpointing, held-out test evaluation and two example inference runs.
