# Indian Sign Language Recognition Using Deep Learning (CNN)

**DataMites AI Capstone Project** | **Project ID:** PRAICP-1000-IndiSignLang  
**Domain:** Computer Vision · Deep Learning · Multiclass Image Classification

## Problem Statement

Develop an image-based model that recognizes Indian Sign Language (ISL) alphabet hand gestures. The project focuses on image preprocessing and predicting the sign represented in an input image, with the broader goal of supporting communication accessibility.

## Dataset

The supplied dataset is organized into folders named after the gesture classes.

- **4,972** image filenames were initially counted across **24 classes**.
- **Classes:** A, B, C, D, E, F, G, H, I, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y (J and Z are absent).
- **Original resolutions:** 640 × 480, 1920 × 1088, and 1088 × 1920.
- **Image mode:** RGB for successfully opened images.
- **Data-quality note:** A later inspection identified five unreadable files: `B/182.jpg`, `X/046.jpg`, `X/020.jpg`, `Y/085.jpg`, and `Y/105.jpg`. The notebook's image counts varied across runs; verify the final usable dataset before reproducing the split.

Dataset source: [DataMites project ZIP](https://d3ilbtxij3aepc.cloudfront.net/projects/AI-Capstone-Projects/PRAICP-1000-IndiSignLang.zip).

## Workflow

1. Understand the problem and inspect class folders.
2. Explore class distribution and sample gesture images.
3. Inspect image dimensions and RGB modes.
4. Resize to **128 × 128** with aspect-ratio-preserving padding.
5. Normalize pixels to **[0, 1]** and encode labels **0–23**.
6. Perform stratified training/validation/test splits (**70% / 15% / 15%**).
7. Train and evaluate a baseline CNN.
8. Review accuracy/loss curves, classification report, confusion matrix, and misclassified examples.
9. Save the trained model and class mapping.

## CNN Architecture

| Layer | Configuration |
|---|---|
| Input | 128 × 128 × 3 |
| Conv2D + MaxPooling2D | 32 filters, 3 × 3, ReLU; pool 2 × 2 |
| Conv2D + MaxPooling2D | 64 filters, 3 × 3, ReLU; pool 2 × 2 |
| Conv2D + MaxPooling2D | 128 filters, 3 × 3, ReLU; pool 2 × 2 |
| Flatten | 25,088 features |
| Dense | 128 units, ReLU |
| Dropout | 0.5 |
| Output | 24 units, Softmax |

**Total trainable parameters:** 3,307,736.

### Training Configuration

- Optimizer: Adam, initial learning rate `0.001`
- Loss: `sparse_categorical_crossentropy`
- Metric: accuracy
- Batch size: `32`; maximum epochs: `20`
- Callbacks: `EarlyStopping` (monitor `val_loss`, patience 5, restore best weights) and `ReduceLROnPlateau` (factor 0.5, patience 3)

## Test Results

| Metric | Result |
|---|---:|
| Test accuracy | **99.73%** |
| Test loss | **0.0134** |
| Test images | **746** |
| Correct predictions | **744** |
| Incorrect predictions | **2** |

Two observed errors were **L → V** and **X → R**. The classification report and confusion matrix show strong performance on this held-out split.

> **Interpretation:** This result applies to the reported test split. Duplicate-image, near-duplicate, and signer-independent leakage checks were not completed. Accuracy on new people, camera conditions, or backgrounds has not been established.

## Suggested Repository Structure

```text
Indian_Sign_Language/
├── notebooks/
│   └── Indian_Sign_Language.ipynb
├── models/
│   ├── isl_cnn_model.keras
│   └── class_mapping.json
├── results/
├── Data/                  # Local dataset; do not commit if large
├── README.md
├── requirements.txt
└── .gitignore
```

Adjust the notebook filename and folder layout to match your actual repository. Model files and results are optional until you have exported them.

## Run the Project

1. Use a compatible Python environment (Python 3.11 is a practical choice).
2. Install dependencies:
   ```bash
   python -m pip install -r requirements.txt
   ```
3. Download and extract the dataset, preserving the alphabet class folders.
4. Open `notebooks/Indian_Sign_Language.ipynb` in Jupyter or Google Colab.
5. **Update `dataset_path`** to point to your local or Drive `Data` directory.
6. Run notebook cells in order: EDA → preprocessing → splitting → CNN → training → evaluation.
7. If using the saved model, load it with `tf.keras.models.load_model(...)` and load `class_mapping.json` to decode predicted indices.

**Note:** The original notebook used Google Colab paths and may require small path adjustments on a local machine. Google Colab provides TensorFlow and many libraries by default; package versions can differ from local installations.

## Limitations and Future Work

- The project covers 24 static alphabet gesture classes; J and Z are not included.
- Some files were unreadable during inspection.
- Evaluate duplicate/near-duplicate images and signer-independent splits before making real-world performance claims.
- Test against new signers, lighting conditions, camera angles, and backgrounds.
- Explore augmentation, transfer learning (e.g., MobileNetV2), and real-time webcam inference.

## Conclusion

A baseline CNN achieved **99.73% test accuracy (744/746 correct)** on the reported Indian Sign Language alphabet image test split. The project demonstrates a complete supervised computer-vision workflow from preprocessing to evaluation and model saving. Broader accessibility applications require additional real-world validation.
