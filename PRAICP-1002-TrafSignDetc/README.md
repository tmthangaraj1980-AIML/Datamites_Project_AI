# Traffic Sign Classification using CNN

**DataMites AI Capstone Project**  
**Project ID:** PRAICP-1002-TrafSignDetc

## Project Overview

This project develops a Convolutional Neural Network (CNN) to classify traffic sign images into 43 categories.

Two CNN models were developed and evaluated to compare the effects of data augmentation on classification performance.

## Dataset

- Training images: 39,209
- Testing images: 12,630
- Number of classes: 43
- Image size: 32 × 32 × 3
- Image normalization: 0–1

## Technologies Used

- Python
- Google Colab
- TensorFlow / Keras
- NumPy and Pandas
- Pillow
- Matplotlib and Seaborn
- Scikit-learn

## Project Workflow

1. Business understanding
2. Dataset exploration and visualization
3. Image preprocessing
4. Training and validation split
5. Baseline CNN development
6. Model training and evaluation
7. Classification report and confusion matrix
8. Data augmentation
9. Improved CNN development
10. Model comparison and conclusion

## Model Architecture

- Three Conv2D layers
- Three MaxPooling2D layers
- GlobalAveragePooling2D
- Dense layer with 128 neurons
- Dropout (0.3)
- Softmax output with 43 classes

Total trainable parameters: 115,307.

## Model Performance

| Metric | Baseline CNN | Augmented CNN |
|---|---:|---:|
| Test accuracy | 93.94% | Approximately 96% |
| Macro F1-score | 0.91 | 0.94 |
| Weighted F1-score | 0.94 | 0.96 |

## Conclusion

The CNN with data augmentation achieved better overall test performance than the baseline CNN.

Data augmentation improved the model's ability to classify traffic signs under varying visual conditions.

The project demonstrates practical applications of deep learning in computer vision and traffic sign recognition.

## How to Run

1. Open the project notebook in Google Colab.
2. Download the dataset from the authorized DataMites project source.
(https://d3ilbtxij3aepc.cloudfront.net/projects/AI-Capstone-Projects/PRAICP-1002-TrafSignDetc.zip)
3. Extract the dataset.
4. Update the dataset path in the notebook.
5. Execute the notebook cells in sequence.

The dataset is not included in this repository.

