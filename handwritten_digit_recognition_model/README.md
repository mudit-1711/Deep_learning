# Handwritten Digit Recognition (MNIST)

A deep learning project comparing Artificial Neural Networks (ANN) and Convolutional Neural Networks (CNN) for handwritten digit recognition on the MNIST dataset using TensorFlow/Keras.

---

## 📌 Overview

This project focuses on classifying handwritten digits (0–9) using grayscale 28x28 pixel images. It demonstrates data preprocessing, model building, training, evaluation, and saving trained Keras models.

---

## 🏗️ Model Architectures & Performance

### 1. Convolutional Neural Network (CNN) — *Recommended*
- **Architecture**:
  - `Conv2D(32, (3,3), relu)` + `MaxPool2D(2,2)`
  - `Conv2D(64, (3,3), relu)` + `MaxPool2D(2,2)`
  - `Flatten`
  - `Dense(128, relu)` + `Dropout(0.5)`
  - `Dense(10, softmax)`
- **Optimizer**: Adam
- **Loss**: Categorical Crossentropy
- **Accuracy**: **~99.48%** on test data

### 2. Artificial Neural Network (ANN) Baseline
- **Architecture**: Dense feedforward network
- **Accuracy**: **~98.09%** on test data

---

## 📁 Directory Structure

```plaintext
handwritten_digit_recognition_model/
├── CNN.ipynb          # Jupyter notebook containing data prep, training, and evaluations
├── cnn_model.keras    # Saved trained CNN model
├── train.csv.zip      # MNIST dataset (zipped CSV)
└── README.md          # Project documentation
```

---

## 🚀 Getting Started

### Prerequisites
Make sure you have Python 3.9+ and the required packages installed:
```bash
pip install tensorflow numpy pandas matplotlib scikit-learn streamlit streamlit-drawable-canvas plotly
```

### Running the Web Application (Interactive Drawing Canvas)
Launch the interactive Streamlit app to draw digits and get real-time predictions:
```bash
streamlit run handwritten_digit_recognition_model/app.py
```

### Running the Notebook
1. Unzip `train.csv.zip` if running locally without automatic zip handling.
2. Open and run [`CNN.ipynb`](CNN.ipynb) in Jupyter Notebook, VS Code, or Kaggle.

### Loading the Trained Model
```python
from tensorflow.keras.models import load_model

model = load_model("cnn_model.keras")
# Predict on 28x28 normalized images reshaped to (1, 28, 28, 1)
predictions = model.predict(image)
```

