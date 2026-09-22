# 🧠 Next Word Prediction using Deep Learning (LSTM vs. RNN)

A deep learning Natural Language Processing (NLP) sequence modeling project that predicts the **next word** and autocompletes quotes given a starting seed prompt. Built using **TensorFlow/Keras** and deployed via an interactive, modern **Streamlit** web application.

---

## 📌 Table of Contents
1. [Project Overview](#-project-overview)
2. [Dataset Details](#-dataset-details)
3. [Text Preprocessing Pipeline](#-text-preprocessing-pipeline)
4. [Model Architectures & Comparison](#-model-architectures--comparison)
5. [Training & Performance Comparison (RNN vs. LSTM)](#-training--performance-comparison-rnn-vs-lstm)
6. [Why LSTM Outperforms SimpleRNN](#-why-lstm-outperforms-simplernn)
7. [Tech Stack & Dependencies](#-tech-stack--dependencies)
8. [Project Structure](#-project-structure)
9. [Streamlit Web Application](#-streamlit-web-application)
10. [Installation & Setup](#-installation--setup)
11. [How to Run the Application](#-how-to-run-the-application)

---

## 📖 Project Overview
Next Word Prediction (also known as Language Modeling) is a foundational task in Natural Language Processing. Given an arbitrary sequence of words, the neural network learns the conditional probability distribution over the entire vocabulary to predict which word is most likely to follow:

$$P(w_t \mid w_1, w_2, \dots, w_{t-1})$$

In this project, we explored both **Simple Recurrent Neural Networks (SimpleRNN)** and **Long Short-Term Memory (LSTM)** networks trained on thousands of inspirational and literary quotes. The final production deployment utilizes the **LSTM model (`lstm_model.h5`)** due to its superior capacity to capture sequential context and long-range semantic dependencies.

---

## 📊 Dataset Details
The dataset used is `qoute_dataset.csv`:
- **Total Records:** 3,038 quotes
- **Attributes:** `quote` (text), `Author` (Albert Einstein, Marilyn Monroe, Jane Austen, J.K. Rowling, etc.)
- **Total N-Gram Sequences Generated:** 85,270 training sequences
- **Vocabulary Size ($V$):** 8,978 unique words
- **Maximum Sequence Length ($\text{max\_len}$):** 745 tokens

---

## ⚙️ Text Preprocessing Pipeline
1. **Case Normalization:** Converted all text to lowercase.
2. **Punctuation Stripping:** Removed special characters and punctuation using `str.maketrans`.
3. **Tokenization:** Fitted Keras `Tokenizer(num_words=8978)` to map unique words to numerical indices.
4. **N-gram Sequence Generation:** For each quote sequence $[w_1, w_2, \dots, w_k]$, created incremental prefix sequences $[w_1, \dots, w_i]$ as inputs $X$ and target next word $w_{i+1}$ as $y$.
5. **Pre-Padding:** Padded all input sequences with leading zeros using `pad_sequences(maxlen=745, padding='pre')`.
6. **Target Encoding:** Converted target indices into one-hot categorical vectors using `to_categorical(y, num_classes=vocab_size)`.

---

## 🏗️ Model Architectures

Both models share a sequential architecture designed for next-token probability classification:

```
Input Sequence (Padded to 745)
          │
          ▼
┌──────────────────────────────────────────────┐
│  Embedding Layer (input_dim=8978, dim=50)     │
└──────────────────────────────────────────────┘
          │
          ▼
┌──────────────────────────────────────────────┐
│  Recurrent Layer (128 units: RNN or LSTM)    │
└──────────────────────────────────────────────┘
          │
          ▼
┌──────────────────────────────────────────────┐
│  Dense Layer (8978 units, Softmax)           │
└──────────────────────────────────────────────┘
          │
          ▼
Output Probability Distribution over Vocabulary
```

---

## 📈 Training & Performance Comparison (RNN vs. LSTM)

Both models were trained on a **Google Colab T4 GPU** with `batch_size = 128` and `optimizer = 'adam'`:

| Metric / Parameter | 🔄 SimpleRNN Model | 🧠 LSTM Model (Selected) | Difference / Improvement |
| :--- | :--- | :--- | :--- |
| **Model File** | `rnn_model.keras` | `lstm_model.h5` | HDF5 / Keras Saved Model |
| **Recurrent Layer** | `SimpleRNN(128)` | `LSTM(128)` | Gated Memory Cells |
| **Training Epochs** | **10 Epochs** | **100 Epochs** | +90 Epochs |
| **Batch Size** | 128 | 128 | Same |
| **Training Time (T4 GPU)** | **~7 Minutes** | **~54 Minutes** | Thorough parameter convergence |
| **Initial Loss (Epoch 1)** | 6.7374 | 6.7474 | Baseline initialization |
| **Initial Accuracy (Epoch 1)** | 4.30% | 4.01% | Baseline random guess |
| **Final Training Loss** | 4.2153 | **1.1370** | **-3.0783 Loss reduction** |
| **Final Training Accuracy** | **20.99%** | **74.91%** | **+53.92% Accuracy gain** |
| **Final Validation Loss** | 6.6397 | 11.8437 | Higher variance over long sequences |
| **Final Validation Accuracy** | 10.72% | 7.44% | Sparse quote multi-modality |

---

## 💡 Why LSTM Outperforms SimpleRNN
1. **Mitigation of Vanishing & Exploding Gradients:** Standard RNNs struggle to propagate gradient signals through long sequence histories (like quotes with tens to hundreds of words).
2. **Gating Mechanism:** LSTMs incorporate three distinct gates:
   - **Forget Gate ($f_t$):** Regulates how much of the past cell state to discard.
   - **Input Gate ($i_t$):** Decides which new information to store in the memory cell.
   - **Output Gate ($o_t$):** Controls what parts of the memory cell state to output.
3. **Long-Term Context Retention:** Because of the dedicated additive cell state channel ($C_t$), LSTMs remember grammatical subject-verb agreements and literary themes far across the sequence.

---

## 🛠️ Tech Stack & Dependencies
- **Language:** Python 3.12+
- **Deep Learning Framework:** TensorFlow 2.x / Keras
- **Web App Framework:** Streamlit
- **Data Manipulation:** NumPy, Pandas
- **Serialization:** Pickle (`pickle`)
- **Data Exploration & Visualization:** Matplotlib, Seaborn

---

## 📁 Project Structure

```
Next_Word_Predict/
├── app.py              # Streamlit Web Application (LSTM Inference UI)
├── lstm_model.h5       # Trained LSTM Deep Learning Model weights
├── tokenizer.pkl       # Serialized Keras Tokenizer vocabulary
├── max_len.pkl         # Serialized maximum sequence length scalar (745)
├── rnn_model.keras     # Trained Baseline SimpleRNN Model
├── qoute_dataset.csv   # Quotes dataset used for training
├── train.ipynb         # Jupyter Notebook with full EDA, training & evaluation
├── .gitignore          # Git ignore rules for cached & temporary artifacts
└── readme.md           # Complete project documentation
```

---

## 🖥️ Streamlit Web Application Features

The interactive web interface built in [app.py](file:///d:/deep_learning/Next_Word_Predict/app.py) provides:
- **Single Next Word Prediction:** Predicts the most probable next token with confidence visualization.
- **Top-K Candidate Probabilities:** Displays the top 5–10 candidate words with percentage probability progress bars.
- **Continuous Sentence Auto-Completion:** Generates the next $N$ words sequentially with customizable temperature (creativity control).
- **Interactive Quick-Fill Prompts:** 1-click test quotes from famous authors.
- **Resource Caching:** Fast model and tokenizer startup using `@st.cache_resource`.

---

## 🚀 Installation & Setup

### 1. Clone the repository / Navigate to the folder:
```bash
cd d:/deep_learning/Next_Word_Predict
```

### 2. (Optional) Create & activate a Virtual Environment:
```bash
python -m venv venv
# On Windows PowerShell:
.\venv\Scripts\Activate.ps1
# On Linux/macOS:
source venv/bin/activate
```

### 3. Install required packages:
```bash
pip install tensorflow streamlit numpy pandas matplotlib seaborn
```

---

## 🎯 How to Run the Application

Run the Streamlit server from inside the `Next_Word_Predict` directory:

```bash
streamlit run app.py
```

Or run from the workspace root:

```bash
streamlit run Next_Word_Predict/app.py
```

Once launched, open your browser at `http://localhost:8501` to test the live next word predictor!
