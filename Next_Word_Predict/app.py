import os
import pickle
import string
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Next Word Predictor | Deep Learning LSTM",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern, premium appearance
st.markdown("""
<style>
    /* Main container and font styling */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1E1B4B 0%, #312E81 50%, #4338CA 100%);
        color: white;
        padding: 2.2rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.8rem;
        box-shadow: 0 10px 25px -5px rgba(67, 56, 202, 0.3);
    }
    
    .main-header h1 {
        color: #FFFFFF;
        font-size: 2.3rem;
        font-weight: 700;
        margin-bottom: 0.4rem;
    }
    
    .main-header p {
        color: #E0E7FF;
        font-size: 1.05rem;
        margin: 0;
    }
    
    .stat-badge {
        display: inline-block;
        background: rgba(255, 255, 255, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.25);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.82rem;
        color: #EEF2FF;
        margin-right: 8px;
        margin-top: 10px;
    }
    
    .card-box {
        background: var(--background-color, #ffffff);
        border: 1px solid rgba(128, 128, 128, 0.18);
        border-radius: 12px;
        padding: 1.4rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.04);
    }
    
    .prediction-pill {
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
        color: white;
        font-weight: 600;
        font-size: 1.35rem;
        padding: 0.6rem 1.4rem;
        border-radius: 10px;
        display: inline-block;
        box-shadow: 0 4px 14px rgba(79, 70, 229, 0.35);
        margin: 0.4rem 0;
    }
    
    .top-candidate-badge {
        background: rgba(99, 102, 241, 0.1);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 8px;
        padding: 8px 14px;
        margin: 4px;
        display: inline-block;
        font-weight: 500;
    }
    
    .generated-text-box {
        background-color: rgba(99, 102, 241, 0.06);
        border-left: 4px solid #4F46E5;
        border-radius: 0 8px 8px 0;
        padding: 1.2rem;
        font-size: 1.15rem;
        line-height: 1.7;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Resource Loading with Cache
# ---------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, "lstm_model.h5")
TOKENIZER_PATH = os.path.join(CURRENT_DIR, "tokenizer.pkl")
MAXLEN_PATH = os.path.join(CURRENT_DIR, "max_len.pkl")

@st.cache_resource(show_spinner="Loading LSTM Neural Network & Tokenizer...")
def load_all_resources():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
    if not os.path.exists(TOKENIZER_PATH):
        raise FileNotFoundError(f"Tokenizer file not found at: {TOKENIZER_PATH}")
    if not os.path.exists(MAXLEN_PATH):
        raise FileNotFoundError(f"Max length file not found at: {MAXLEN_PATH}")
    
    # Load LSTM Model
    model = load_model(MODEL_PATH)
    
    # Load Tokenizer
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
        
    # Load Max Length
    with open(MAXLEN_PATH, "rb") as f:
        max_len = pickle.load(f)
        
    # Build inverted index for fast lookup
    index_to_word = {idx: word for word, idx in tokenizer.word_index.items()}
    
    return model, tokenizer, max_len, index_to_word

try:
    model, tokenizer, max_len, index_to_word = load_all_resources()
except Exception as e:
    st.error(f"Error loading model resources: {str(e)}")
    st.stop()

# ---------------------------------------------------------
# Preprocessing & Inference Helpers
# ---------------------------------------------------------
def preprocess_text(text: str) -> str:
    """Lowercase and strip punctuation to match training distribution."""
    text = text.lower()
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)

def get_next_word_predictions(text: str, top_k: int = 5):
    """Predicts next word along with top-k candidates and confidence scores."""
    clean_text = preprocess_text(text)
    if not clean_text.strip():
        return None, []
    
    seq = tokenizer.texts_to_sequences([clean_text])
    if not seq or not seq[0]:
        return None, []
    
    padded_seq = pad_sequences([seq[0]], maxlen=max_len, padding='pre')
    preds = model.predict(padded_seq, verbose=0)[0]
    
    top_indices = np.argsort(preds)[::-1][:top_k]
    candidates = []
    
    for idx in top_indices:
        word = index_to_word.get(int(idx), None)
        prob = float(preds[idx])
        if word:
            candidates.append((word, prob))
            
    best_word = candidates[0][0] if candidates else None
    return best_word, candidates

def generate_continuous_text(seed_text: str, n_words: int = 5, temperature: float = 1.0) -> str:
    """Generates continuous sequence of N words following the seed text."""
    current_text = seed_text
    
    for _ in range(n_words):
        clean_text = preprocess_text(current_text)
        seq = tokenizer.texts_to_sequences([clean_text])
        if not seq or not seq[0]:
            break
            
        padded_seq = pad_sequences([seq[0]], maxlen=max_len, padding='pre')
        preds = model.predict(padded_seq, verbose=0)[0]
        
        if temperature > 0.05:
            # Apply temperature scaling
            preds = np.asarray(preds).astype('float64')
            preds = np.log(preds + 1e-10) / temperature
            exp_preds = np.exp(preds)
            preds = exp_preds / np.sum(exp_preds)
            # Sample from distribution
            pred_idx = np.random.choice(len(preds), p=preds)
        else:
            # Greedy argmax
            pred_idx = np.argmax(preds)
            
        next_word = index_to_word.get(int(pred_idx), None)
        if not next_word or next_word == "<unk>":
            break
            
        current_text += " " + next_word
        
    return current_text

# ---------------------------------------------------------
# Sidebar Configuration & Architecture Overview
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/brain.png", width=110)
    st.title("Model Settings")
    
    mode = st.radio(
        "🎯 Select Mode:",
        ["Next Single Word", "Sentence Completion (Multi-Word)"],
        index=0
    )
    
    st.markdown("---")
    st.subheader("⚙️ Hyperparameters")
    
    if mode == "Next Single Word":
        top_k = st.slider("Top Candidates to show:", min_value=1, max_value=10, value=5)
    else:
        num_words = st.slider("Number of words to generate:", min_value=1, max_value=25, value=7)
        temp = st.slider("Creativity / Temperature:", min_value=0.0, max_value=1.5, value=0.7, step=0.1,
                         help="0.0 = Pure deterministic (greedy), > 0.7 = More diverse and creative")
    
    st.markdown("---")
    st.subheader("📊 Model Metadata")
    st.markdown(f"""
    - **Architecture**: LSTM (Long Short-Term Memory)
    - **Embedding Dim**: 50
    - **LSTM Units**: 128
    - **Vocabulary Size**: {len(tokenizer.word_index):,} words
    - **Max Sequence Len**: {max_len} tokens
    - **Trained Epochs**: 100 Epochs
    - **Train Accuracy**: **74.91%**
    - **Dataset**: Quotes Dataset (3,038 quotes)
    """)

# ---------------------------------------------------------
# Main UI Layout
# ---------------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1>🧠 Next Word Prediction with LSTM</h1>
    <p>Empowered by Deep Learning sequence modeling to predict contextually relevant words and auto-complete quotes.</p>
    <div>
        <span class="stat-badge">⚡ 100 Epochs Trained</span>
        <span class="stat-badge">🎯 74.91% Training Accuracy</span>
        <span class="stat-badge">📚 8,978 Vocabulary</span>
        <span class="stat-badge">🚀 Powered by TensorFlow & Keras</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Sample prompt buttons
st.write("💡 **Quick Try Examples:**")
col_e1, col_e2, col_e3, col_e4 = st.columns(4)

sample_prompt = ""
if col_e1.button("“The world as we”"):
    sample_prompt = "The world as we"
if col_e2.button("“There are only two”"):
    sample_prompt = "There are only two"
if col_e3.button("“It is our choices”"):
    sample_prompt = "It is our choices"
if col_e4.button("“What are you”"):
    sample_prompt = "what are you"

# Input Box
default_text = sample_prompt if sample_prompt else "what are you"
user_text = st.text_input(
    "✍️ Enter input prompt:",
    value=default_text,
    placeholder="Type a phrase, quote, or sentence here..."
)

# ---------------------------------------------------------
# Execution Logic
# ---------------------------------------------------------
if mode == "Next Single Word":
    col_btn, _ = st.columns([1, 4])
    with col_btn:
        predict_clicked = st.button("🔮 Predict Next Word", type="primary", use_container_width=True)
    
    if predict_clicked or sample_prompt:
        if not user_text.strip():
            st.warning("⚠️ Please provide a valid sentence or phrase.")
        else:
            with st.spinner("Analyzing sequence context..."):
                best_word, candidates = get_next_word_predictions(user_text, top_k=top_k)
                
            if best_word:
                st.markdown("### 🏆 Prediction Result")
                
                c1, c2 = st.columns([1, 2])
                with c1:
                    st.markdown(f"""
                    <div class="card-box" style="text-align: center;">
                        <span style="font-size: 0.95rem; color: #6B7280; font-weight: 500;">Most Probable Next Word</span><br>
                        <div class="prediction-pill">✨ {best_word}</div>
                        <p style="margin-top: 8px; color: #4B5563; font-size: 0.9rem;">
                            Full sequence: <em>"{user_text} <b style='color:#4F46E5;'>{best_word}</b>"</em>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with c2:
                    st.markdown("<div class='card-box'>", unsafe_allow_html=True)
                    st.markdown("#### 📈 Top Candidate Probabilities")
                    for word, prob in candidates:
                        percent = prob * 100
                        st.write(f"**{word}** ({percent:.2f}%)")
                        st.progress(min(max(prob, 0.0), 1.0))
                    st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.info("No matching words found in vocabulary for this input.")

else:
    # Sentence completion mode
    col_btn, _ = st.columns([1, 4])
    with col_btn:
        generate_clicked = st.button("🚀 Complete Sentence", type="primary", use_container_width=True)
        
    if generate_clicked or sample_prompt:
        if not user_text.strip():
            st.warning("⚠️ Please provide a valid starting phrase.")
        else:
            with st.spinner(f"Generating next {num_words} words using LSTM..."):
                full_generated = generate_continuous_text(
                    seed_text=user_text,
                    n_words=num_words,
                    temperature=temp
                )
            
            st.markdown("### 📜 Generated Sentence")
            st.markdown(f"""
            <div class="generated-text-box">
                {full_generated}
            </div>
            """, unsafe_allow_html=True)

# ---------------------------------------------------------
# Architecture Comparison & Details Section
# ---------------------------------------------------------
st.markdown("---")
with st.expander("🔍 **Model Comparison: Why LSTM over SimpleRNN?**", expanded=False):
    st.markdown("""
    | Metric / Feature | SimpleRNN Baseline | LSTM Model (Active) | Advantage |
    | :--- | :--- | :--- | :--- |
    | **Training Epochs** | 10 Epochs | **100 Epochs** | Deeper convergence |
    | **Training Time (T4 GPU)** | ~7 mins | **~54 mins** | Thorough representation learning |
    | **Training Accuracy** | 20.99% | **74.91%** | **+53.92% Accuracy improvement** |
    | **Training Loss** | 4.2153 | **1.1370** | Significantly lower error |
    | **Long-Term Memory** | Suffers from vanishing gradient | **Gating mechanisms (Forget, Input, Output)** | Captures semantic context across long quotes |
    """)

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("""
<div style="text-align: center; color: #9CA3AF; padding: 20px 0; font-size: 0.85rem;">
    LSTM Next Word Prediction Application • Built with TensorFlow, Keras & Streamlit
</div>
""", unsafe_allow_html=True)