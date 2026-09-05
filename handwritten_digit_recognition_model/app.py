import os
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Handwritten Digit Recognizer",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1rem;
        color: #888888;
        margin-bottom: 1.5rem;
    }
    .prediction-card {
        background: linear-gradient(135deg, rgba(79, 172, 254, 0.15) 0%, rgba(0, 242, 254, 0.05) 100%);
        border: 1px solid rgba(79, 172, 254, 0.3);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        margin-bottom: 20px;
    }
    .digit-display {
        font-size: 4.5rem;
        font-weight: 900;
        color: #00f2fe;
        line-height: 1;
        margin: 10px 0;
    }
    .confidence-badge {
        display: inline-block;
        background-color: rgba(0, 242, 254, 0.2);
        color: #00f2fe;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 1.1rem;
    }
    .canvas-container {
        display: flex;
        justify-content: center;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Model Loader
# ---------------------------------------------------------
@st.cache_resource
def load_digit_model():
    # Look for model in local folder or parent directory
    possible_paths = [
        Path(__file__).parent / "cnn_model.keras",
        Path(__file__).parent.parent / "cnn_model.keras",
        Path("cnn_model.keras"),
        Path("handwritten_digit_recognition_model/cnn_model.keras"),
    ]
    model_path = None
    for p in possible_paths:
        if p.exists():
            model_path = p
            break

    if model_path is None:
        return None, "Model file 'cnn_model.keras' not found."

    try:
        model = tf.keras.models.load_model(str(model_path))
        return model, str(model_path)
    except Exception as e:
        return None, str(e)


# ---------------------------------------------------------
# Image Preprocessing (MNIST Centered & Scaled)
# ---------------------------------------------------------
def preprocess_drawn_image(rgba_image_data):
    """
    Takes an RGBA uint8 image from canvas, extracts the stroke,
    crops the bounding box, resizes with aspect ratio to fit inside 20x20,
    pads to 28x28 (MNIST standard), and normalizes pixel values to [0, 1].
    """
    if rgba_image_data is None:
        return None, None

    # Convert RGBA to Grayscale
    gray = cv2.cvtColor(rgba_image_data.astype(np.uint8), cv2.COLOR_RGBA2GRAY)

    # Check if user has drawn anything
    if not np.any(gray > 30):
        return None, None

    # Find bounding box of drawing
    y_indices, x_indices = np.where(gray > 30)
    ymin, ymax = int(y_indices.min()), int(y_indices.max())
    xmin, xmax = int(x_indices.min()), int(x_indices.max())
    cropped = gray[ymin : ymax + 1, xmin : xmax + 1]

    # Calculate scale factor to fit inside 20x20 box preserving aspect ratio
    h, w = cropped.shape
    if h > w:
        new_h = 20
        new_w = max(1, int(round(w * 20.0 / h)))
    else:
        new_w = 20
        new_h = max(1, int(round(h * 20.0 / w)))

    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Place centered inside a 28x28 black canvas
    padded = np.zeros((28, 28), dtype=np.float32)
    start_y = (28 - new_h) // 2
    start_x = (28 - new_w) // 2
    padded[start_y : start_y + new_h, start_x : start_x + new_w] = resized

    # Normalize to [0.0, 1.0]
    normalized = padded / 255.0
    input_tensor = normalized.reshape(1, 28, 28, 1)

    return input_tensor, padded.astype(np.uint8)


# ---------------------------------------------------------
# Main App Layout
# ---------------------------------------------------------
st.markdown('<div class="main-title">✍️ Handwritten Digit Recognizer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Draw a digit (0–9) on the canvas below and let the trained CNN model predict it in real-time.</div>',
    unsafe_allow_html=True,
)

# Load model
model, status_info = load_digit_model()

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Canvas Settings")
    stroke_width = st.slider("Stroke Width (Pen Thickness)", min_value=8, max_value=30, value=18, step=2)
    realtime_update = st.toggle("Real-time Prediction", value=True)
    
    st.divider()
    st.header("🧠 Model Details")
    if model is not None:
        st.success(f"Loaded: `{os.path.basename(status_info)}`")
        st.markdown(
            """
            - **Architecture**: Conv2D + MaxPool + Dense
            - **Test Accuracy**: **~99.48%**
            - **Input Size**: 28 x 28 x 1 (Grayscale)
            - **Classes**: 10 (Digits 0 to 9)
            """
        )
    else:
        st.error(f"Error loading model: {status_info}")

    st.divider()
    st.markdown("💡 **Tip**: Draw a single digit clearly in the center of the canvas.")

# Two Column Layout
col_canvas, col_results = st.columns([1.1, 1.3], gap="large")

with col_canvas:
    st.subheader("🎨 Draw Digit Here")
    st.caption("Use your mouse, trackpad, or stylus to draw any digit from 0 to 9:")

    # Streamlit Drawable Canvas
    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 0)",
        stroke_width=stroke_width,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="digit_canvas",
        update_streamlit=realtime_update,
        return_image_data=True,
    )

    col_btn1, col_btn2 = st.columns([1, 1])
    with col_btn1:
        predict_clicked = st.button("🔍 Predict Digit", type="primary", use_container_width=True)

with col_results:
    st.subheader("📊 Prediction Results")

    if model is None:
        st.error("Cannot make predictions because the model is not loaded.")
    elif canvas_result is None:
        st.info("👋 Please draw a digit (0–9) on the canvas to see predictions.")
    else:
        try:
            image_data = canvas_result.image_data
        except Exception:
            image_data = None

        has_drawing = image_data is not None
        should_run = (realtime_update and has_drawing) or predict_clicked

        if should_run and image_data is not None:
            input_tensor, processed_28x28 = preprocess_drawn_image(image_data)

            if input_tensor is None:
                st.info("👋 The canvas is currently empty. Please draw a digit (0–9) to see predictions.")
            else:
                # Model Inference
                predictions = model.predict(input_tensor, verbose=0)[0]
                predicted_digit = int(np.argmax(predictions))
                confidence = float(predictions[predicted_digit]) * 100.0

                # Display Main Prediction Card
                st.markdown(
                    f"""
                    <div class="prediction-card">
                        <div style="font-size: 1.1rem; color: #ccc;">Predicted Digit</div>
                        <div class="digit-display">{predicted_digit}</div>
                        <div class="confidence-badge">Confidence: {confidence:.2f}%</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Show Processed 28x28 representation & Probability Chart
                col_preview, col_chart = st.columns([1, 2.2])

                with col_preview:
                    st.markdown("**Model Input (28x28)**")
                    preview_img = Image.fromarray(processed_28x28)
                    st.image(preview_img, width=110, caption="MNIST 28x28 Input")

                    # Top 3 probabilities
                    top_3_indices = np.argsort(predictions)[-3:][::-1]
                    st.markdown("**Top 3 Guesses:**")
                    for rank, idx in enumerate(top_3_indices, 1):
                        prob = predictions[idx] * 100
                        st.markdown(f"{rank}. **Digit {idx}**: `{prob:.1f}%`")

                with col_chart:
                    st.markdown("**Confidence Distribution (0–9)**")
                    prob_df = pd.DataFrame({
                        "Digit": [str(i) for i in range(10)],
                        "Probability (%)": predictions * 100.0
                    })

                    fig = px.bar(
                        prob_df,
                        x="Digit",
                        y="Probability (%)",
                        text=prob_df["Probability (%)"].apply(lambda v: f"{v:.1f}%" if v > 5 else ""),
                        color="Probability (%)",
                        color_continuous_scale="Blues",
                        range_y=[0, 100],
                    )
                    fig.update_layout(
                        margin=dict(l=10, r=10, t=10, b=10),
                        height=220,
                        coloraxis_showscale=False,
                        xaxis_title="Digit",
                        yaxis_title="Probability (%)",
                    )
                    fig.update_traces(textposition="outside")
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Draw a digit on the left to start recognition.")
