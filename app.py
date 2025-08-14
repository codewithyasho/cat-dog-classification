import os
# Suppress TF/absl/protobuf noisy logs before any TF/Keras import
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # 0=all, 3=errors only
# os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")  # uncomment if you prefer to disable oneDNN note entirely

import time
from typing import Tuple, Optional, Callable

import numpy as np
from PIL import Image
import streamlit as st
import warnings
import logging

# Quiet common noisy loggers
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
for name in ("tensorflow", "absl", "google", "google.protobuf"):
    try:
        logging.getLogger(name).setLevel(logging.ERROR)
    except Exception:
        pass
try:
    import absl.logging as absl_logging  # type: ignore
    absl_logging.set_verbosity(absl_logging.ERROR)
except Exception:
    pass

# Try keras 3 first, then fall back to tf.keras for broader compatibility
# Tuple[backend_name, loader_fn]
_LOADERS: tuple[Optional[str], Optional[Callable]] = (None, None)
try:  # keras 3
    from keras.models import load_model as _keras_load_model  # type: ignore
    _LOADERS = ("keras", _keras_load_model)
except Exception:  # pragma: no cover
    try:
        from tensorflow.keras.models import load_model as _tf_load_model  # type: ignore
        _LOADERS = ("tf.keras", _tf_load_model)
    except Exception as e:  # pragma: no cover
        _LOADERS = (None, None)


@st.cache_resource(show_spinner=True)
def get_model(model_path: str = "model.h5"):
    backend, loader = _LOADERS
    if loader is None:
        raise RuntimeError(
            "Neither 'keras' nor 'tensorflow.keras' load_model is available. Install 'keras' or 'tensorflow'."
        )
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    with st.spinner(f"Loading model using {backend}…"):
        assert loader is not None
        model = loader(model_path)
    return model


def infer_input_size(model) -> Tuple[int, int, bool]:
    """
    Infer (height, width, channels_last) from model.input_shape.
    Defaults to (150, 150, True) if ambiguous.
    """
    h, w, channels_last = 150, 150, True
    input_shape = getattr(model, "input_shape", None)
    if isinstance(input_shape, (list, tuple)) and len(input_shape) > 0 and isinstance(input_shape[0], (list, tuple)):
        input_shape = input_shape[0]
    if input_shape and len(input_shape) >= 4:
        if input_shape[-1] in (1, 3, 4):
            # channels last: (None, H, W, C)
            h = input_shape[1] or h
            w = input_shape[2] or w
            channels_last = True
        elif input_shape[1] in (1, 3, 4):
            # channels first: (None, C, H, W)
            h = input_shape[2] or h
            w = input_shape[3] or w
            channels_last = False
    return int(h), int(w), bool(channels_last)


def prepare_image(img: Image.Image, target_size: Tuple[int, int], channels_last: bool) -> np.ndarray:
    # Ensure RGB
    img = img.convert("RGB")
    img = img.resize(target_size)
    arr = np.asarray(img).astype("float32") / 255.0  # (H, W, 3)
    if channels_last:
        arr = np.expand_dims(arr, axis=0)  # (1, H, W, 3)
    else:
        arr = np.transpose(arr, (2, 0, 1))  # (3, H, W)
        arr = np.expand_dims(arr, axis=0)  # (1, 3, H, W)
    return arr


def get_probability(preds: np.ndarray) -> float:
    # Supports sigmoid (1 unit) or softmax (2 units) heads
    preds = np.array(preds)
    if preds.ndim == 1:
        # e.g., shape (1,)
        return float(preds[0])
    if preds.ndim == 2:
        if preds.shape[1] == 1:  # sigmoid
            return float(preds[0, 0])
        if preds.shape[1] == 2:  # softmax [cat, dog] (assumed)
            return float(preds[0, 1])
        # Fallback: take the max class probability as 'dog'
        return float(preds[0].max())
    # Last resort
    return float(preds.ravel()[0])


st.set_page_config(page_title="Cat vs Dog Classifier",
                   page_icon="🐾", layout="centered")
st.set_option('client.showErrorDetails', False)
st.title("🐾 Cat vs Dog Classifier")
st.write("Upload an image, and the model will predict whether it's a Cat or a Dog.")

# Sidebar: settings
with st.sidebar:
    st.header("Settings")
    model_path = st.text_input("Model file", value="model.h5")
    threshold = st.slider(
        "Decision threshold (Dog if ≥ threshold)", 0.05, 0.95, 0.5, 0.01)
    st.caption("Tip: If predictions seem biased, try adjusting the threshold.")

# Load model lazily
model = None
load_error = None
try:
    model = get_model(model_path)
except Exception as e:
    load_error = str(e)

uploaded = st.file_uploader("Choose an image…", type=[
                            "jpg", "jpeg", "png", "webp"])

col1, col2 = st.columns([1, 1])
with col1:
    if uploaded:
        try:
            _preview = Image.open(uploaded)
        except Exception:
            _preview = uploaded  # fall back, Streamlit will handle bytes-like
        st.image(_preview, caption="Uploaded image", use_container_width=True)
    else:
        st.info("No image uploaded yet.")

with col2:
    if load_error:
        st.error(f"Model couldn't be loaded: {load_error}")
    elif model is None:
        st.warning("Model not loaded yet.")
    elif uploaded is None:
        st.info("Upload an image to get a prediction.")
    else:
        # Predict button
        if st.button("Predict", type="primary"):
            try:
                img = Image.open(uploaded)
                h, w, channels_last = infer_input_size(model)
                x = prepare_image(img, (w, h), channels_last)
                start = time.time()
                preds = model.predict(x, verbose=0)
                elapsed = time.time() - start
                proba_dog = get_probability(preds)
                proba_cat = 1.0 - proba_dog
                label = "Dog" if proba_dog >= threshold else "Cat"
                conf = proba_dog if label == "Dog" else proba_cat

                st.success(f"Prediction: {label} (confidence: {conf:.2%})")
                st.metric(label="Dog probability", value=f"{proba_dog:.2%}")
                st.caption(
                    f"Model input expected: {(h, w)} | channels_last={channels_last} | time: {elapsed*1000:.1f} ms")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

st.divider()
st.caption("This app loads a pre-trained Keras model ('model.h5') and performs a simple binary classification.")
