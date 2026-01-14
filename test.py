import streamlit as st
import torch
import cv2
import numpy as np
from model2 import RRDBNet

# -----------------------------
# Config
# -----------------------------
MODEL_PATH = "RealESRGAN_x4.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

st.set_page_config(page_title="Greeting Card Enhancer", layout="centered")

st.title("🖼️ Greeting Card Image Enhancer")
st.write("Upload a low-resolution greeting card image and get high-quality output")

# -----------------------------
# Load model
# -----------------------------
@st.cache_resource
def load_model():
    model = RRDBNet()
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# -----------------------------
# Image upload
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload low-resolution image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    st.subheader("Input Image")
    st.image(img, use_container_width=True)

    # Preprocess
    img_f = img.astype(np.float32) / 255.0
    img_t = torch.from_numpy(img_f).permute(2, 0, 1).unsqueeze(0)
    img_t = img_t.to(DEVICE)

    # Inference
    with torch.no_grad():
        output = model(img_t)

    # Postprocess
    output = output.squeeze(0).permute(1, 2, 0)
    output = torch.clamp(output, 0.0, 1.0)
    output = (output.cpu().numpy() * 255).astype(np.uint8)

    st.subheader("Enhanced Image")
    st.image(output, use_container_width=True)

    # Download
    _, buffer = cv2.imencode(".png", cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
    st.download_button(
        label="⬇️ Download Enhanced Image",
        data=buffer.tobytes(),
        file_name="enhanced.png",
        mime="image/png"
    )
