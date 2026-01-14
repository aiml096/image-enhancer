import torch
import tf2onnx
import tensorflow as tf
from model import build_docsr_generator
import os

# Load PyTorch weights
state = torch.load("pretrained/RRDB_ESRGAN_x2.pth", map_location="cpu")

# Build TF model
model = build_docsr_generator(scale=2)

# ⚠ Manual mapping is complex in ESRGAN
# Recommended approach:
# Convert PyTorch → ONNX → TF

dummy = torch.randn(1, 3, 128, 128)
torch.onnx.export(
    state["model"],
    dummy,
    "pretrained/esrgan.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=13
)

print("ONNX exported")

# Convert ONNX → TF
os.system(
    "python -m tf2onnx.convert --onnx pretrained/esrgan.onnx --output pretrained/esrgan_tf"
)
