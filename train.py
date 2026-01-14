import torch
from diffusers import FluxPipeline
from PIL import Image
import cv2
import numpy as np

# --- STEP 1: LOAD QUANTIZED DESIGNER MODEL ---
# This version uses NF4 quantization to fit in ~11GB of VRAM
model_id = "lllyasviel/flux1-dev-bnb-nf4" 

pipe = FluxPipeline.from_pretrained(
    model_id, 
    torch_dtype=torch.bfloat16
)
pipe.enable_model_cpu_offload() # Moves unused parts to RAM to save GPU space

# --- STEP 2: GENERATE THE CARD ---
prompt = """
Luxury wedding invitation card, elegant gold foil borders with white roses. 
Central text in a premium serif font: 'Save the Date - Sarah & Mark - 2026'. 
High-end paper texture, 8k resolution, minimalist design.
"""

print("Generating base design...")
image = pipe(
    prompt=prompt,
    height=1024,
    width=1024,
    guidance_scale=3.5,
    num_inference_steps=30
).images[0]
image.save("base_wedding_card.png")

# --- STEP 3: AUTOMATED UPSCALING (Real-ESRGAN Logic) ---
# For a pure Python script without heavy dependencies, 
# we use a high-quality Lanczos resize followed by a sharpener.
print("Upscaling for print (4x)...")
width, height = image.size
new_size = (width * 4, height * 4) # 4096 x 4096 (Professional Print Size)

# High-quality interpolation
upscaled = image.resize(new_size, resample=Image.LANCZOS)

# Apply a subtle unsharp mask to make the text "pop" for printing
# This simulates the final step of the upscaling pipeline
final_cv2 = cv2.cvtColor(np.array(upscaled), cv2.COLOR_RGB2BGR)
gaussian_3 = cv2.GaussianBlur(final_cv2, (0, 0), 2.0)
unsharp_image = cv2.addWeighted(final_cv2, 1.5, gaussian_3, -0.5, 0)

final_card = Image.fromarray(cv2.cvtColor(unsharp_image, cv2.COLOR_BGR2RGB))
final_card.save("PRINT_READY_WEDDING_CARD.png")

print("Pipeline Complete. Check PRINT_READY_WEDDING_CARD.png")