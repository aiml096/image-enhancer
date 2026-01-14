import os
from huggingface_hub import snapshot_download

# Define your project folder
PROJECT_DIR = "./wedding_card_ai"
MODEL_DIR = os.path.join(PROJECT_DIR, "models/flux_onnx")

def download_low_vram_flux():
    print("Starting download for 8GB VRAM optimized FLUX...")
    
    # We use allow_patterns to ignore the heavy BF16/FP8 files 
    # and only get the FP4 version and necessary configs.
    snapshot_download(
        repo_id="black-forest-labs/FLUX.1-dev-onnx",
        local_dir=MODEL_DIR,
        allow_patterns=[
            "transformer_fp4/*",  # The 4-bit core (approx 7.5GB)
            "vae/*",              # The image decoder
            "clip/*",             # Text encoder 1
            "t5_fp8/*",           # Text encoder 2 (quantized)
            "scheduler/*",        # The math logic
            "*.json"              # Config files
        ],
        ignore_patterns=["*.onnx_data", "transformer_bf16/*", "transformer_fp8/*"]
    )
    print(f"Download complete! Models saved in {MODEL_DIR}")

if __name__ == "__main__":
    download_low_vram_flux()