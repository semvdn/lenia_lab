import torch
import os

# --- Main Configuration ---
UI_PANEL_WIDTH = 420
GRID_SIZE = 800
GRID_DIM = (200, 200)
KERNEL_SIZE = 31

SETTINGS_FOLDER = "saved_settings"
GIFS_FOLDER = "gifs"
PRESETS_FOLDER = os.path.join(SETTINGS_FOLDER, "presets")

# --- Device Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == 'cuda':
    try:
        torch.zeros(1).to(device)
        print("CUDA device is available. Using GPU.")
    except RuntimeError as e:
        print(f"--- CUDA WARNING ---\n{e}\nFalling back to CPU.")
        device = torch.device("cpu")
else:
    print("No CUDA device found. Using CPU.")