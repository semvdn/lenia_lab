# Lenia Lab
A Tkinter-based Lenia sandbox for building multi-channel cellular automata, tracking organisms, and recording results.

## Showcase
<table>
  <tr>
    <td><sub>1-channel cells (full board)</sub><br><img src="showcase/1ch_cells.gif" alt="1-channel cells" width="380"></td>
    <td><sub>3-channel cells (full board)</sub><br><img src="showcase/3ch_cells.gif" alt="3-channel cells" width="380"></td>
  </tr>
  <tr>
    <td colspan="2" align="center"><sub>3-channel split views</sub><br><img src="showcase/3ch_cells_split.gif" alt="3-channel split views" width="760"></td>
  </tr>
</table>

## Requirements
- Python 3.10+ with Tkinter available
- Packages: `torch`, `numpy`, `pillow`, `matplotlib`, `scikit-image`, `imageio`, `pandas`
- GPU is optional; PyTorch will use CUDA if available

Install (example):
```
python -m venv .venv
.venv\Scripts\activate
pip install torch numpy pillow matplotlib scikit-image imageio pandas
# For the best PyTorch wheel, see https://pytorch.org/get-started/locally/
```

## Run
```
python app.py
```

## Data & folders
- `presets/` - bundled presets shipped with the app
- `saved_settings/` - your saved simulation settings (`*.json`)
- `saved_settings/organisms/` - saved organisms/presets you capture from the board (`*.json`)
- `gifs/` - recordings written by the recorder (GIFs, stats, and snapshots)

The preset list shows everything together with prefixes:
- `[Preset]` built-in items from `presets/`
- `[Saved Organism]` your captured organisms from `saved_settings/organisms/`
- `[Saved Settings]` full simulation configs from `saved_settings/`

## Quick usage
- Spacebar toggles pause/play.
- Left-drag draws mass on the selected channel; right-drag erases; Shift+Click selects an organism (when tracking is enabled).
- Add and configure channels in the "Channels & Kernels" tab; adjust interactions via the matrix.
- Use the visualization dropdowns to switch between final board, potential, growth, and flow fields; toggle split-screen or zoom view as needed.

## Saving & loading
- **Save Settings** stores the entire simulation configuration to `saved_settings/<name>.json`.
- **Load Settings** restores a saved configuration from that folder.
- **Save Selected** (in the presets panel) saves the currently selected organism to `saved_settings/organisms/<name>.json`.
- The list allows loading any entry; built-in presets are read-only, while saved organisms/settings can be renamed or deleted.

## Recording
- **Record Full GIF** captures the rendered board to a GIF under `gifs/`.
- **Record Organism Stats** logs per-frame metrics and writes cropped GIFs for multiple view modes into a timestamped folder under `gifs/`.
- Recording resolution is set via `RECORDING_RESOLUTION` in `config.py` (set to `None` to keep native size).

## AI assistance
- Some scripting and refactors were produced with help from Google's Gemini 2.5 Pro.

## Notes
- The first run creates needed folders automatically.
- If PyTorch cannot find CUDA, it falls back to CPU automatically.
