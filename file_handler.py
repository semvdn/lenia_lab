import os
import json
import torch
from tkinter import filedialog, simpledialog, messagebox
from PIL import Image, ImageTk

from config import SETTINGS_FOLDER, PRESETS_FOLDER, SAVED_ORGANISMS_FOLDER, device, GRID_DIM
from simulation import SimulationState, Channel, local_param_maps, LOCAL_PARAM_NAMES
from canvas_manager import _get_multichannel_array

ENTRY_TYPE_LABELS = {
    'preset': 'Preset',
    'saved_organism': 'Saved Organism',
    'saved_settings': 'Saved Settings'
}
ENTRY_SORT_ORDER = {'preset': 0, 'saved_organism': 1, 'saved_settings': 2}

def _display_name(entry_type, name):
    """Formats a display label for listbox entries based on type."""
    label = ENTRY_TYPE_LABELS.get(entry_type, "Preset")
    return f"[{label}] {name}"

def _load_json_files(folder, entry_type, skip_dirs=True):
    """Loads JSON files from a folder and returns a mapping of display names to entry data."""
    entries = {}
    if not os.path.isdir(folder):
        return entries
    for fn in os.listdir(folder):
        path = os.path.join(folder, fn)
        if skip_dirs and os.path.isdir(path):
            continue
        if not fn.endswith(".json"):
            continue
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            base = fn[:-5]
            display = _display_name(entry_type, base)
            entries[display] = {'name': base, 'type': entry_type, 'data': data, 'path': path}
        except Exception as e:
            print(f"Error loading {entry_type} {fn}: {e}")
    return entries

def _refresh_presets(app):
    """Reloads preset metadata from disk and repopulates the listbox."""
    app.organism_presets = load_presets(app)
    update_preset_listbox(app)

def _apply_settings_data(app, settings):
    """Replaces the simulation state with values loaded from settings data."""
    new_state = SimulationState()
    new_state.interaction_matrix = settings.get('interaction_matrix', [[1.0]])
    new_state.channels = [Channel(**d) for d in settings.get('channels', [])]
    app.sim_state = new_state

    app.game_board = app._initialize_board_circle_seed()
    app._initialize_all_local_param_maps()
    app.draw_channel_index = 0
    app._build_ui()

def _load_settings_from_path(app, fp):
    """Loads a settings JSON file from disk and applies it to the app."""
    with open(fp, 'r') as f:
        settings = json.load(f)
    _apply_settings_data(app, settings)
    return settings

def save_settings(app):
    """Prompts the user for a settings name and writes the current state to disk."""
    name = simpledialog.askstring("Save Settings", "Enter settings name:", parent=app.root)
    if not name or not name.strip():
        return
    safe_name = name.strip()
    fp = os.path.join(SETTINGS_FOLDER, f"{safe_name}.json")
    with open(fp, 'w') as f:
        json.dump({'channels': [c.__dict__ for c in app.sim_state.channels], 'interaction_matrix': app.sim_state.interaction_matrix}, f, indent=4)
    _refresh_presets(app)

def load_settings(app):
    """Opens a file picker and applies a chosen settings file."""
    fp = filedialog.askopenfilename(initialdir=SETTINGS_FOLDER, filetypes=[("JSON", "*.json")])
    if not fp:
        return
    _load_settings_from_path(app, fp)

def load_presets(app=None):
    """Aggregates built-in, saved organism, and saved settings entries into one dict."""
    presets = {}
    presets.update(_load_json_files(PRESETS_FOLDER, 'preset'))
    presets.update(_load_json_files(SAVED_ORGANISMS_FOLDER, 'saved_organism'))
    presets.update(_load_json_files(SETTINGS_FOLDER, 'saved_settings'))
    return presets

def update_preset_listbox(app):
    """Refreshes the preset listbox contents using the current preset metadata."""
    app.preset_listbox.delete(0, 'end')
    sorted_items = sorted(app.organism_presets.items(), key=lambda item: (ENTRY_SORT_ORDER.get(item[1]['type'], 99), item[1]['name'].lower()))
    for display, _ in sorted_items:
        app.preset_listbox.insert('end', display)
    update_preset_preview(app)

def save_preset(app):
    """Prompts for a name and saves the currently selected tracked organism."""
    if not app.selected_organism_id or app.selected_organism_id not in app.persistent_tracked_organisms:
        messagebox.showwarning("Save Error", "No organism selected to save.")
        return
    name = simpledialog.askstring("Save Preset", "Enter preset name:", parent=app.root)
    if name and name.strip():
        _save_preset_logic(app, name)

def _save_preset_logic(app, name):
    """Persists the selected organism tensor and metadata under the provided name."""
    if not app.selected_organism_id or app.selected_organism_id not in app.persistent_tracked_organisms:
        return
    org_data = app.persistent_tracked_organisms[app.selected_organism_id]
    bbox = org_data.get('bbox')
    if not bbox:
        messagebox.showwarning("Save Error", "Selected organism has no bounding box data.")
        return

    min_r, min_c, max_r, max_c = bbox
    tensor_slice = app.game_board[:, min_r:max_r, min_c:max_c]

    local_maps_data = {}
    for ch in app.sim_state.channels:
        if ch.has_local_params and ch.id in local_param_maps:
            channel_slices = {
                name: local_param_maps[ch.id][name][min_r:max_r, min_c:max_c].detach().cpu().tolist()
                for name in LOCAL_PARAM_NAMES
            }
            local_maps_data[ch.id] = channel_slices

    preset_data = {
        'name': name,
        'tensor': tensor_slice.detach().cpu().tolist(),
        'params': {'channels': [c.__dict__ for c in app.sim_state.channels], 'interaction_matrix': app.sim_state.interaction_matrix},
        'local_param_maps': local_maps_data
    }

    with open(os.path.join(SAVED_ORGANISMS_FOLDER, f"{name}.json"), 'w') as f:
        json.dump(preset_data, f, indent=2)
    _refresh_presets(app)

def load_preset(app):
    """Loads the selected preset or settings entry into the simulation and UI."""
    sel = app.preset_listbox.curselection()
    if not sel:
        return
    display_name = app.preset_listbox.get(sel[0])
    entry = app.organism_presets.get(display_name)
    if not entry:
        return

    if entry['type'] == 'saved_settings':
        _apply_settings_data(app, entry['data'])
        app.update_canvas()
        return

    preset = entry['data']
    p_data = preset['params']

    new_state = SimulationState()
    new_state.interaction_matrix = p_data['interaction_matrix']
    new_state.channels = [Channel(**d) for d in p_data['channels']]
    app.sim_state = new_state

    app.game_board = app._clear_board()
    app._initialize_all_local_param_maps()

    tensor_data = torch.tensor(preset['tensor'], device=device)
    h, w = tensor_data.shape[1], tensor_data.shape[2]
    ch, cw = GRID_DIM[0] // 2, GRID_DIM[1] // 2
    y_start, x_start = ch - h // 2, cw - w // 2
    y_end, x_end = y_start + h, x_start + w
    app.game_board[:, y_start:y_end, x_start:x_end] = tensor_data

    for ch_id, channel_slices in preset.get('local_param_maps', {}).items():
        if ch_id in local_param_maps:
            for param_name, slice_data in channel_slices.items():
                slice_tensor = torch.tensor(slice_data, device=device)
                if slice_tensor.shape == (h, w):
                    local_param_maps[ch_id][param_name][y_start:y_end, x_start:x_end] = slice_tensor

    app.draw_channel_index = 0
    app._build_ui()
    app.update_canvas()

def delete_preset(app):
    """Deletes a user-created preset after confirmation, preserving built-ins."""
    sel = app.preset_listbox.curselection()
    if not sel:
        return
    display_name = app.preset_listbox.get(sel[0])
    entry = app.organism_presets.get(display_name)
    if not entry:
        return
    if entry['type'] == 'preset':
        messagebox.showinfo("Delete Not Allowed", "Built-in presets cannot be deleted.")
        return
    if messagebox.askyesno("Confirm Delete", f"Delete '{display_name}'?"):
        os.remove(entry['path'])
        _refresh_presets(app)

def rename_preset(app):
    """Renames a user-created preset, ensuring conflicts are avoided."""
    sel = app.preset_listbox.curselection()
    if not sel:
        return
    display_name = app.preset_listbox.get(sel[0])
    entry = app.organism_presets.get(display_name)
    if not entry:
        return
    if entry['type'] == 'preset':
        messagebox.showinfo("Rename Not Allowed", "Built-in presets cannot be renamed.")
        return
    new_name = simpledialog.askstring("Rename Preset", "Enter new name:", initialvalue=entry['name'], parent=app.root)
    if not new_name or not new_name.strip():
        return
    safe_name = new_name.strip()
    target_path = os.path.join(os.path.dirname(entry['path']), f"{safe_name}.json")
    if os.path.exists(target_path):
        messagebox.showerror("Rename Error", "A file with that name already exists.")
        return
    os.rename(entry['path'], target_path)
    _refresh_presets(app)

def update_preset_preview(app, event=None):
    """Updates the preview panel to reflect the currently selected preset entry."""
    sel = app.preset_listbox.curselection()
    if not sel:
        app.preset_preview_label.config(image='', text='')
        app.preset_preview_photo = None
        return
    display_name = app.preset_listbox.get(sel[0])
    entry = app.organism_presets.get(display_name)
    app.preset_preview_label.config(image='', text='')
    app.preset_preview_photo = None
    if not entry:
        return
    if entry['type'] == 'saved_settings':
        num_channels = len(entry['data'].get('channels', []))
        app.preset_preview_label.config(text=f"Saved Settings\nChannels: {num_channels}")
        return
    preset = entry['data']
    channels = [Channel(**c) for c in preset['params']['channels']]
    arr = _get_multichannel_array(torch.tensor(preset['tensor'], device=device), channels)
    if arr is not None:
        img = Image.fromarray(arr).resize((100, 100), Image.NEAREST)
        app.preset_preview_photo = ImageTk.PhotoImage(image=img)
        app.preset_preview_label.config(image=app.preset_preview_photo)
