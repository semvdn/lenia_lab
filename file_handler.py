import os
import json
import torch
from tkinter import filedialog, simpledialog, messagebox
from PIL import Image, ImageTk

from config import SETTINGS_FOLDER, PRESETS_FOLDER, device, GRID_DIM
from simulation import SimulationState, Channel, local_param_maps, LOCAL_PARAM_NAMES
from canvas_manager import _get_multichannel_array

def save_settings(app):
    fp = filedialog.asksaveasfilename(defaultextension=".json", initialdir=SETTINGS_FOLDER, filetypes=[("JSON", "*.json")])
    if not fp: return
    with open(fp, 'w') as f:
        json.dump({'channels': [c.__dict__ for c in app.sim_state.channels], 'interaction_matrix': app.sim_state.interaction_matrix}, f, indent=4)

def load_settings(app):
    fp = filedialog.askopenfilename(initialdir=SETTINGS_FOLDER, filetypes=[("JSON", "*.json")])
    if not fp: return
    with open(fp, 'r') as f:
        settings = json.load(f)
    
    new_state = SimulationState()
    new_state.interaction_matrix = settings.get('interaction_matrix', [[1.0]])
    new_state.channels = [Channel(**d) for d in settings.get('channels', [])]
    app.sim_state = new_state
    
    app.game_board = app._initialize_board_circle_seed()
    app._initialize_all_local_param_maps()
    app.draw_channel_index = 0
    app._build_ui()

def load_presets(app):
    presets = {}
    if not os.path.isdir(PRESETS_FOLDER): return presets
    for fn in os.listdir(PRESETS_FOLDER):
        if fn.endswith(".json"):
            try:
                with open(os.path.join(PRESETS_FOLDER, fn), 'r') as f:
                    presets[fn.replace(".json", "")] = json.load(f)
            except Exception as e:
                print(f"Error loading preset {fn}: {e}")
    return presets

def update_preset_listbox(app):
    app.preset_listbox.delete(0, 'end')
    for name in sorted(app.organism_presets.keys()):
        app.preset_listbox.insert('end', name)
    update_preset_preview(app)

def save_preset(app):
    if not app.selected_organism_id or app.selected_organism_id not in app.persistent_tracked_organisms:
        messagebox.showwarning("Save Error", "No organism selected to save.")
        return
    name = simpledialog.askstring("Save Preset", "Enter preset name:", parent=app.root)
    if name and name.strip():
        _save_preset_logic(app, name)

def _save_preset_logic(app, name):
    if not app.selected_organism_id or app.selected_organism_id not in app.persistent_tracked_organisms: return
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
    
    with open(os.path.join(PRESETS_FOLDER, f"{name}.json"), 'w') as f:
        json.dump(preset_data, f, indent=2)
    app.organism_presets[name] = preset_data
    update_preset_listbox(app)

def load_preset(app):
    sel = app.preset_listbox.curselection()
    if not sel: return
    preset = app.organism_presets[app.preset_listbox.get(sel[0])]
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
    sel = app.preset_listbox.curselection()
    if not sel: return
    name = app.preset_listbox.get(sel[0])
    if name in app.organism_presets and messagebox.askyesno("Confirm Delete", f"Delete preset '{name}'?"):
        del app.organism_presets[name]
        os.remove(os.path.join(PRESETS_FOLDER, f"{name}.json"))
        update_preset_listbox(app)

def rename_preset(app):
    sel = app.preset_listbox.curselection()
    if not sel: return
    old_name = app.preset_listbox.get(sel[0])
    new_name = simpledialog.askstring("Rename Preset", "Enter new name:", initialvalue=old_name, parent=app.root)
    if not new_name or not new_name.strip() or new_name == old_name: return
    if new_name in app.organism_presets:
        messagebox.showerror("Rename Error", "A preset with that name already exists.")
        return
    app.organism_presets[new_name] = app.organism_presets.pop(old_name)
    os.rename(os.path.join(PRESETS_FOLDER, f"{old_name}.json"), os.path.join(PRESETS_FOLDER, f"{new_name}.json"))
    update_preset_listbox(app)

def update_preset_preview(app, event=None):
    sel = app.preset_listbox.curselection()
    if not sel:
        app.preset_preview_label.config(image='')
        return
    name = app.preset_listbox.get(sel[0])
    preset = app.organism_presets[name]
    channels = [Channel(**c) for c in preset['params']['channels']]
    arr = _get_multichannel_array(torch.tensor(preset['tensor'], device=device), channels)
    if arr is not None:
        img = Image.fromarray(arr).resize((100, 100), Image.NEAREST)
        app.preset_preview_photo = ImageTk.PhotoImage(image=img)
        app.preset_preview_label.config(image=app.preset_preview_photo)