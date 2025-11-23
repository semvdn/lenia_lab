import os
import json
import datetime
import imageio
import pandas as pd
import numpy as np
from tkinter import filedialog, simpledialog, messagebox
from PIL import Image
from config import GIFS_FOLDER, GRID_DIM, RECORDING_RESOLUTION
from file_handler import _save_preset_logic
from organism_tracker import masses_for_label
from canvas_manager import _get_view_array_by_name

def _target_size():
    if not RECORDING_RESOLUTION:
        return None
    if isinstance(RECORDING_RESOLUTION, (list, tuple)) and len(RECORDING_RESOLUTION) == 2:
        return (int(RECORDING_RESOLUTION[0]), int(RECORDING_RESOLUTION[1]))
    return (int(RECORDING_RESOLUTION), int(RECORDING_RESOLUTION))

def _resize_frame(arr):
    size = _target_size()
    if not size:
        return arr
    return np.array(Image.fromarray(arr).resize(size, Image.NEAREST))

def record_gif(app):
    app.is_recording = not app.is_recording
    app.record_button.config(text="Stop & Save GIF" if app.is_recording else "Record Full GIF")
    if not app.is_recording and app.gif_frames:
        fp = filedialog.asksaveasfilename(defaultextension=".gif", initialdir=GIFS_FOLDER, filetypes=[("GIF", "*.gif")])
        if fp:
            frames = [_resize_frame(f) for f in app.gif_frames]
            imageio.mimsave(fp, frames, fps=30, loop=0)
        app.gif_frames = []

def record_organism_stats(app):
    if not app.selected_organism_id and not app.is_stats_recording:
        messagebox.showwarning("Recording Error", "An organism must be selected to begin recording.")
        return
    app.is_stats_recording = not app.is_stats_recording
    app.stats_record_button.config(text="Stop & Log Stats" if app.is_stats_recording else "Record Organism Stats")
    
    if app.is_stats_recording:
        session_name = simpledialog.askstring("Recording Session", "Enter a name for this session:",
                                              initialvalue=f"rec_{datetime.datetime.now():%Y%m%d_%H%M%S}", parent=app.root)
        if not session_name or not session_name.strip():
            app.is_stats_recording = False
            app.stats_record_button.config(text="Record Organism Stats")
            return
        
        _save_preset_logic(app, session_name)
        app.rec_dir = os.path.join(GIFS_FOLDER, session_name)
        app._ensure_dir(app.rec_dir)
        
        with open(os.path.join(app.rec_dir, "parameters.json"), 'w') as f:
            json.dump({'channels': [c.__dict__ for c in app.sim_state.channels], 'interaction_matrix': app.sim_state.interaction_matrix}, f, indent=4)
            
        app.stats_log_path = os.path.join(app.rec_dir, "stats_log.csv")
        header = ['frame', *[f'mass_ch{i+1}' for i in range(len(app.sim_state.channels))], 'velocity', 'direction', 'division_event', 'parent_id', 'mass_ratio']
        app.stats_log = [header]
        
        view_modes = ["Final Board", "Potential Field", "Growth Field", "Flow Field"]
        app.stats_gif_writers = {v: imageio.get_writer(os.path.join(app.rec_dir, f"{v.lower().replace(' ', '_')}.gif"), mode='I', loop=0) for v in view_modes}
    else:
        for writer in app.stats_gif_writers.values():
            writer.close()
        pd.DataFrame(app.stats_log[1:], columns=app.stats_log[0]).to_csv(app.stats_log_path, index=False)
        app.stats_gif_writers = {}
        app.stats_log = []

def log_stats_and_gifs(app):
    if not app.selected_organism_id or app.selected_organism_id not in app.persistent_tracked_organisms:
        record_organism_stats(app)
        return
        
    org_data = app.persistent_tracked_organisms[app.selected_organism_id]
    masses = masses_for_label(app, org_data.get('label_id'))
    vy, vx = org_data['smooth_vel']
    vel = np.sqrt(vx**2 + vy**2)
    direction = np.degrees(np.arctan2(-vy, vx))
    
    div_event, parent_id, mass_ratio = 0, -1, 0.0
    if (pid := org_data.get('parent_id')) is not None:
        for event in app.division_events:
            if event['parent_id'] == pid and app.selected_organism_id in event['children']:
                div_event = 1
                parent_id = pid
                mass_ratio = event['children'][app.selected_organism_id] / event['parent_mass'] if event['parent_mass'] > 0 else 0
                del event['children'][app.selected_organism_id]
                if not event['children']: app.division_events.remove(event)
                break
    app.stats_log.append([len(app.stats_log), *masses, vel, direction, div_event, parent_id, mass_ratio])

    bbox = org_data.get('bbox')
    if not bbox:
        crop_box = (0, 0, GRID_DIM[1], GRID_DIM[0])
    else:
        min_r, min_c, max_r, max_c = bbox
        pad = 30
        crop_box = (max(0, min_c - pad), max(0, min_r - pad), min(GRID_DIM[1], max_c + pad), min(GRID_DIM[0], max_r + pad))
    
    for view_mode, writer in app.stats_gif_writers.items():
        ch_idx = app.draw_channel_index if app.draw_channel_index >= 0 else 0
        arr = _get_view_array_by_name(app, f"Ch {ch_idx + 1}: {view_mode}")
        if arr is not None:
            frame = Image.fromarray(arr).crop(crop_box)
            writer.append_data(_resize_frame(np.array(frame)))
