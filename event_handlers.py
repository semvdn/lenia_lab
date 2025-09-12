import tkinter as tk
import uuid
import copy
import torch
from simulation import Channel, local_param_maps

def update_slider_val(app, label, item_id, attr, v_str, is_layer, is_int):
    v = float(v_str)
    label.config(text=f"{int(v)}" if is_int else f"{v:.2f}")
    v = int(v) if is_int else v
    
    ch = app._get_channel_by_any_id(item_id)
    if is_layer:
        update_layer_attr(app, item_id, attr, v)
    else:
        update_channel_attr(app, item_id, attr, v)
        
    if ch:
        app._update_kernel_preview(ch.id)

def update_channel_attr(app, id, attr, val):
    """Updates an attribute for a given channel."""
    if (ch := app._get_channel_by_id(id)):
        setattr(ch, attr, val)

def update_layer_attr(app, id, attr, val):
    """Updates an attribute for a given kernel layer."""
    if (layer := app._get_layer_by_id(id)):
        layer[attr] = val

def update_interaction(app, i, j, v_str):
    app.sim_state.interaction_matrix[i][j] = float(v_str)

def update_brush_size(app, v_str):
    app.draw_brush_size = int(float(v_str))

def on_draw_channel_selected(app, event):
    app.draw_channel_index = int(app.draw_channel_var.get()) - 1
    app._update_local_param_draw_ui()
    app._update_vis_options()

def add_channel(app):
    colors = ["#FF0000", "#00FF00", "#FFFF00", "#FF00FF", "#FFFFFF"]
    app.sim_state.channels.append(Channel(color_hex=colors[len(app.sim_state.channels) % len(colors)]))
    for row in app.sim_state.interaction_matrix:
        row.append(0.0)
    app.sim_state.interaction_matrix.append([0.0] * (len(app.sim_state.channels) - 1) + [1.0])
    app.game_board = app._initialize_board_circle_seed()
    app._build_ui()

def delete_channel(app, id):
    if len(app.sim_state.channels) <= 1: return
    idx = next((i for i, c in enumerate(app.sim_state.channels) if c.id == id), None)
    if idx is not None:
        if id in local_param_maps:
            del local_param_maps[id]
        app.sim_state.channels.pop(idx)
        app.sim_state.interaction_matrix.pop(idx)
        for row in app.sim_state.interaction_matrix: row.pop(idx)
        app.game_board = app._initialize_board_circle_seed()
        app.draw_channel_index = min(app.draw_channel_index, len(app.sim_state.channels) - 1)
        app._build_ui()

def duplicate_channel(app, id):
    src = app._get_channel_by_id(id)
    if not src: return
    idx = app.sim_state.channels.index(src)
    new = copy.deepcopy(src)
    new.id = str(uuid.uuid4())
    for l in new.kernel_layers: l['id'] = str(uuid.uuid4())
    app.sim_state.channels.insert(idx + 1, new)
    if new.has_local_params: app._initialize_local_param_maps(new)
    for row in app.sim_state.interaction_matrix: row.insert(idx + 1, row[idx])
    app.sim_state.interaction_matrix.insert(idx + 1, app.sim_state.interaction_matrix[idx][:])
    app.game_board = torch.cat((app.game_board[:idx + 1], app.game_board[idx].unsqueeze(0), app.game_board[idx + 1:]), dim=0)
    app._build_ui()

def add_kernel_layer(app, id):
    if (ch := app._get_channel_by_id(id)):
        ch.kernel_layers.append({'id': str(uuid.uuid4()), 'type': 'Gaussian Ring', 'radius': 5, 'weight': 1.0, 'op': '+', 'is_active': True})
        app._rebuild_channels_ui()

def clear_kernel_layers(app, id):
    if (ch := app._get_channel_by_id(id)):
        ch.kernel_layers = [{'id': str(uuid.uuid4()), 'type': 'Gaussian Ring', 'radius': 13, 'weight': 1.0, 'op': '+', 'is_active': True}]
        app._rebuild_channels_ui()

def remove_layer(app, id):
    ch = app._get_channel_by_any_id(id)
    if ch and len(ch.kernel_layers) > 1:
        ch.kernel_layers = [l for l in ch.kernel_layers if l['id'] != id]
        app._rebuild_channels_ui()

def move_layer(app, id, d):
    ch = app._get_channel_by_any_id(id)
    if ch:
        idx = next((i for i, l in enumerate(ch.kernel_layers) if l['id'] == id), None)
        if idx is not None:
            n_idx = idx + d
            if 0 <= n_idx < len(ch.kernel_layers):
                ch.kernel_layers.insert(n_idx, ch.kernel_layers.pop(idx))
                app._rebuild_channels_ui()

def reset_seed(app):
    app.game_board = app._initialize_board_circle_seed()
    app._initialize_all_local_param_maps()
    app.update_canvas()

def randomize_board(app):
    app.game_board = app._randomize_board()
    app._initialize_all_local_param_maps()
    app.update_canvas()

def clear_board(app):
    app.game_board = app._clear_board()
    app._initialize_all_local_param_maps()
    app.update_canvas()

def toggle_pause(app):
    app.paused = not app.paused

def toggle_zoom_view(app):
    app.view_is_zoomed = not app.view_is_zoomed
    app.zoom_button.config(text="Switch to Global View" if app.view_is_zoomed else "Switch to Zoom View")

def on_view_mode_selected(app, event=None):
    selected = app.view_mode_var.get()
    if selected == "Final Board":
        app.sim_state.view_mode = "Final Board"
    elif selected.startswith("Ch "):
        try:
            ch_part, mode = selected.split(': ')
            ch_idx = int(ch_part.split(' ')[1]) - 1
            app.sim_state.active_channel_idx_for_view = ch_idx
            app.sim_state.view_mode = mode
        except (ValueError, IndexError):
            app.sim_state.view_mode = "Final Board"

def toggle_split_screen(app, initial=False):
    if not initial:
        app.is_split_screen.set(not app.is_split_screen.get())
    if app.is_split_screen.get():
        app.split_screen_button.config(text="Disable Split Screen")
        app.single_view_frame.pack_forget()
        app.split_view_frame.pack(fill="x")
    else:
        app.split_screen_button.config(text="Enable Split Screen")
        app.split_view_frame.pack_forget()
        app.single_view_frame.pack(fill="x")

def toggle_local_params(app, channel_id, enabled, widgets):
    ch = app._get_channel_by_id(channel_id)
    if not ch: return
    ch.has_local_params = enabled
    if enabled and channel_id not in local_param_maps:
        app._initialize_local_param_maps(ch)
    elif not enabled and channel_id in local_param_maps:
        ch.flow_responsive_params = False
        del local_param_maps[channel_id]
        app._rebuild_channels_ui()
    app._set_channel_ui_state(ch, widgets)
    app._update_local_param_draw_ui()
    app._update_vis_options()

def toggle_flow_responsive_params(app, channel_id, enabled, widgets):
    ch = app._get_channel_by_id(channel_id)
    if not ch: return
    ch.flow_responsive_params = enabled
    app._set_channel_ui_state(ch, widgets)
    app._update_local_param_draw_ui()

def toggle_channel_active(app, channel_id, var):
    channel = app._get_channel_by_id(channel_id)
    if channel:
        channel.is_active = var.get()
        app._rebuild_channels_ui()
        app._rebuild_interactions_ui()

def toggle_layer_active(app, layer_id, var):
    layer = app._get_layer_by_id(layer_id)
    if layer:
        layer['is_active'] = var.get()
        app._rebuild_channels_ui()

def on_param_draw_target_selected(app, event=None):
    target = app.param_draw_target.get()
    if target == "Mass":
        app.param_draw_val_slider.config(state='disabled')
        app.param_draw_val_label.config(text="N/A")
        return
    app.param_draw_val_slider.config(state='normal')
    ranges = {'mu': (0.0, 1.0), 'sigma': (0.001, 0.2), 'dt': (0.01, 0.2), 'flow_strength': (0.0, 20.0)}
    from_, to_ = ranges.get(target, (0, 1))
    app.param_draw_val_slider.config(from_=from_, to=to_)
    if (0 <= app.draw_channel_index < len(app.sim_state.channels)):
        ch = app.sim_state.channels[app.draw_channel_index]
        app.param_draw_value.set(getattr(ch, target))
        update_param_draw_slider(app)

def update_param_draw_slider(app, event=None):
    val = app.param_draw_value.get()
    app.param_draw_val_label.config(text=f"Val: {val:.2f}")