import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageTk, ImageDraw
from tkinter import TclError
from simulation import generate_composite_kernel, local_param_maps, LOCAL_PARAM_NAMES
from config import GRID_DIM, device

def update_canvas(app):
    """Renders the current simulation view onto the Tk canvas, handling zoom/split options."""
    if len(app.sim_state.channels) == 0:
        app.canvas.delete("all")
        return

    zoom_box = None
    if app.view_is_zoomed and app.selected_organism_id in app.persistent_tracked_organisms:
        org_data = app.persistent_tracked_organisms[app.selected_organism_id]
        bbox = org_data.get('bbox', (0, 0, GRID_DIM[0], GRID_DIM[1]))
        min_r, min_c, max_r, max_c = bbox
        pad = max((max_r - min_r), (max_c - min_c))
        zoom_box = (max(0, min_c - pad), max(0, min_r - pad), min(GRID_DIM[1], max_c + pad), min(GRID_DIM[0], max_r + pad))

    if app.is_split_screen.get():
        final_img = Image.new('RGB', (GRID_DIM[1] * 2, GRID_DIM[0] * 2))
        q_size = (GRID_DIM[1], GRID_DIM[0])
        positions = [(0, 0), (q_size[0], 0), (0, q_size[1]), (q_size[0], q_size[1])]

        for i in range(4):
            view_name = app.split_view_vars[i].get()
            img_array = _get_view_array_by_name(app, view_name)
            if img_array is not None:
                quadrant_img = Image.fromarray(img_array)
                offset_x, offset_y = 0, 0
                
                if zoom_box:
                    offset_x, offset_y = zoom_box[0], zoom_box[1]
                    quadrant_img = quadrant_img.crop(zoom_box)
                
                scale_x = quadrant_img.width / (zoom_box[2] - zoom_box[0]) if zoom_box else quadrant_img.width / GRID_DIM[1]
                scale_y = quadrant_img.height / (zoom_box[3] - zoom_box[1]) if zoom_box else quadrant_img.height / GRID_DIM[0]
                
                if app.tracking_enabled.get():
                    _draw_tracking_overlay(app, quadrant_img, scale_x, scale_y, offset_x, offset_y)
                    
                final_img.paste(quadrant_img.resize(q_size, Image.NEAREST), positions[i])
        img = final_img
    else:
        view_name = app.view_mode_var.get()
        img_array = _get_view_array_by_name(app, view_name)
        img = Image.fromarray(img_array) if img_array is not None else Image.new('RGB', (GRID_DIM[1], GRID_DIM[0]))

        offset_x, offset_y = 0, 0
        if zoom_box:
            offset_x, offset_y = zoom_box[0], zoom_box[1]
            img = img.crop(zoom_box)
        
        scale_x = img.width / (zoom_box[2] - zoom_box[0]) if zoom_box else img.width / GRID_DIM[1]
        scale_y = img.height / (zoom_box[3] - zoom_box[1]) if zoom_box else img.height / GRID_DIM[0]

        if app.tracking_enabled.get():
            _draw_tracking_overlay(app, img, scale_x, scale_y, offset_x, offset_y)
    
    app.last_drawn_array = np.array(img)
    
    if app.canvas.winfo_width() > 1:
        img = img.resize((app.canvas.winfo_width(), app.canvas.winfo_height()), Image.NEAREST)
    
    app.photo = ImageTk.PhotoImage(image=img)
    if app.canvas_image_id is None:
        app.canvas_image_id = app.canvas.create_image(0, 0, image=app.photo, anchor="nw")
    else:
        app.canvas.itemconfig(app.canvas_image_id, image=app.photo)

def _get_view_array_by_name(app, view_name):
    """Returns an RGB numpy array for the requested view selection."""
    channels = app.sim_state.channels
    if view_name == "Final Board":
        return _get_multichannel_array(app.game_board, channels)
    
    if view_name.startswith("Ch "):
        try:
            parts = view_name.split(': ')
            ch_part, mode = parts[0], parts[1]
            ch_idx = int(ch_part.split(' ')[1]) - 1

            if not (0 <= ch_idx < len(channels)): return None
            
            if app.sim_fields and mode in ["Potential Field", "Growth Field", "Flow Field"]:
                if mode == "Potential Field": return _get_single_channel_array(app.sim_fields['potential'][ch_idx], 'viridis')
                if mode == "Growth Field": return _get_single_channel_array(app.sim_fields['growth'][ch_idx], 'coolwarm')
                if mode == "Flow Field": return _get_flow_field_array(app.sim_fields['flow_x'][ch_idx], app.sim_fields['flow_y'][ch_idx])
            
            ch_id = channels[ch_idx].id
            if ch_id in local_param_maps and mode in local_param_maps[ch_id]:
                map_tensor = local_param_maps[ch_id][mode]
                return _get_single_channel_array(map_tensor, 'plasma')
        except (ValueError, IndexError):
            return _get_multichannel_array(app.game_board, channels)
    
    return _get_multichannel_array(app.game_board, channels)

def _get_multichannel_array(board_tensor, channels):
    """Combines active channel slices into an RGB array using their configured colors."""
    active_channels = [ch for ch in channels if ch.is_active]
    if not active_channels: return np.zeros((*GRID_DIM, 3), dtype=np.uint8)
    active_indices = [i for i, ch in enumerate(channels) if ch.is_active]
    active_board = board_tensor[active_indices, :, :]
    try:
        colors = [Image.new('RGB', (1, 1), color=c.color_hex).getpixel((0, 0)) for c in active_channels]
        c_tensor = torch.tensor(colors, device=device, dtype=torch.float32) / 255.0
    except (ValueError, TclError):
        return None
    img_tensor = torch.clamp(torch.matmul(active_board.permute(1, 2, 0), c_tensor), 0, 1)
    return (img_tensor * 255).byte().cpu().numpy()

def _get_single_channel_array(tensor, colormap='magma'):
    """Normalizes a tensor slice and maps it through a matplotlib colormap."""
    t_min, t_max = tensor.min(), tensor.max()
    if t_max > t_min: tensor = (tensor - t_min) / (t_max - t_min)
    return (plt.get_cmap(colormap)(tensor.detach().cpu().numpy())[:, :, :3] * 255).astype(np.uint8)

def _get_flow_field_array(flow_x, flow_y):
    """Encodes flow magnitude and direction into an HSV-based RGB image."""
    mag = torch.sqrt(flow_x**2 + flow_y**2)
    ang = torch.atan2(flow_y, flow_x)
    h = (ang + np.pi) / (2 * np.pi)
    s = torch.ones_like(mag)
    v = mag / (mag.max() + 1e-6)
    hsv = torch.stack((h, s, v), dim=2).cpu().numpy()
    rgb = plt.cm.hsv(hsv[:, :, 0])
    rgb[:, :, :3] *= hsv[:, :, 2, None]
    return (rgb[:, :, :3] * 255).astype(np.uint8)

def update_kernel_preview(app, id):
    """Refreshes the cached kernel preview widget for the specified channel id."""
    if not (ch := app._get_channel_by_id(id)) or id not in app.kernel_previews: return
    k, _, _ = app.kernel_cache[id]
    max_abs = torch.max(torch.abs(k)) or 1.0
    norm_k = (k + max_abs) / (2 * max_abs)
    rgb = (plt.get_cmap('coolwarm')(norm_k.detach().cpu().numpy())[:, :, :3] * 255).astype(np.uint8)
    img = Image.fromarray(rgb).resize((80, 80), Image.NEAREST)
    photo = ImageTk.PhotoImage(image=img)
    app.kernel_previews[id].configure(image=photo)
    app.kernel_previews[id].image = photo

def draw_on_canvas(app, event, right_click=False):
    """Handles user drawing on the canvas to add/remove mass or edit local parameter maps."""
    num_channels = len(app.sim_state.channels)
    if num_channels == 0:
        return
    
    cw, ch = app.canvas.winfo_width(), app.canvas.winfo_height()
    if cw <= 1 or ch <= 1: return

    if app.is_split_screen.get():
        quad_w, quad_h = cw / 2, ch / 2
        rel_x, rel_y = event.x % quad_w, event.y % quad_h
        gx = int((rel_x / quad_w) * GRID_DIM[1])
        gy = int((rel_y / quad_h) * GRID_DIM[0])
    else:
        gx = int((event.x / cw) * GRID_DIM[1])
        gy = int((event.y / ch) * GRID_DIM[0])
    
    r = app.draw_brush_size
    ys, ye = max(0, gy - r), min(GRID_DIM[0], gy + r)
    xs, xe = max(0, gx - r), min(GRID_DIM[1], gx + r)
    if ys >= ye or xs >= xe: return
    
    yy, xx = torch.meshgrid(torch.arange(ys, ye, device=device), torch.arange(xs, xe, device=device), indexing='ij')
    mask = torch.sqrt((yy - gy)**2 + (xx - gx)**2) <= r
    
    target = app.param_draw_target.get()
    targets = range(num_channels) if app.draw_channel_index == -1 else [app.draw_channel_index]
    if target == "Mass":
        val = 0.0 if right_click else 1.0
        for idx in targets:
            app.game_board[idx, ys:ye, xs:xe][mask] = val
    else:
        for idx in targets:
            ch = app.sim_state.channels[idx]
            if ch.has_local_params and ch.id in local_param_maps and target in local_param_maps[ch.id]:
                val = app.param_draw_value.get()
                local_param_maps[ch.id][target][ys:ye, xs:xe][mask] = val

    update_canvas(app)

def _draw_tracking_overlay(app, pil_image, scale_x, scale_y, offset_x=0, offset_y=0):
    """Draws tracked organism outlines, centers, and velocity indicators on the PIL image."""
    draw = ImageDraw.Draw(pil_image)
    margin = float(app.outline_margin.get())

    for pid, data in app.persistent_tracked_organisms.items():
        if app.show_outlines.get():
            contours = app._margin_contours(data, margin)
            color = data.get('color', "#FFFFFF") if pid != app.selected_organism_id else "yellow"
            for cont in contours:
                xs = (cont[:, 1] - offset_x) * scale_x
                ys = (cont[:, 0] - offset_y) * scale_y
                pts = np.column_stack((xs, ys)).ravel().tolist()
                if len(pts) >= 4:
                    draw.line(pts, fill=color, width=1)

        cy, cx = data['centroid']
        sx, sy = (cx - offset_x) * scale_x, (cy - offset_y) * scale_y
        
        if app.show_com.get():
            draw.ellipse((sx - 1, sy - 1, sx + 1, sy + 1), fill="#888888")

            # --- FIX: Draw "ghost" CoM markers for wrapped organisms ---
            lab = data.get('label_id')
            if lab and app._touches_seam_cache.get(lab, False):
                # The full visual width/height of the grid in scaled pixels
                grid_w_scaled = GRID_DIM[1] * scale_x
                grid_h_scaled = GRID_DIM[0] * scale_y
                _, mask_cc = data['mask_indices']
                mask_rr, _ = data['mask_indices']

                # Check if organism is split horizontally (left/right edges)
                if np.any(mask_cc == 0) and np.any(mask_cc == GRID_DIM[1] - 1):
                    if cx > GRID_DIM[1] / 2: # CoM is near the right edge
                        draw.ellipse((sx - grid_w_scaled - 1, sy - 1, sx - grid_w_scaled + 1, sy + 1), fill="#888888")
                    else: # CoM is near the left edge
                        draw.ellipse((sx + grid_w_scaled - 1, sy - 1, sx + grid_w_scaled + 1, sy + 1), fill="#888888")
                
                # Check if organism is split vertically (top/bottom edges)
                if np.any(mask_rr == 0) and np.any(mask_rr == GRID_DIM[0] - 1):
                    if cy > GRID_DIM[0] / 2: # CoM is near the bottom edge
                        draw.ellipse((sx - 1, sy - grid_h_scaled - 1, sx + 1, sy - grid_h_scaled + 1), fill="#888888")
                    else: # CoM is near the top edge
                        draw.ellipse((sx - 1, sy + grid_h_scaled - 1, sx + 1, sy + grid_h_scaled + 1), fill="#888888")

        if app.show_direction.get():
            vy, vx = data['smooth_vel']
            vel = np.sqrt(vx**2 + vy**2)
            direction_vec = data['smooth_direction']
            if vel > 0.01:
                vy_dir, vx_dir = direction_vec
                length = max(3, min(40, vel * app.velocity_sensitivity.get()))
                tail_len = 3
                draw.line(((sx - vx_dir * tail_len, sy - vy_dir * tail_len), (sx + vx_dir * length, sy + vy_dir * length)), fill="#888888", width=1)
