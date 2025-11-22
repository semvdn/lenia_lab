import numpy as np
import torch
from skimage import measure
from scipy import ndimage
from scipy.ndimage import binary_fill_holes, binary_dilation
from toroidal_helpers import toroidal_segment, fast_toroidal_weighted_centroid, toroidal_distance, toroidal_delta
from config import GRID_DIM

def perform_organism_tracking(app):
    """
    Performs organism tracking on the simulation grid.

    This function takes the application instance, processes the simulation board to identify organisms,
    and tracks them across frames. It handles segmentation, centroid calculation, matching organisms
    over time, and detecting division events.

    Args:
        app: The main application instance containing the simulation state and tracking data.
    """
    # If there are no channels, reset tracking data and exit
    if len(app.sim_state.channels) == 0:
        app.persistent_tracked_organisms = {}
        app.current_labeled_map = None
        app._base_contour_cache.clear()
        app._touches_seam_cache.clear()
        return

    # Sum the channels to get a magnitude map and create a binary map based on a threshold
    mag_map = torch.sum(app.game_board, dim=0).detach().cpu().numpy()
    binary_map = mag_map > 0.1
    
    # Segment the binary map to label individual organisms
    seg_mode = "watershed" if app.use_watershed.get() else "label"
    labeled_map = toroidal_segment(binary_map, mode=seg_mode, peak_distance=app.watershed_peak_distance.get(), min_size=app.min_organism_mass.get())
    app.current_labeled_map = labeled_map
    labels_present = np.unique(labeled_map)
    labels_present = labels_present[labels_present != 0]

    # If no organisms are found, reset tracking data
    if labels_present.size == 0:
        app.persistent_tracked_organisms = {}
        app.selected_organism_id = None
        app.organism_count_var.set("Count: 0")
        return

    # Calculate properties for each labeled organism
    all_bboxes = ndimage.find_objects(labeled_map)
    total_masses = ndimage.sum_labels(mag_map, labels=labeled_map, index=labels_present)
    
    H, W = GRID_DIM
    label_info = {}
    current_centroids = []

    for i, lab in enumerate(labels_present):
        bbox_slice = all_bboxes[lab - 1]
        min_r, max_r = bbox_slice[0].start, bbox_slice[0].stop
        min_c, max_c = bbox_slice[1].start, bbox_slice[1].stop
        bbox_labels = labeled_map[bbox_slice]
        mask = bbox_labels == lab
        local_rr, local_cc = np.nonzero(mask)
        rr = local_rr + min_r
        cc = local_cc + min_c
        wts = mag_map[rr, cc]
        cent = fast_toroidal_weighted_centroid(rr, cc, wts, H, W)
        current_centroids.append(cent)
        label_info[lab] = {'centroid': cent, 'mask_indices': (rr, cc), 'mass_total': float(total_masses[i]), 'bbox': (min_r, min_c, max_r, max_c)}
        
        # Cache whether the organism touches the grid seam and its base contour
        touches = (rr == 0).any() or (rr == H - 1).any() or (cc == 0).any() or (cc == W - 1).any()
        app._touches_seam_cache[lab] = bool(touches)
        if lab not in app._base_contour_cache:
            roi = np.zeros((max_r - min_r, max_c - min_c), dtype=bool)
            roi[local_rr, local_cc] = True
            filled = binary_fill_holes(roi)
            base = measure.find_contours(np.pad(filled, 1), 0.5)
            mapped = [np.column_stack((c[:, 0] + min_r - 1, c[:, 1] + min_c - 1)) for c in base]
            app._base_contour_cache[lab] = mapped

    # Match current organisms with previously tracked ones
    old_centroids_map = {pid: data['centroid'] for pid, data in app.persistent_tracked_organisms.items()}
    new_org_data, matched_new_labels, disappeared_ids = {}, set(), set(old_centroids_map.keys())

    if old_centroids_map and current_centroids:
        old_ids = list(old_centroids_map.keys())
        dist_matrix = np.array([[toroidal_distance(old, new, H, W) for new in current_centroids] for old in old_centroids_map.values()])
        labels_array = np.array(list(labels_present), dtype=np.int32)
        for i, old_id in enumerate(old_ids):
            if dist_matrix.shape[1] > 0:
                min_dist_idx = np.argmin(dist_matrix[i, :])
                if dist_matrix[i, min_dist_idx] < 25:
                    lab = labels_array[min_dist_idx]
                    if lab not in matched_new_labels:
                        new_org_data[old_id] = {**app.persistent_tracked_organisms[old_id], **label_info[lab], 'label_id': lab}
                        matched_new_labels.add(lab)
                        disappeared_ids.discard(old_id)
                        dist_matrix[:, min_dist_idx] = np.inf

    # Handle organism divisions
    unmatched_labels = [lab for lab in labels_present if lab not in matched_new_labels]
    app.division_events = []
    for old_id in list(disappeared_ids):
        if old_id not in app.persistent_tracked_organisms: continue
        old_cent = app.persistent_tracked_organisms[old_id]['centroid']
        nearby = [lab for lab in unmatched_labels if toroidal_distance(old_cent, label_info[lab]['centroid'], H, W) < 40]
        if len(nearby) >= 2:
            div_event = {'parent_id': old_id, 'parent_mass': app.persistent_tracked_organisms[old_id]['mass_total'], 'children': {}}
            for lab in nearby:
                new_id = app.next_persistent_id; app.next_persistent_id += 1
                new_org_data[new_id] = {'id': new_id, 'parent_id': old_id, 'label_id': lab, **label_info[lab]}
                unmatched_labels.remove(lab)
                div_event['children'][new_id] = label_info[lab]['mass_total']
            app.division_events.append(div_event)

    # Assign new IDs and colors to new organisms
    used_colors = {data.get('color') for data in new_org_data.values() if data.get('color')}
    for lab in unmatched_labels:
        new_id = app.next_persistent_id; app.next_persistent_id += 1
        new_color = next((c for c in app.PALETTE if c not in used_colors), app.PALETTE[new_id % len(app.PALETTE)])
        new_org_data[new_id] = {'id': new_id, 'color': new_color, 'label_id': lab, **label_info[lab]}
        used_colors.add(new_color)

    # Update velocity and direction for each organism
    for pid, data in new_org_data.items():
        cur_cent = np.array(data['centroid'], dtype=np.float64)
        vel = np.array([0., 0.])
        if pid in app.persistent_tracked_organisms:
            prev_cent = np.array(app.persistent_tracked_organisms[pid]['centroid'], dtype=np.float64)
            vel = np.array([toroidal_delta(prev_cent[0], cur_cent[0], H), toroidal_delta(prev_cent[1], cur_cent[1], W)])
            v_alpha = 1.0 - app.velocity_smoothing_factor.get()
            prev_vel = np.array(app.persistent_tracked_organisms[pid].get('smooth_vel', vel))
            data['smooth_vel'] = v_alpha * vel + (1.0 - v_alpha) * prev_vel
            d_alpha = 1.0 - app.direction_smoothing_factor.get()
            norm_vel = vel / (np.linalg.norm(vel) + 1e-6)
            prev_dir = np.array(app.persistent_tracked_organisms[pid].get('smooth_direction', norm_vel))
            new_dir = d_alpha * norm_vel + (1.0 - d_alpha) * prev_dir
            data['smooth_direction'] = new_dir / (np.linalg.norm(new_dir) + 1e-6)
        else:
            data['smooth_vel'] = vel
            data['smooth_direction'] = vel / (np.linalg.norm(vel) + 1e-6)

    # Update the application state with the new tracking data
    app.persistent_tracked_organisms = new_org_data
    if app.selected_organism_id not in app.persistent_tracked_organisms:
        app.selected_organism_id = None
    app.organism_count_var.set(f"Count: {len(app.persistent_tracked_organisms)}")

def on_canvas_select(app, event):
    """
    Handles organism selection from a canvas click event.

    This function determines which organism was clicked based on the event coordinates,
    updates the selected organism ID, and triggers a display update.

    Args:
        app: The main application instance.
        event: The tkinter event object containing click coordinates.
    """
    if not app.tracking_enabled.get() or not app.persistent_tracked_organisms or app.current_labeled_map is None: return
    cw, ch = app.canvas.winfo_width(), app.canvas.winfo_height()
    if cw <= 1 or ch <= 1: return
    
    # Convert canvas coordinates to grid coordinates
    if app.is_split_screen.get():
        quad_w, quad_h = cw / 2, ch / 2
        rel_x, rel_y = event.x % quad_w, event.y % quad_h
        cx, cy = int((rel_x / quad_w) * GRID_DIM[1]), int((rel_y / quad_h) * GRID_DIM[0])
    elif app.view_is_zoomed and app.selected_organism_id in app.persistent_tracked_organisms:
        org_data = app.persistent_tracked_organisms[app.selected_organism_id]
        min_r, min_c, max_r, max_c = org_data.get('bbox', (0, 0, GRID_DIM[0], GRID_DIM[1]))
        pad = max((max_r - min_r), (max_c - min_c))
        zx0, zy0 = max(0, min_c - pad), max(0, min_r - pad)
        zx1, zy1 = min(GRID_DIM[1], max_c + pad), min(GRID_DIM[0], max_r + pad)
        cx, cy = int(zx0 + (event.x / cw) * (zx1 - zx0)), int(zy0 + (event.y / ch) * (zy1 - zy0))
    else:
        cx, cy = int((event.x / cw) * GRID_DIM[1]), int((event.y / ch) * GRID_DIM[0])

    # Identify the organism at the clicked coordinates and update selection
    lab = int(app.current_labeled_map[cy, cx]) if (0 <= cy < GRID_DIM[0] and 0 <= cx < GRID_DIM[1]) else 0
    new_selected_id = next((pid for pid, data in app.persistent_tracked_organisms.items() if data.get('label_id') == lab), None) if lab != 0 else None
    app.selected_organism_id = None if app.selected_organism_id == new_selected_id else new_selected_id
    update_analysis_display(app)

def update_analysis_display(app):
    """
    Updates the analysis display with information about the selected organism.

    Args:
        app: The main application instance.
    """
    if app.selected_organism_id and app.selected_organism_id in app.persistent_tracked_organisms:
        data = app.persistent_tracked_organisms[app.selected_organism_id]
        masses = masses_for_label(app, data.get('label_id'))
        app.stats_mass_label.config(text=f"Mass (per ch): {', '.join(f'{m:.2f}' for m in masses)}")
        vy, vx = data['smooth_vel']
        vel = np.sqrt(vx**2 + vy**2)
        direction = np.degrees(np.arctan2(-vy, vx))
        app.stats_vel_label.config(text=f"Velocity: {vel:.2f} px/frame")
        app.stats_dir_label.config(text=f"Direction: {direction:.1f} deg")
    else:
        # Clear the display if no organism is selected
        app.stats_mass_label.config(text="Mass (per ch): N/A"); app.stats_vel_label.config(text="Velocity: N/A"); app.stats_dir_label.config(text="Direction: N/A")

def masses_for_label(app, label_id):
    """
    Calculates the mass per channel for a given organism label.

    Args:
        app: The main application instance.
        label_id: The label of the organism.

    Returns:
        A list of floats representing the mass for each channel.
    """
    if app.current_labeled_map is None or label_id is None or label_id == 0:
        return [0.0] * len(app.sim_state.channels)
    rr, cc = np.nonzero(app.current_labeled_map == int(label_id))
    g = app.game_board.detach().cpu().numpy()
    return [float(np.sum(g[i, rr, cc])) for i in range(len(app.sim_state.channels))]

def _margin_contours(app, data, margin):
    """
    Calculates the contours of an organism with an added margin.

    This is a helper function to visualize organism boundaries, selections, or other effects
    by dilating the organism's mask before finding contours.

    Args:
        app: The main application instance.
        data: The data dictionary for the organism.
        margin: The margin to add around the organism's mask.

    Returns:
        A list of contours, where each contour is a numpy array of [row, col] coordinates.
    """
    lab = data.get('label_id')
    if lab is None or app.current_labeled_map is None: return []
    if margin < 0.5 and lab in app._base_contour_cache:
        return app._base_contour_cache[lab]

    rr, cc = data['mask_indices']
    H, W = GRID_DIM
    touches = app._touches_seam_cache.get(lab, False)
    r = int(np.ceil(margin))
    se = app._get_selem(r)

    # Optimized path for organisms not touching the toroidal seam
    if not touches:
        min_r, min_c, max_r, max_c = data['bbox']
        pad = r + 2
        y0p, x0p = max(0, min_r - pad), max(0, min_c - pad)
        y1p, x1p = min(H, max_r + pad), min(W, max_c + pad)
        roi = np.zeros((y1p - y0p, x1p - x0p), dtype=bool)
        roi[rr - y0p, cc - x0p] = True
        if r > 0: roi = binary_dilation(roi, structure=se)
        cnts = measure.find_contours(np.pad(roi, 1), 0.5)
        return [np.column_stack((c[:, 0] + y0p - 1, c[:, 1] + x0p - 1)) for c in cnts]
    else:
        # Path for organisms touching the seam, requiring component analysis
        temp_mask = np.zeros(GRID_DIM, dtype=bool)
        temp_mask[rr, cc] = True
        labeled_components, num_features = measure.label(temp_mask, connectivity=1, return_num=True)
        all_contours = []
        for i in range(1, num_features + 1):
            comp_rr, comp_cc = np.nonzero(labeled_components == i)
            pad = r + 2
            min_r_c, max_r_c = comp_rr.min(), comp_rr.max()
            min_c_c, max_c_c = comp_cc.min(), comp_cc.max()
            y0p, x0p = max(0, min_r_c - pad), max(0, min_c_c - pad)
            y1p, x1p = min(H, max_r_c + pad + 1), min(W, max_c_c + pad + 1)
            roi = np.zeros((y1p - y0p, x1p - x0p), dtype=bool)
            roi[comp_rr - y0p, comp_cc - x0p] = True
            if r > 0: roi = binary_dilation(roi, structure=se)
            cnts = measure.find_contours(np.pad(roi, 1), 0.5)
            all_contours.extend([np.column_stack((c[:, 0] + y0p - 1, c[:, 1] + x0p - 1)) for c in cnts])
        return all_contours
