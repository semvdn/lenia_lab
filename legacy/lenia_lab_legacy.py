import tkinter as tk
from tkinter import ttk, filedialog, simpledialog, messagebox
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageTk, ImageDraw
import json
import imageio
import uuid
import copy
import os
import datetime
import pandas as pd
from skimage import measure
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.morphology import remove_small_objects
from scipy.ndimage import distance_transform_edt, binary_fill_holes, binary_dilation

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

# ------------------ Toroidal helpers ------------------

class _DSU:
    def __init__(self, n):
        self.parent = np.arange(n + 1, dtype=np.int32)
        self.rank = np.zeros(n + 1, dtype=np.int8)
    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb: return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1

def _remap_sequential(labels):
    if labels.size == 0: return labels
    uniq = np.unique(labels); uniq = uniq[uniq != 0]
    if uniq.size == 0: return labels
    mapping = {old: i + 1 for i, old in enumerate(uniq)}
    out = np.zeros_like(labels, dtype=np.int32)
    for old, new in mapping.items():
        out[labels == old] = new
    return out

def toroidal_delta(old, new, size):
    d = new - old
    if d > size / 2: d -= size
    elif d < -size / 2: d += size
    return d

def toroidal_distance(a, b, H, W):
    dr = toroidal_delta(a[0], b[0], H)
    dc = toroidal_delta(a[1], b[1], W)
    return np.hypot(dr, dc)

def circular_mean(vals, weights, size):
    if vals.size == 0: return 0.0
    angles = 2.0 * np.pi * (vals.astype(np.float64) / float(size))
    w = weights.astype(np.float64)
    W = np.sum(w)
    if W <= 0: return float(np.mean(vals) % size)
    x = np.sum(w * np.cos(angles)) / W
    y = np.sum(w * np.sin(angles)) / W
    ang = np.arctan2(y, x)
    if ang < 0: ang += 2.0 * np.pi
    return (ang / (2.0 * np.pi)) * size

def toroidal_weighted_centroid(rows, cols, weights, H, W):
    r_c = circular_mean(rows, weights, H)
    c_c = circular_mean(cols, weights, W)
    return np.array([r_c, c_c], dtype=np.float64)

def _union_wrap_seams(labels, min_size=1):
    H, W = labels.shape
    n = int(labels.max())
    if n == 0:
        return labels.astype(np.int32)

    dsu = _DSU(n)

    for r in range(H):
        a = int(labels[r, 0])
        if a > 0:
            for dr in (-1, 0, 1):
                rr = (r + dr) % H
                b = int(labels[rr, W - 1])
                if b > 0:
                    dsu.union(a, b)

    for c in range(W):
        a = int(labels[0, c])
        if a > 0:
            for dc in (-1, 0, 1):
                cc = (c + dc) % W
                b = int(labels[H - 1, cc])
                if b > 0:
                    dsu.union(a, b)

    map_root = np.arange(n + 1, dtype=np.int32)
    for lab in range(1, n + 1):
        map_root[lab] = dsu.find(lab)
    map_root[0] = 0
    roots_img = map_root[labels]

    if int(min_size) > 1:
        counts = np.bincount(roots_img.ravel(), minlength=map_root.max() + 1)
        keep_root = counts >= int(min_size)
        new_id = np.zeros_like(keep_root, dtype=np.int32)
        cid = 1
        for rid, keep in enumerate(keep_root):
            if rid == 0: continue
            if keep:
                new_id[rid] = cid
                cid += 1
        out = new_id[roots_img]
    else:
        out = roots_img

    return out.astype(np.int32)

def toroidal_segment(binary_map, mode="label", peak_distance=7, min_size=1):
    H, W = binary_map.shape
    if not np.any(binary_map):
        return np.zeros((H, W), dtype=np.int32)

    if mode == "watershed":
        big_mask = np.tile(binary_map.astype(bool), (3, 3))
        distance = distance_transform_edt(big_mask)
        coords = peak_local_max(distance, min_distance=int(peak_distance), labels=big_mask)
        peak_mask = np.zeros_like(distance, dtype=bool)
        if coords.size > 0:
            peak_mask[tuple(coords.T)] = True
        markers, _ = measure.label(peak_mask, return_num=True, connectivity=2)
        labels_big = watershed(-distance, markers, mask=big_mask)
        labels_center = labels_big[H:2*H, W:2*W].astype(np.int32)
        labels_final = _union_wrap_seams(labels_center, min_size=min_size)
        return _remap_sequential(labels_final)

    labels_center = measure.label(binary_map.astype(bool), connectivity=2).astype(np.int32)
    if labels_center.max() == 0:
        return labels_center
    labels_final = _union_wrap_seams(labels_center, min_size=min_size)
    return _remap_sequential(labels_final)

# --- DATA STRUCTURES & SIMULATION CORE ---
class Channel:
    def __init__(self, color_hex='#FF00FF'):
        self.id = str(uuid.uuid4())
        self.color_hex = color_hex
        self.mu = 0.14
        self.sigma = 0.04
        self.dt = 0.1
        self.flow_strength = 5.0
        self.use_flow = True
        self.has_local_params = False
        self.flow_responsive_params = False
        self.flow_sensitivity = 10.0
        self.mu_range = [0.1, 0.2]
        self.sigma_range = [0.03, 0.06]
        self.dt_range = [0.05, 0.15]
        self.flow_strength_range = [2.0, 8.0]
        self.growth_func_name = "Gaussian Bell"
        self.kernel_layers = [{'id': str(uuid.uuid4()), 'type': 'Gaussian Ring', 'radius': 13, 'weight': 1.0, 'op': '+', 'is_active': True}]
        self.is_active = True
        
        # --- NEW SENSORIMOTOR ATTRIBUTES ---
        self.channel_type = 'lenia'  # 'lenia' or 'environment'
        self.is_sensorimotor = False
        self.sensing_target_channel_idx = 0
        self.sensorimotor_sensitivity = 1.0
        self.sensing_kernel_layers = [{'id': str(uuid.uuid4()), 'type': 'Filled Gaussian', 'radius': 7, 'weight': 1.0, 'op': '+', 'is_active': True}]

class SimulationState:
    def __init__(self):
        self.channels = [Channel(color_hex='#00FFFF')]; self.interaction_matrix = [[1.0]]
        self.view_mode = "Final Board"; self.active_channel_idx_for_view = 0
params = SimulationState()

local_param_maps = {}
LOCAL_PARAM_NAMES = ['mu', 'sigma', 'dt', 'flow_strength']

def kernel_gaussian_ring(r, ks):
    x, y = torch.meshgrid(torch.arange(ks, device=device), torch.arange(ks, device=device), indexing='ij'); c=ks//2
    d = torch.sqrt((x-c)**2+(y-c)**2); s = torch.exp(-((d-r)**2)/(2*(max(1,r)/3)**2)); return s*(d>0)
def kernel_filled_gaussian(r, ks):
    x, y = torch.meshgrid(torch.arange(ks, device=device), torch.arange(ks, device=device), indexing='ij'); c=ks//2
    d = torch.sqrt((x-c)**2+(y-c)**2); return torch.exp(-(d**2)/(2*(max(1,r)/2)**2))
def kernel_square(r, ks):
    x, y = torch.meshgrid(torch.arange(ks, device=device), torch.arange(ks, device=device), indexing='ij'); c=ks//2
    m = (torch.abs(x-c)<r)&(torch.abs(y-c)<r); return m.float()
kernel_functions = {"Gaussian Ring": kernel_gaussian_ring, "Filled Gaussian": kernel_filled_gaussian, "Square": kernel_square}

def growth_gaussian_bell(x, mu, sigma):
    if isinstance(sigma, torch.Tensor):
        clamped_sigma = torch.clamp(sigma, min=1e-6)
    else:
        clamped_sigma = max(sigma, 1e-6)
    return torch.exp(-((x - mu)**2) / (2 * clamped_sigma**2)) * 2 - 1

def growth_step_function(x, mu, sigma): return (((x>mu-sigma)&(x<mu+sigma)).float()*2-1)
growth_functions = { "Gaussian Bell": growth_gaussian_bell, "Step Function": growth_step_function }

def generate_composite_kernel(kernel_layers):
    active_layers = [l for l in kernel_layers if l.get('is_active', True)]
    if not active_layers: return torch.zeros(KERNEL_SIZE, KERNEL_SIZE, device=device)
    k = kernel_functions.get(active_layers[0]['type'])(active_layers[0]['radius'], KERNEL_SIZE)
    if torch.sum(k)>0: k/=torch.sum(k)
    k*=active_layers[0]['weight']
    for layer in active_layers[1:]:
        op=layer.get('op','+'); sub_k=kernel_functions.get(layer['type'])(layer['radius'], KERNEL_SIZE)
        if torch.sum(sub_k)>0: sub_k/=torch.sum(sub_k)
        sub_k*=layer['weight']
        if op=='+':k+=sub_k
        elif op=='-':k-=sub_k
        elif op=='*':k*=sub_k
    return k

def update_multichannel_board(game_board, channels, interaction_matrix):
    num_channels = len(channels); new_board = game_board.clone()
    sim_fields = {'potential':torch.zeros_like(game_board), 'growth':torch.zeros_like(game_board), 'flow_y':torch.zeros_like(game_board), 'flow_x':torch.zeros_like(game_board)}
    padding = KERNEL_SIZE//2
    grid_y, grid_x = torch.meshgrid(torch.linspace(-1,1,steps=GRID_DIM[0],device=device), torch.linspace(-1,1,steps=GRID_DIM[1],device=device), indexing='ij')
    
    active_indices = [i for i, ch in enumerate(channels) if ch.is_active]

    for i in active_indices:
        channel_params = channels[i]

        if channel_params.channel_type == 'environment':
            continue
        
        influenced_slice=torch.zeros((1,1,*GRID_DIM),device=device)
        for j in range(num_channels):
            if channels[j].is_active and (w:=interaction_matrix[i][j])!=0: 
                influenced_slice += game_board[j,:,:].unsqueeze(0).unsqueeze(0) * w
        influenced_slice=torch.clamp(influenced_slice,0,1)
        
        kernel=generate_composite_kernel(channel_params.kernel_layers)
        padded_board=torch.nn.functional.pad(influenced_slice,(padding,)*4,mode='circular')
        potential=torch.nn.functional.conv2d(padded_board, kernel.unsqueeze(0).unsqueeze(0)).squeeze()
        sim_fields['potential'][i]=potential

        original_slice=game_board[i,:,:].unsqueeze(0).unsqueeze(0)
        kernel_grad_y, kernel_grad_x = torch.gradient(kernel)
        padded_original=torch.nn.functional.pad(original_slice,(padding,)*4,mode='circular')
        flow_y=-torch.nn.functional.conv2d(padded_original, kernel_grad_y.unsqueeze(0).unsqueeze(0)).squeeze()
        flow_x=-torch.nn.functional.conv2d(padded_original, kernel_grad_x.unsqueeze(0).unsqueeze(0)).squeeze()
        sim_fields['flow_y'][i],sim_fields['flow_x'][i]=flow_y,flow_x
        
        sensed_field = None
        if channel_params.is_sensorimotor and channel_params.has_local_params:
            target_idx = channel_params.sensing_target_channel_idx
            if 0 <= target_idx < num_channels and channels[target_idx].channel_type == 'environment':
                target_slice = game_board[target_idx,:,:].unsqueeze(0).unsqueeze(0)
                sensing_kernel = generate_composite_kernel(channel_params.sensing_kernel_layers)
                if torch.sum(sensing_kernel) > 0:
                    padded_target = torch.nn.functional.pad(target_slice, (padding,) * 4, mode='circular')
                    sensed_field = torch.nn.functional.conv2d(padded_target, sensing_kernel.unsqueeze(0).unsqueeze(0)).squeeze()

        if channel_params.has_local_params and channel_params.id in local_param_maps:
            response_factor = None
            if channel_params.is_sensorimotor and sensed_field is not None:
                response_factor = torch.clamp(sensed_field / (channel_params.sensorimotor_sensitivity + 1e-6), 0.0, 1.0)
            elif channel_params.flow_responsive_params:
                flow_mag = torch.sqrt(flow_x**2 + flow_y**2)
                response_factor = torch.clamp(flow_mag / (channel_params.flow_sensitivity + 1e-6), 0.0, 1.0)
            
            if response_factor is not None:
                p_maps = local_param_maps[channel_params.id]
                p_maps['mu'] = channel_params.mu_range[0] + response_factor * (channel_params.mu_range[1] - channel_params.mu_range[0])
                p_maps['sigma'] = channel_params.sigma_range[0] + response_factor * (channel_params.sigma_range[1] - channel_params.sigma_range[0])
                p_maps['dt'] = channel_params.dt_range[0] + response_factor * (channel_params.dt_range[1] - channel_params.dt_range[0])
                p_maps['flow_strength'] = channel_params.flow_strength_range[0] + response_factor * (channel_params.flow_strength_range[1] - channel_params.flow_strength_range[0])

        if channel_params.has_local_params and channel_params.id in local_param_maps:
            p_maps = local_param_maps[channel_params.id]
            mu, sigma, dt, flow_strength = p_maps['mu'], p_maps['sigma'], p_maps['dt'], p_maps['flow_strength']
        else:
            mu, sigma, dt, flow_strength = channel_params.mu, channel_params.sigma, channel_params.dt, channel_params.flow_strength
        
        growth=growth_functions.get(channel_params.growth_func_name)(potential, mu, sigma)
        sim_fields['growth'][i]=growth

        if channel_params.use_flow:
            scaled_flow_x=flow_x*(2.0/GRID_DIM[1])*flow_strength
            scaled_flow_y=flow_y*(2.0/GRID_DIM[0])*flow_strength
            sampling_grid=torch.stack((grid_x-scaled_flow_x, grid_y-scaled_flow_y),dim=2)
            sampling_grid = torch.remainder(sampling_grid + 1.0, 2.0) - 1.0

            if channel_params.has_local_params and channel_params.id in local_param_maps:
                maps_to_advect = [game_board[i,:,:]] + [p_maps[name] for name in LOCAL_PARAM_NAMES]
                stacked_maps = torch.stack(maps_to_advect, dim=0).unsqueeze(0)
                advected_stack = torch.nn.functional.grid_sample(stacked_maps, sampling_grid.unsqueeze(0), mode='bicubic', padding_mode='zeros', align_corners=True).squeeze(0)
                advected_board = advected_stack[0,:,:]
                for idx, name in enumerate(LOCAL_PARAM_NAMES):
                    local_param_maps[channel_params.id][name] = advected_stack[idx+1,:,:]
            else:
                advected_board=torch.nn.functional.grid_sample(original_slice, sampling_grid.unsqueeze(0), mode='bicubic', padding_mode='zeros', align_corners=True).squeeze()
            
            new_board[i,:,:]=torch.clamp(advected_board + dt * growth, 0, 1)
        else:
            sim_fields['flow_y'][i].zero_(); sim_fields['flow_x'][i].zero_()
            current_slice = game_board[i,:,:]
            new_board[i,:,:]=torch.clamp(current_slice + dt * growth, 0, 1)
    
    sim_fields['final_board']=new_board
    return sim_fields

# --- TKINTER APPLICATION CLASS ---
class LeniaApp:
    def __init__(self, root):
        self.root = root; self.root.title("Lenia Lab Scientific")
        self._ensure_dir(SETTINGS_FOLDER); self._ensure_dir(GIFS_FOLDER); self._ensure_dir(PRESETS_FOLDER)
        
        self.paused = False; self.game_board = self._initialize_board_circle_seed()
        self.sim_fields = None; self.last_drawn_array = None

        self.tracking_enabled = tk.BooleanVar(value=False)
        self.show_outlines = tk.BooleanVar(value=True); self.show_com = tk.BooleanVar(value=True); self.show_direction = tk.BooleanVar(value=True)
        self.persistent_tracked_organisms = {}; self.next_persistent_id = 0
        self.selected_organism_id = None; self.organism_presets = self._load_presets()
        self.view_is_zoomed = False; self.outline_margin = tk.DoubleVar(value=2.0)
        self.min_organism_mass = tk.IntVar(value=20)
        self.use_watershed = tk.BooleanVar(value=False)
        self.watershed_peak_distance = tk.IntVar(value=7)
        self.division_events = []
        self.velocity_smoothing_factor = tk.DoubleVar(value=0.8)
        self.direction_smoothing_factor = tk.DoubleVar(value=0.7)
        self.velocity_sensitivity = tk.DoubleVar(value=5.0)
        self.organism_count_var = tk.StringVar(value="Count: 0")

        self.is_split_screen = tk.BooleanVar(value=False)
        self.split_view_vars = [tk.StringVar(value="Final Board") for _ in range(4)]
        self.view_mode_var = tk.StringVar(value="Final Board")

        self.current_labeled_map = None

        self._selem_cache = {}
        self._base_contour_cache = {}
        self._touches_seam_cache = {}

        self.is_recording = False; self.gif_frames = []
        self.is_stats_recording = False; self.stats_log = []; self.stats_gif_writers = {}
        
        self.draw_channel_index = 0
        self.draw_brush_size = 10
        self.param_draw_target = tk.StringVar(value="Mass")
        self.param_draw_value = tk.DoubleVar(value=0.5)

        self.PALETTE = ['#FF5733', '#33FF57', '#3357FF', '#FF33A1', '#A133FF', '#33FFA1', '#FFC300', '#C70039', '#900C3F', '#581845']

        self.main_frame = ttk.Frame(self.root); self.main_frame.pack(fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(self.main_frame, width=GRID_SIZE, height=GRID_SIZE, bg='black'); self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.ui_frame = ttk.Frame(self.main_frame, width=UI_PANEL_WIDTH); self.ui_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5); self.ui_frame.pack_propagate(False)
        self._build_ui()

        self.root.bind("<space>", lambda event: self.toggle_pause())
        self.canvas.bind("<B1-Motion>", self.draw_on_canvas); self.canvas.bind("<Button-1>", self.draw_on_canvas)
        self.canvas.bind("<B3-Motion>", lambda e: self.draw_on_canvas(e, right_click=True)); self.canvas.bind("<Button-3>", lambda e: self.draw_on_canvas(e, right_click=True))
        self.canvas.bind("<Shift-Button-1>", self._on_canvas_select)

        self._initialize_all_local_param_maps()
        self.update_loop()

    def _get_selem(self, r):
        r = int(max(0, r))
        if r == 0: return np.ones((1,1), dtype=bool)
        if r in self._selem_cache: return self._selem_cache[r]
        y, x = np.ogrid[-r:r+1, -r:r+1]
        se = (x*x + y*y) <= r*r
        self._selem_cache[r] = se
        return se

    def _ensure_dir(self, dir_path):
        if not os.path.exists(dir_path): os.makedirs(dir_path)

    def _destroy_children(self, widget):
        for child in widget.winfo_children(): child.destroy()
            
    def _build_ui(self):
        self._destroy_children(self.ui_frame)
        notebook = ttk.Notebook(self.ui_frame); notebook.pack(fill=tk.BOTH, expand=True)
        tab1 = ttk.Frame(notebook); tab2 = ttk.Frame(notebook)
        notebook.add(tab1, text="Channels & Kernels"); notebook.add(tab2, text="Analysis & Tracking")
        self._build_channels_tab(tab1); self._build_analysis_tab(tab2)
        self._update_local_param_draw_ui()

    def _build_channels_tab(self, parent):
        top_frame = ttk.Frame(parent); interactions_container = ttk.Frame(parent)
        channel_container = ttk.Frame(parent); bottom_frame = ttk.Frame(parent)
        top_frame.pack(fill=tk.X, pady=5, side=tk.TOP)
        interactions_container.pack(fill=tk.X, pady=5, side=tk.TOP)
        bottom_frame.pack(fill=tk.X, pady=5, side=tk.BOTTOM)
        channel_container.pack(fill=tk.BOTH, expand=True, side=tk.TOP)
        
        ttk.Button(top_frame, text="Add New Channel", command=self.add_channel).pack(fill=tk.X)
        self.interactions_frame = ttk.LabelFrame(interactions_container, text="Channel Interactions (J → I)")
        self.interactions_frame.pack(fill=tk.X); self._rebuild_interactions_ui()

        scroll_canvas = tk.Canvas(channel_container, highlightthickness=0); scrollbar = ttk.Scrollbar(channel_container, orient="vertical", command=scroll_canvas.yview)
        self.scrollable_frame = ttk.Frame(scroll_canvas); self.scrollable_frame.bind("<Configure>", lambda e: scroll_canvas.configure(scrollregion=scroll_canvas.bbox("all")))
        scroll_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw", width=UI_PANEL_WIDTH - 25); scroll_canvas.configure(yscrollcommand=scrollbar.set)
        scroll_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True); scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._rebuild_channels_ui()

        sim_controls = ttk.Frame(bottom_frame); sim_controls.pack(fill=tk.X, pady=5)
        ttk.Button(sim_controls, text="Reset Seed", command=self.reset_seed).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(sim_controls, text="Randomize", command=self.randomize_board).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(sim_controls, text="Clear", command=self.clear_board).pack(side=tk.LEFT, expand=True, fill=tk.X)
        
        draw_controls_frame = ttk.LabelFrame(bottom_frame, text="Drawing Controls")
        draw_controls_frame.pack(fill=tk.X, pady=(5,0), padx=2)
        
        draw_row1 = ttk.Frame(draw_controls_frame)
        draw_row1.pack(fill=tk.X, pady=2)
        ttk.Label(draw_row1, text="Draw Ch:").pack(side=tk.LEFT, padx=5)
        self.draw_channel_var = tk.StringVar(); self.draw_channel_dd = ttk.Combobox(draw_row1, textvariable=self.draw_channel_var, state="readonly", width=3)
        self.draw_channel_dd.bind("<<ComboboxSelected>>", self._on_draw_channel_selected); self.draw_channel_dd.pack(side=tk.LEFT)
        self._update_draw_channel_selector()
        ttk.Label(draw_row1, text="Brush Size:").pack(side=tk.LEFT, padx=(10, 0))
        self.brush_size_var = tk.IntVar(value=self.draw_brush_size)
        ttk.Scale(draw_row1, from_=1, to=50, orient=tk.HORIZONTAL, variable=self.brush_size_var, command=self._update_brush_size).pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.param_draw_frame = ttk.Frame(draw_controls_frame)
        self.param_draw_frame.pack(fill=tk.X, pady=2)
        ttk.Label(self.param_draw_frame, text="Draw Target:").pack(side=tk.LEFT, padx=5)
        self.param_draw_dd = ttk.Combobox(self.param_draw_frame, textvariable=self.param_draw_target, state="readonly", width=12, values=["Mass"] + LOCAL_PARAM_NAMES)
        self.param_draw_dd.bind("<<ComboboxSelected>>", self._on_param_draw_target_selected)
        self.param_draw_dd.pack(side=tk.LEFT)
        self.param_draw_val_label = ttk.Label(self.param_draw_frame, text="Val: 0.50", width=8)
        self.param_draw_val_slider = ttk.Scale(self.param_draw_frame, from_=0, to=1, orient=tk.HORIZONTAL, variable=self.param_draw_value, command=self._update_param_draw_slider)
        self.param_draw_val_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.param_draw_val_label.pack(side=tk.LEFT, padx=(5,0))
        
        self.record_button = ttk.Button(bottom_frame, text="Record Full GIF", command=self.record_gif); self.record_button.pack(fill=tk.X, pady=2)
        file_controls = ttk.Frame(bottom_frame); file_controls.pack(fill=tk.X)
        ttk.Button(file_controls, text="Save Settings", command=self.save_settings).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(file_controls, text="Load Settings", command=self.load_settings).pack(side=tk.LEFT, expand=True, fill=tk.X)

    def _build_analysis_tab(self, parent):
        tracking_frame = ttk.LabelFrame(parent, text="Organism Tracking"); tracking_frame.pack(fill=tk.X, pady=5)
        top_track_frame = ttk.Frame(tracking_frame); top_track_frame.pack(fill=tk.X)
        ttk.Checkbutton(top_track_frame, text="Enable Tracking", variable=self.tracking_enabled).pack(side=tk.LEFT)
        ttk.Label(top_track_frame, text="Min Mass:").pack(side=tk.LEFT, padx=(10,2))
        ttk.Spinbox(top_track_frame, from_=5, to=1000, textvariable=self.min_organism_mass, width=5).pack(side=tk.LEFT)
        ttk.Label(top_track_frame, textvariable=self.organism_count_var).pack(side=tk.RIGHT, padx=10)

        watershed_frame = ttk.Frame(tracking_frame); watershed_frame.pack(fill=tk.X, padx=5, pady=(5,0))
        ttk.Checkbutton(watershed_frame, text="Use Watershed Segmentation", variable=self.use_watershed).pack(side=tk.LEFT)
        ttk.Label(watershed_frame, text="Peak Dist:").pack(side=tk.LEFT, padx=(10,2))
        ttk.Spinbox(watershed_frame, from_=2, to=20, textvariable=self.watershed_peak_distance, width=4).pack(side=tk.LEFT)

        vis_frame = ttk.Frame(tracking_frame); vis_frame.pack(fill=tk.X, padx=10)
        ttk.Checkbutton(vis_frame, text="Outlines", variable=self.show_outlines).pack(side=tk.LEFT, expand=True)
        ttk.Checkbutton(vis_frame, text="CoM", variable=self.show_com).pack(side=tk.LEFT, expand=True)
        ttk.Checkbutton(vis_frame, text="Direction", variable=self.show_direction).pack(side=tk.LEFT, expand=True)
        
        indicator_frame = ttk.LabelFrame(tracking_frame, text="Indicator Controls"); indicator_frame.pack(fill=tk.X, padx=5, pady=(5,0))
        ttk.Label(indicator_frame, text="Vel. Smooth:").grid(row=0, column=0, sticky="w", padx=2)
        ttk.Scale(indicator_frame, from_=0.01, to=0.99, orient=tk.HORIZONTAL, variable=self.velocity_smoothing_factor).grid(row=0, column=1, sticky="ew")
        ttk.Label(indicator_frame, text="Dir. Smooth:").grid(row=1, column=0, sticky="w", padx=2)
        ttk.Scale(indicator_frame, from_=0.01, to=0.99, orient=tk.HORIZONTAL, variable=self.direction_smoothing_factor).grid(row=1, column=1, sticky="ew")
        ttk.Label(indicator_frame, text="Vel. Sense:").grid(row=2, column=0, sticky="w", padx=2)
        ttk.Scale(indicator_frame, from_=1.0, to=20.0, orient=tk.HORIZONTAL, variable=self.velocity_sensitivity).grid(row=2, column=1, sticky="ew")
        ttk.Label(indicator_frame, text="Out. Margin:").grid(row=3, column=0, sticky="w", padx=2)
        ttk.Scale(indicator_frame, from_=0, to=10, orient=tk.HORIZONTAL, variable=self.outline_margin).grid(row=3, column=1, sticky="ew")
        indicator_frame.grid_columnconfigure(1, weight=1)

        vis_select_frame = ttk.LabelFrame(parent, text="Canvas Visualization"); vis_select_frame.pack(fill=tk.X, pady=5)
        
        self.split_screen_button = ttk.Button(vis_select_frame, text="Enable Split Screen", command=self.toggle_split_screen)
        self.split_screen_button.pack(fill=tk.X, pady=(0,5))
        
        self.single_view_frame = ttk.Frame(vis_select_frame)
        self.single_view_frame.pack(fill=tk.X)
        self.vis_dd = ttk.Combobox(self.single_view_frame, textvariable=self.view_mode_var, state="readonly")
        self.vis_dd.pack(fill=tk.X)
        self.vis_dd.bind("<<ComboboxSelected>>", self._on_view_mode_selected)
        
        self.split_view_frame = ttk.Frame(vis_select_frame)
        self.split_view_controls = []
        for i in range(4):
            row = i // 2; col = i % 2
            frame = ttk.Frame(self.split_view_frame)
            frame.grid(row=row, column=col, sticky="ew", padx=2, pady=2)
            dd = ttk.Combobox(frame, textvariable=self.split_view_vars[i], state="readonly")
            dd.pack(fill=tk.X, expand=True)
            self.split_view_controls.append(dd)
        self.split_view_frame.grid_columnconfigure((0,1), weight=1)

        self._update_vis_options()
        self.toggle_split_screen(initial=True)
        
        self.zoom_button = ttk.Button(vis_select_frame, text="Switch to Zoom View", command=self.toggle_zoom_view)
        self.zoom_button.pack(fill=tk.X, pady=(5,0))

        stats_frame = ttk.LabelFrame(parent, text="Selected Organism Stats"); stats_frame.pack(fill=tk.X, pady=5)
        self.stats_mass_label = ttk.Label(stats_frame, text="Mass (per ch): N/A", wraplength=UI_PANEL_WIDTH-20); self.stats_mass_label.pack(anchor="w")
        self.stats_vel_label = ttk.Label(stats_frame, text="Velocity: N/A"); self.stats_vel_label.pack(anchor="w")
        self.stats_dir_label = ttk.Label(stats_frame, text="Direction: N/A"); self.stats_dir_label.pack(anchor="w")
        self.stats_record_button = ttk.Button(stats_frame, text="Record Organism Stats", command=self.record_organism_stats)
        self.stats_record_button.pack(fill=tk.X, pady=(5,0))

        preset_frame = ttk.LabelFrame(parent, text="Organism Presets"); preset_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        preset_list_frame = ttk.Frame(preset_frame); preset_list_frame.pack(fill=tk.BOTH, expand=True)
        self.preset_listbox = tk.Listbox(preset_list_frame); self.preset_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.preset_listbox.bind("<<ListboxSelect>>", self._update_preset_preview)
        preset_scroll = ttk.Scrollbar(preset_list_frame, orient="vertical", command=self.preset_listbox.yview)
        preset_scroll.pack(side=tk.RIGHT, fill=tk.Y); self.preset_listbox.config(yscrollcommand=preset_scroll.set)
        
        preview_frame = ttk.LabelFrame(preset_frame, text="Preview"); preview_frame.pack(fill=tk.X, pady=(5,0))
        self.preset_preview_label = ttk.Label(preview_frame); self.preset_preview_label.pack(pady=5, padx=5)
        self.preset_preview_photo = None

        preset_btn_frame = ttk.Frame(preset_frame); preset_btn_frame.pack(fill=tk.X, pady=(5,0))
        ttk.Button(preset_btn_frame, text="Save Selected", command=self.save_preset).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(preset_btn_frame, text="Load", command=self.load_preset).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(preset_btn_frame, text="Rename", command=self.rename_preset).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(preset_btn_frame, text="Delete", command=self.delete_preset).pack(side=tk.LEFT, expand=True, fill=tk.X)
        self._update_preset_listbox()

    def _rebuild_interactions_ui(self):
        self._destroy_children(self.interactions_frame); num_ch = len(params.channels)
        if num_ch == 0: return
        for i in range(num_ch):
            self.interactions_frame.grid_columnconfigure(i, weight=1)
            for j in range(num_ch):
                frame = ttk.Frame(self.interactions_frame); frame.grid(row=i, column=j, padx=2, pady=2, sticky="ew")
                label = ttk.Label(frame, text=f'{j+1}→{i+1}')
                label.pack()
                scale = ttk.Scale(frame, from_=0.0, to=1.5, orient=tk.HORIZONTAL, value=params.interaction_matrix[i][j], command=lambda val,i=i,j=j: self._update_interaction(i,j,val))
                scale.pack(fill=tk.X, expand=True)
                is_env_i = params.channels[i].channel_type == 'environment'
                is_env_j = params.channels[j].channel_type == 'environment'
                if not params.channels[i].is_active or not params.channels[j].is_active or is_env_i:
                    label.config(state='disabled'); scale.config(state='disabled')

    def _rebuild_channels_ui(self):
        self._destroy_children(self.scrollable_frame); self.kernel_previews = {}
        for i, ch in enumerate(params.channels):
            ch_frame = ttk.LabelFrame(self.scrollable_frame, text=f"Channel {i+1}"); ch_frame.pack(fill=tk.X, expand=True, padx=5, pady=5)
            self._build_single_channel_ui(ch_frame, ch)
    
    def _build_single_channel_ui(self, parent_frame, ch):
        widgets = {}

        type_row = ttk.Frame(parent_frame); type_row.pack(fill=tk.X, pady=2, padx=5)
        ttk.Label(type_row, text="Channel Type:", width=12).pack(side=tk.LEFT)
        type_var = tk.StringVar(value=ch.channel_type.capitalize())
        type_dd = ttk.Combobox(type_row, textvariable=type_var, values=['Lenia', 'Environment'], state="readonly")
        type_dd.pack(side=tk.LEFT, fill=tk.X, expand=True)
        type_dd.bind("<<ComboboxSelected>>", lambda e, id=ch.id, v=type_var: self._on_channel_type_changed(id, v))
        widgets['type_dd'] = type_dd

        widgets['lenia_widgets_frame'] = ttk.Frame(parent_frame)
        widgets['lenia_widgets_frame'].pack(fill=tk.X, expand=True)
        
        parent = widgets['lenia_widgets_frame'] # All subsequent widgets go in here

        top_row = ttk.Frame(parent); top_row.pack(fill=tk.X, pady=2)
        ch_active_var = tk.BooleanVar(value=ch.is_active)
        widgets['ch_check'] = ttk.Checkbutton(top_row, text="Active", variable=ch_active_var, command=lambda id=ch.id, v=ch_active_var: self._toggle_channel_active(id, v))
        widgets['ch_check'].pack(side=tk.LEFT)
        
        controls_frame = ttk.Frame(top_row); controls_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(controls_frame, text="Color:").pack(side=tk.LEFT)
        color_var = tk.StringVar(value=ch.color_hex); color_entry = ttk.Entry(controls_frame, textvariable=color_var, width=10); color_entry.pack(side=tk.LEFT, padx=5)
        color_entry.bind("<FocusOut>", lambda e, id=ch.id, v=color_var: self._update_channel_attr(id, 'color_hex', v.get()))
        ttk.Button(controls_frame, text="Duplicate", command=lambda id=ch.id: self.duplicate_channel(id)).pack(side=tk.RIGHT, padx=2)
        ttk.Button(controls_frame, text="Delete", command=lambda id=ch.id: self.delete_channel(id)).pack(side=tk.RIGHT)

        sim_mode_frame = ttk.Frame(parent); sim_mode_frame.pack(fill=tk.X, padx=5, pady=2)
        flow_active_var = tk.BooleanVar(value=ch.use_flow)
        local_params_var = tk.BooleanVar(value=ch.has_local_params)
        flow_responsive_var = tk.BooleanVar(value=ch.flow_responsive_params)
        
        widgets['flow_check'] = ttk.Checkbutton(sim_mode_frame, text="Use Flow", variable=flow_active_var, command=lambda id=ch.id, v=flow_active_var: self._update_channel_attr(id, 'use_flow', v.get()))
        widgets['flow_check'].pack(side=tk.LEFT)
        
        widgets['local_params_check'] = ttk.Checkbutton(sim_mode_frame, text="Enable Local Params", variable=local_params_var, command=lambda id=ch.id, v=local_params_var: self._toggle_local_params(id, v.get()) or self._rebuild_channels_ui())
        widgets['local_params_check'].pack(side=tk.LEFT, padx=(10,0))
        
        widgets['flow_responsive_check'] = ttk.Checkbutton(sim_mode_frame, text="Flow-Responsive", variable=flow_responsive_var, command=lambda id=ch.id, v=flow_responsive_var: self._toggle_flow_responsive_params(id, v.get()) or self._rebuild_channels_ui())
        widgets['flow_responsive_check'].pack(side=tk.LEFT, padx=(10,0))
        
        widgets['sliders'] = {}
        param_list = [("Mu", 'mu', 0.0, 1.0), ("Sigma", 'sigma', 0.001, 0.2), ("DT", 'dt', 0.01, 0.2), ("Flow", 'flow_strength', 0.0, 20.0)]
        for label, attr, min_val, max_val in param_list:
            widgets['sliders'][attr] = self._create_slider(parent, label, ch.id, attr, min_val, max_val)

        widgets['flow_responsive_frame'] = self._build_flow_responsive_ui(parent, ch)
        
        # Sensorimotor Frame
        sm_frame = ttk.LabelFrame(parent, text="Sensorimotor Settings"); widgets['sensorimotor_frame'] = sm_frame
        sm_check_var = tk.BooleanVar(value=ch.is_sensorimotor)
        widgets['sm_check'] = ttk.Checkbutton(sm_frame, text="Enable Environment Sensing", variable=sm_check_var, command=lambda id=ch.id, v=sm_check_var: self._update_channel_attr(id, 'is_sensorimotor', v.get()) or self._rebuild_channels_ui())
        widgets['sm_check'].pack(fill=tk.X, padx=5)
        
        sm_target_frame = ttk.Frame(sm_frame); widgets['sm_target_frame'] = sm_target_frame; sm_target_frame.pack(fill=tk.X, padx=5)
        ttk.Label(sm_target_frame, text="Target Ch:", width=10).pack(side=tk.LEFT)
        env_channels = {i + 1: c for i, c in enumerate(params.channels) if c.channel_type == 'environment'}
        sm_target_dd = ttk.Combobox(sm_target_frame, values=list(env_channels.keys()), state="readonly", width=5)
        for num, env_ch in env_channels.items():
            if params.channels.index(env_ch) == ch.sensing_target_channel_idx: sm_target_dd.set(num); break
        sm_target_dd.bind("<<ComboboxSelected>>", lambda e, id=ch.id: self._update_channel_attr(id, 'sensing_target_channel_idx', int(e.widget.get())-1))
        sm_target_dd.pack(side=tk.LEFT, fill=tk.X, expand=True)

        widgets['sm_sensitivity_slider'] = self._create_slider(sm_frame, "Sensitivity", ch.id, 'sensorimotor_sensitivity', 0.1, 20.0)
        widgets['sm_kernel_builder'] = self._build_kernel_builder_ui(sm_frame, ch, 'sensing_kernel_layers', "Sensing Kernel Builder")
        
        growth_frame = ttk.Frame(parent); growth_frame.pack(fill=tk.X, padx=5, pady=2); ttk.Label(growth_frame, text="Growth Func:", width=10).pack(side=tk.LEFT)
        growth_dd = ttk.Combobox(growth_frame, values=list(growth_functions.keys()), state="readonly"); growth_dd.set(ch.growth_func_name)
        growth_dd.bind("<<ComboboxSelected>>", lambda e, id=ch.id: self._update_channel_attr(id, 'growth_func_name', e.widget.get())); growth_dd.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        widgets['main_kernel_builder'] = self._build_kernel_builder_ui(parent, ch, 'kernel_layers', "Growth Kernel Builder")

        self._set_channel_ui_state(ch, widgets)
        if not ch.is_active: self._set_widget_state(parent_frame, 'disabled', exceptions=[widgets['ch_check'], widgets['type_dd']])

    def _build_kernel_builder_ui(self, parent, ch, layer_attr, title):
        kb_frame = ttk.LabelFrame(parent, text=title);
        preview_label = ttk.Label(kb_frame); preview_label.pack(side=tk.RIGHT, padx=5)
        
        # Use a unique key for each kernel preview
        preview_key = f"{ch.id}_{layer_attr}"
        self.kernel_previews[preview_key] = preview_label
        
        kernel_controls_frame = ttk.Frame(kb_frame); kernel_controls_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        kb_buttons_frame = ttk.Frame(kernel_controls_frame); kb_buttons_frame.pack(fill=tk.X)
        ttk.Button(kb_buttons_frame, text="Add Layer", command=lambda id=ch.id, la=layer_attr: self.add_kernel_layer(id, la)).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(kb_buttons_frame, text="Clear Layers", command=lambda id=ch.id, la=layer_attr: self.clear_kernel_layers(id, la)).pack(side=tk.LEFT, expand=True, fill=tk.X)
        
        kernel_layers = getattr(ch, layer_attr)
        for k_idx, layer in enumerate(kernel_layers): self._build_kernel_layer_ui(kernel_controls_frame, ch.id, layer, k_idx, len(kernel_layers), layer_attr)
        self._update_kernel_preview(ch.id, layer_attr)
        return kb_frame
        
    def _build_flow_responsive_ui(self, parent, ch):
        frame = ttk.LabelFrame(parent, text="Flow Response Settings")
        
        self._create_slider(frame, "Sensitivity", ch.id, 'flow_sensitivity', 1.0, 50.0)
        self._create_range_slider(frame, "Mu Range", ch.id, 'mu_range', 0.0, 1.0)
        self._create_range_slider(frame, "Sigma Range", ch.id, 'sigma_range', 0.001, 0.2)
        self._create_range_slider(frame, "DT Range", ch.id, 'dt_range', 0.01, 0.2)
        self._create_range_slider(frame, "Flow Str Range", ch.id, 'flow_strength_range', 0.0, 20.0)
        
        return frame

    def _create_range_slider(self, parent, label_text, ch_id, attr, from_, to_):
        frame = ttk.Frame(parent); frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(frame, text=label_text, width=12).pack(side=tk.LEFT)
        
        ch = self._get_channel_by_id(ch_id)
        current_range = getattr(ch, attr, [from_, to_])
        
        min_var = tk.DoubleVar(value=current_range[0])
        max_var = tk.DoubleVar(value=current_range[1])
        
        label = ttk.Label(frame, text=f"[{current_range[0]:.2f}, {current_range[1]:.2f}]", width=12)
        
        def _update(v=None):
            min_val = min_var.get(); max_val = max_var.get()
            if min_val > max_val:
                min_var.set(max_val); min_val = max_val
            label.config(text=f"[{min_val:.2f}, {max_val:.2f}]")
            self._update_channel_attr(ch_id, attr, [min_val, max_val])

        min_scale = ttk.Scale(frame, from_=from_, to=to_, orient=tk.HORIZONTAL, variable=min_var, command=_update)
        max_scale = ttk.Scale(frame, from_=from_, to=to_, orient=tk.HORIZONTAL, variable=max_var, command=_update)
        
        min_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        max_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        label.pack(side=tk.LEFT, padx=(5,0))
        
        return frame
        
    def _build_kernel_layer_ui(self, parent, ch_id, layer, k_idx, total_layers, layer_attr):
        layer_frame = ttk.Frame(parent, borderwidth=1, relief="solid"); layer_frame.pack(fill=tk.X, pady=3, padx=2)
        row1 = ttk.Frame(layer_frame); row1.pack(fill=tk.X)
        
        layer_active_var = tk.BooleanVar(value=layer.get('is_active', True))
        layer_check = ttk.Checkbutton(row1, text="Active", variable=layer_active_var, command=lambda l_id=layer['id'], v=layer_active_var, la=layer_attr: self._toggle_layer_active(l_id, v, la))
        layer_check.pack(side=tk.LEFT)
        
        controls_frame = ttk.Frame(row1); controls_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        if k_idx > 0:
            op_dd = ttk.Combobox(controls_frame, values=['+', '-', '*'], state="readonly", width=3); op_dd.set(layer.get('op', '+'))
            op_dd.bind("<<ComboboxSelected>>", lambda e,l_id=layer['id'],la=layer_attr: self._update_layer_attr(l_id,'op',e.widget.get(),la)); op_dd.pack(side=tk.LEFT)
        else: ttk.Label(controls_frame, text="Base:").pack(side=tk.LEFT)
        type_dd = ttk.Combobox(controls_frame, values=list(kernel_functions.keys()), state="readonly"); type_dd.set(layer['type'])
        type_dd.bind("<<ComboboxSelected>>", lambda e,l_id=layer['id'],la=layer_attr: self._update_layer_attr(l_id,'type',e.widget.get(),la)); type_dd.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        btn_frame = ttk.Frame(controls_frame); btn_frame.pack(side=tk.LEFT)
        up_btn = ttk.Button(btn_frame, text="▲", width=3, command=lambda l_id=layer['id'],la=layer_attr: self.move_layer(l_id, -1, la)); up_btn.pack(side=tk.LEFT); up_btn['state'] = 'disabled' if k_idx==0 else 'normal'
        down_btn = ttk.Button(btn_frame, text="▼", width=3, command=lambda l_id=layer['id'],la=layer_attr: self.move_layer(l_id, 1, la)); down_btn.pack(side=tk.LEFT); down_btn['state'] = 'disabled' if k_idx==total_layers-1 else 'normal'
        ttk.Button(btn_frame, text="Del", width=4, command=lambda l_id=layer['id'],la=layer_attr: self.remove_layer(l_id, la)).pack(side=tk.LEFT)
        
        self._create_slider(layer_frame, "Radius", layer['id'], 'radius', 1, KERNEL_SIZE//2, is_layer=True, is_int=True, layer_attr=layer_attr)
        self._create_slider(layer_frame, "Weight", layer['id'], 'weight', -2.0, 2.0, is_layer=True, layer_attr=layer_attr)

        if not layer.get('is_active', True):
            self._set_widget_state(layer_frame, 'disabled', exceptions=[layer_check])

    def _create_slider(self, parent, label_text, item_id, attr, from_, to, is_layer=False, is_int=False, layer_attr=None):
        frame = ttk.Frame(parent); frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(frame, text=label_text, width=12).pack(side=tk.LEFT)
        
        if is_layer: item=self._get_layer_by_id(item_id, layer_attr); val=item.get(attr)
        else: item=self._get_channel_by_id(item_id); val=getattr(item, attr)
        if val is None: val=0
        
        var=tk.DoubleVar(value=val); label=ttk.Label(frame, text=f"{val:.2f}" if not is_int else f"{int(val)}", width=5)
        scale=ttk.Scale(frame, from_=from_, to=to, orient=tk.HORIZONTAL, variable=var, command=lambda v,lbl=label,id=item_id,a=attr: self._update_slider_val(lbl,id,a,v,is_layer,is_int,layer_attr))
        scale.pack(side=tk.LEFT, fill=tk.X, expand=True); label.pack(side=tk.LEFT)
        return frame

    def _update_slider_val(self, label, item_id, attr, v_str, is_layer, is_int, layer_attr):
        v=float(v_str); label.config(text=f"{int(v)}" if is_int else f"{v:.2f}"); v=int(v) if is_int else v
        if is_layer: self._update_layer_attr(item_id, attr, v, layer_attr)
        else: self._update_channel_attr(item_id, attr, v)
        if (ch:=self._get_channel_by_any_id(item_id, layer_attr)): self._update_kernel_preview(ch.id, layer_attr)
        
    def _update_channel_attr(self, id, attr, val):
        if(ch:=self._get_channel_by_id(id)): setattr(ch,attr,val)
    def _update_layer_attr(self, id, attr, val, layer_attr):
        if(l:=self._get_layer_by_id(id, layer_attr)): l[attr]=val
        if(ch:=self._get_channel_by_any_id(id, layer_attr)): self._update_kernel_preview(ch.id, layer_attr)
    def _update_interaction(self, i, j, v_str): params.interaction_matrix[i][j]=float(v_str)
    def _update_brush_size(self, v_str): self.draw_brush_size=int(float(v_str))
    def _on_draw_channel_selected(self, e): 
        self.draw_channel_index = int(self.draw_channel_var.get()) - 1
        
        if 0 <= self.draw_channel_index < len(params.channels):
            ch = params.channels[self.draw_channel_index]
            if ch.channel_type == 'environment':
                self.param_draw_target.set("Mass")
                self.param_draw_frame.pack_forget()
                self.update() # Force UI update
                return

        self._update_local_param_draw_ui()
        self._update_vis_options()

    def _update_draw_channel_selector(self):
        num_ch=len(params.channels); self.draw_channel_dd['values']=list(range(1,num_ch+1)) if num_ch>0 else []
        if self.draw_channel_index>=num_ch: self.draw_channel_index=max(0,num_ch-1)
        self.draw_channel_var.set(str(self.draw_channel_index+1) if num_ch>0 else "")

    def add_channel(self):
        colors=["#FF0000","#00FF00","#FFFF00","#FF00FF","#FFFFFF"]; params.channels.append(Channel(color_hex=colors[len(params.channels)%len(colors)]))
        for row in params.interaction_matrix: row.append(0.0)
        params.interaction_matrix.append([0.0]*(len(params.channels)-1)+[1.0]); self.game_board=self._initialize_board_circle_seed(); self._build_ui()
    def delete_channel(self, id):
        if len(params.channels)<=1: return
        if (idx:=next((i for i,c in enumerate(params.channels) if c.id==id),None)) is not None:
            if id in local_param_maps:
                del local_param_maps[id]
            params.channels.pop(idx); params.interaction_matrix.pop(idx)
            for row in params.interaction_matrix: row.pop(idx)
            self.game_board=self._initialize_board_circle_seed(); self.draw_channel_index=min(self.draw_channel_index,len(params.channels)-1); self._build_ui()
    def duplicate_channel(self, id):
        if not (src:=self._get_channel_by_id(id)): return
        idx=params.channels.index(src); new=copy.deepcopy(src); new.id=str(uuid.uuid4())
        for layer_attr in ['kernel_layers', 'sensing_kernel_layers']:
            for l in getattr(new, layer_attr): l['id']=str(uuid.uuid4())
        params.channels.insert(idx+1,new)
        if new.has_local_params:
            self._initialize_local_param_maps(new)
        for row in params.interaction_matrix: row.insert(idx+1,row[idx])
        params.interaction_matrix.insert(idx+1,params.interaction_matrix[idx][:]); 
        self.game_board=torch.cat((self.game_board[:idx+1],self.game_board[idx].unsqueeze(0),self.game_board[idx+1:]),dim=0)
        self._build_ui()

    def add_kernel_layer(self, id, layer_attr):
        if(ch:=self._get_channel_by_id(id)): getattr(ch, layer_attr).append({'id':str(uuid.uuid4()),'type':'Gaussian Ring','radius':5,'weight':1.0,'op':'+','is_active':True}); self._rebuild_channels_ui()
    def clear_kernel_layers(self, id, layer_attr):
        if(ch:=self._get_channel_by_id(id)): setattr(ch, layer_attr, [{'id':str(uuid.uuid4()),'type':'Gaussian Ring','radius':13,'weight':1.0,'op':'+','is_active':True}]); self._rebuild_channels_ui()
    def remove_layer(self, id, layer_attr):
        if(ch:=self._get_channel_by_any_id(id, layer_attr)):
            layers = getattr(ch, layer_attr)
            if len(layers) > 1:
                setattr(ch, layer_attr, [l for l in layers if l['id']!=id]); self._rebuild_channels_ui()
    def move_layer(self, id, d, layer_attr):
        if(ch:=self._get_channel_by_any_id(id, layer_attr)):
            layers = getattr(ch, layer_attr)
            if(idx:=next((i for i,l in enumerate(layers) if l['id']==id),None)) is not None:
                if 0<=(n_idx:=idx+d)<len(layers): layers.insert(n_idx,layers.pop(idx)); self._rebuild_channels_ui()
    
    def save_settings(self):
        if not (fp:=filedialog.asksaveasfilename(defaultextension=".json",initialdir=SETTINGS_FOLDER,filetypes=[("JSON","*.json")])): return
        with open(fp,'w') as f: json.dump({'channels':[c.__dict__ for c in params.channels],'interaction_matrix':params.interaction_matrix},f,indent=4)
    def load_settings(self):
        if not (fp:=filedialog.askopenfilename(initialdir=SETTINGS_FOLDER,filetypes=[("JSON","*.json")])): return
        with open(fp,'r') as f: settings=json.load(f)
        params.interaction_matrix=settings.get('interaction_matrix',[[1.0]]); params.channels=[]
        for d in settings.get('channels',[]): ch=Channel(); ch.__dict__.update(d); params.channels.append(ch)
        self.game_board=self._initialize_board_circle_seed()
        self._initialize_all_local_param_maps()
        self.draw_channel_index=0; self._build_ui()

    def reset_seed(self): 
        self.game_board=self._initialize_board_circle_seed()
        self._initialize_all_local_param_maps()
        self.update_canvas()
    def randomize_board(self): 
        self.game_board=self._randomize_board()
        self._initialize_all_local_param_maps()
        self.update_canvas()
    def clear_board(self): 
        self.game_board=self._clear_board()
        self._initialize_all_local_param_maps()
        self.update_canvas()
    def record_gif(self):
        self.is_recording=not self.is_recording
        self.record_button.config(text="Stop & Save GIF" if self.is_recording else "Record Full GIF")
        if not self.is_recording and self.gif_frames:
            if(fp:=filedialog.asksaveasfilename(defaultextension=".gif",initialdir=GIFS_FOLDER,filetypes=[("GIF","*.gif")])): imageio.mimsave(fp,self.gif_frames,fps=30)
            self.gif_frames=[]

    def toggle_pause(self): self.paused=not self.paused
    def toggle_zoom_view(self):
        self.view_is_zoomed = not self.view_is_zoomed
        self.zoom_button.config(text="Switch to Global View" if self.view_is_zoomed else "Switch to Zoom View")
    
    def update_loop(self):
        if not self.paused:
            if len(params.channels)>0:
                self.sim_fields=update_multichannel_board(self.game_board, params.channels, params.interaction_matrix)
                self.game_board=self.sim_fields['final_board']
        
        if self.tracking_enabled.get():
            self._perform_organism_tracking()
            self._update_analysis_display()
        else:
            self.persistent_tracked_organisms={}
            self.selected_organism_id=None
            self._update_analysis_display()
            self.organism_count_var.set("Count: 0")

        self.update_canvas()
        if self.is_recording and self.last_drawn_array is not None:
            self.gif_frames.append(self.last_drawn_array)
        if self.is_stats_recording:
            self._log_stats_and_gifs()
            
        self.root.after(16, self.update_loop)

    def update_canvas(self):
        if len(params.channels) == 0:
            self.canvas.delete("all")
            return

        zoom_box = None
        if self.view_is_zoomed and self.selected_organism_id in self.persistent_tracked_organisms:
            org_data = self.persistent_tracked_organisms[self.selected_organism_id]
            bbox = org_data.get('bbox', (0,0,GRID_DIM[0],GRID_DIM[1]))
            min_r, min_c, max_r, max_c = bbox
            pad = max((max_r-min_r), (max_c-min_c))
            zoom_box = (max(0,min_c-pad), max(0,min_r-pad), min(GRID_DIM[1],max_c+pad), min(GRID_DIM[0],max_r+pad))

        if self.is_split_screen.get():
            final_img = Image.new('RGB', (GRID_DIM[1] * 2, GRID_DIM[0] * 2))
            q_size = (GRID_DIM[1], GRID_DIM[0])
            positions = [(0, 0), (q_size[0], 0), (0, q_size[1]), (q_size[0], q_size[1])]

            for i in range(4):
                view_name = self.split_view_vars[i].get()
                img_array = self._get_view_array_by_name(view_name)
                if img_array is not None:
                    quadrant_img = Image.fromarray(img_array)
                    offset_x, offset_y = 0, 0
                    
                    if zoom_box:
                        offset_x, offset_y = zoom_box[0], zoom_box[1]
                        quadrant_img = quadrant_img.crop(zoom_box)
                    
                    scale_x = quadrant_img.width / (zoom_box[2] - zoom_box[0]) if zoom_box else quadrant_img.width / GRID_DIM[1]
                    scale_y = quadrant_img.height / (zoom_box[3] - zoom_box[1]) if zoom_box else quadrant_img.height / GRID_DIM[0]
                    
                    if self.tracking_enabled.get():
                        self._draw_tracking_overlay(quadrant_img, scale_x, scale_y, offset_x, offset_y)
                        
                    final_img.paste(quadrant_img.resize(q_size, Image.NEAREST), positions[i])
            img = final_img
        else:
            view_name = self.view_mode_var.get()
            img_array = self._get_view_array_by_name(view_name)
            img = Image.fromarray(img_array) if img_array is not None else Image.new('RGB', (GRID_DIM[1], GRID_DIM[0]))

            offset_x, offset_y = 0, 0
            if zoom_box:
                offset_x, offset_y = zoom_box[0], zoom_box[1]
                img = img.crop(zoom_box)
            
            scale_x = img.width / (zoom_box[2] - zoom_box[0]) if zoom_box else img.width / GRID_DIM[1]
            scale_y = img.height / (zoom_box[3] - zoom_box[1]) if zoom_box else img.height / GRID_DIM[0]

            if self.tracking_enabled.get():
                self._draw_tracking_overlay(img, scale_x, scale_y, offset_x, offset_y)
        
        self.last_drawn_array = np.array(img)
        
        if self.canvas.winfo_width() > 1:
            img = img.resize((self.canvas.winfo_width(), self.canvas.winfo_height()), Image.NEAREST)
        
        self.photo = ImageTk.PhotoImage(image=img)
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

    def _get_view_array_by_name(self, view_name):
        if view_name == "Final Board":
            return self._get_multichannel_array(self.game_board, params.channels)
        
        if view_name.startswith("Ch "):
            try:
                parts = view_name.split(': ')
                ch_part, mode = parts[0], parts[1]
                ch_idx = int(ch_part.split(' ')[1]) - 1

                if not (0 <= ch_idx < len(params.channels)): return None
                
                if self.sim_fields and mode in ["Potential Field", "Growth Field", "Flow Field"]:
                    if mode == "Potential Field": return self._get_single_channel_array(self.sim_fields['potential'][ch_idx], 'viridis')
                    if mode == "Growth Field": return self._get_single_channel_array(self.sim_fields['growth'][ch_idx], 'coolwarm')
                    if mode == "Flow Field": return self._get_flow_field_array(self.sim_fields['flow_x'][ch_idx], self.sim_fields['flow_y'][ch_idx])
                
                ch_id = params.channels[ch_idx].id
                if ch_id in local_param_maps and mode in local_param_maps[ch_id]:
                    map_tensor = local_param_maps[ch_id][mode]
                    return self._get_single_channel_array(map_tensor, 'plasma')
            except (ValueError, IndexError):
                return self._get_multichannel_array(self.game_board, params.channels)
        
        return self._get_multichannel_array(self.game_board, params.channels)

    def _get_multichannel_array(self, board_tensor, channels):
        active_channels = [ch for ch in channels if ch.is_active]
        if not active_channels: return np.zeros((*GRID_DIM, 3), dtype=np.uint8)
        active_indices = [i for i, ch in enumerate(channels) if ch.is_active]
        active_board = board_tensor[active_indices, :, :]
        try:
            colors=[Image.new('RGB',(1,1),color=c.color_hex).getpixel((0,0)) for c in active_channels]
            c_tensor=torch.tensor(colors,device=device,dtype=torch.float32)/255.0
        except (ValueError, tk.TclError):
            return np.zeros((*GRID_DIM, 3), dtype=np.uint8)
        img_tensor=torch.clamp(torch.matmul(active_board.permute(1,2,0),c_tensor),0,1)
        return(img_tensor*255).byte().cpu().numpy()

    def _get_single_channel_array(self, tensor, colormap='magma'):
        t_min,t_max=tensor.min(),tensor.max()
        if t_max>t_min: tensor=(tensor-t_min)/(t_max-t_min)
        return(plt.get_cmap(colormap)(tensor.detach().cpu().numpy())[:,:,:3]*255).astype(np.uint8)
    
    def _get_flow_field_array(self, flow_x, flow_y):
        mag=torch.sqrt(flow_x**2+flow_y**2); ang=torch.atan2(flow_y,flow_x)
        h=(ang+np.pi)/(2*np.pi); s=torch.ones_like(mag); v=mag/(mag.max()+1e-6)
        hsv=torch.stack((h,s,v),dim=2).cpu().numpy(); rgb=plt.cm.hsv(hsv[:,:,0]); rgb[:,:,:3]*=hsv[:,:,2,None]
        return(rgb[:,:,:3]*255).astype(np.uint8)

    def _update_kernel_preview(self, id, layer_attr):
        if not (ch:=self._get_channel_by_id(id)): return
        preview_key = f"{id}_{layer_attr}"
        if preview_key not in self.kernel_previews: return
        
        layers = getattr(ch, layer_attr)
        k=generate_composite_kernel(layers); max_abs=torch.max(torch.abs(k)) or 1.0; norm_k=(k+max_abs)/(2*max_abs)
        rgb=(plt.get_cmap('coolwarm')(norm_k.detach().cpu().numpy())[:,:,:3]*255).astype(np.uint8)
        img=Image.fromarray(rgb).resize((80,80),Image.NEAREST); photo=ImageTk.PhotoImage(image=img)
        self.kernel_previews[preview_key].configure(image=photo); self.kernel_previews[preview_key].image=photo

    def draw_on_canvas(self, event, right_click=False):
        if not (0 <= self.draw_channel_index < len(params.channels)): return
        
        cw,ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        if cw <= 1 or ch <= 1: return

        if self.is_split_screen.get():
            quad_w, quad_h = cw / 2, ch / 2
            rel_x, rel_y = event.x % quad_w, event.y % quad_h
            gx = int((rel_x / quad_w) * GRID_DIM[1])
            gy = int((rel_y / quad_h) * GRID_DIM[0])
        else:
            gx = int((event.x / cw) * GRID_DIM[1])
            gy = int((event.y / ch) * GRID_DIM[0])
        
        r=self.draw_brush_size
        ys,ye=max(0,gy-r),min(GRID_DIM[0],gy+r); xs,xe=max(0,gx-r),min(GRID_DIM[1],gx+r)
        if ys>=ye or xs>=xe: return
        
        yy,xx=torch.meshgrid(torch.arange(ys,ye,device=device),torch.arange(xs,xe,device=device),indexing='ij')
        mask=torch.sqrt((yy-gy)**2+(xx-gx)**2)<=r
        
        target = self.param_draw_target.get()
        ch = params.channels[self.draw_channel_index]

        if target == "Mass":
            val = 0.0 if right_click else 1.0
            self.game_board[self.draw_channel_index,ys:ye,xs:xe][mask]=val
        elif ch.has_local_params and ch.id in local_param_maps and target in local_param_maps[ch.id]:
            val = self.param_draw_value.get()
            local_param_maps[ch.id][target][ys:ye,xs:xe][mask] = val

        self.update_canvas()

    def _initialize_board_circle_seed(self):
        num_ch=len(params.channels); board=torch.zeros((num_ch,*GRID_DIM),dtype=torch.float32,device=device)
        if num_ch>0:
            cy,cx=GRID_DIM[0]//2,GRID_DIM[1]//2; r=25; y,x=np.ogrid[-cy:GRID_DIM[0]-cy, -cx:GRID_DIM[1]-cx]
            mask=x*x+y*y<=r*r
            if params.channels[0].channel_type == 'lenia':
                board[0,torch.from_numpy(mask).to(device)]=1.0
        return board
    
    def _randomize_board(self): return torch.rand((len(params.channels),*GRID_DIM),dtype=torch.float32,device=device)
    def _clear_board(self): return torch.zeros((len(params.channels),*GRID_DIM),dtype=torch.float32,device=device)
    def _get_channel_by_id(self,id): return next((c for c in params.channels if c.id==id),None)
    def _get_layer_by_id(self,id, layer_attr):
        if not layer_attr: return None
        return next((l for c in params.channels for l in getattr(c, layer_attr, []) if l['id']==id),None)
    def _get_channel_by_any_id(self,id, layer_attr):
        return self._get_channel_by_id(id) or next((c for c in params.channels for l in getattr(c, layer_attr, []) if l['id']==id), None)

    def _initialize_local_param_maps(self, channel):
        maps = {}
        for name in LOCAL_PARAM_NAMES:
            default_val = getattr(channel, name)
            maps[name] = torch.full(GRID_DIM, default_val, dtype=torch.float32, device=device)
        local_param_maps[channel.id] = maps

    def _initialize_all_local_param_maps(self):
        global local_param_maps
        local_param_maps.clear()
        for ch in params.channels:
            if ch.has_local_params:
                self._initialize_local_param_maps(ch)
    
    def _toggle_local_params(self, channel_id):
        if (ch := self._get_channel_by_id(channel_id)):
            ch.has_local_params = not ch.has_local_params
            if ch.has_local_params and channel_id not in local_param_maps:
                self._initialize_local_param_maps(ch)
            elif not ch.has_local_params and channel_id in local_param_maps:
                ch.flow_responsive_params = False
                ch.is_sensorimotor = False
                del local_param_maps[channel_id]
            self._update_local_param_draw_ui()
            self._update_vis_options()
            
    def _toggle_flow_responsive_params(self, channel_id):
        if (ch := self._get_channel_by_id(channel_id)):
            ch.flow_responsive_params = not ch.flow_responsive_params

    def _on_channel_type_changed(self, channel_id, type_var):
        ch = self._get_channel_by_id(channel_id)
        if ch:
            ch.channel_type = type_var.get().lower()
            self._rebuild_channels_ui()
            self._rebuild_interactions_ui()

    def _set_channel_ui_state(self, ch, widgets):
        # Handle top-level type
        if ch.channel_type == 'environment':
            self._set_widget_state(widgets['lenia_widgets_frame'], 'disabled')
            return
        else:
            self._set_widget_state(widgets['lenia_widgets_frame'], 'normal')

        # Handle Lenia channel states
        is_local = ch.has_local_params
        is_flow_responsive = ch.flow_responsive_params
        is_sm_enabled = ch.is_sensorimotor

        widgets['flow_responsive_check'].config(state='normal' if is_local else 'disabled')
        
        for slider_frame in widgets['sliders'].values():
            self._set_widget_state(slider_frame, 'disabled' if is_local and (is_flow_responsive or is_sm_enabled) else 'normal')
        
        if is_local and is_flow_responsive:
            widgets['flow_responsive_frame'].pack(fill=tk.X, padx=5, pady=5)
        else:
            widgets['flow_responsive_frame'].pack_forget()
        
        sm_frame = widgets['sensorimotor_frame']
        if is_local:
            sm_frame.pack(fill=tk.X, padx=5, pady=5)
            self._set_widget_state(widgets['sm_target_frame'], 'normal' if is_sm_enabled else 'disabled')
            self._set_widget_state(widgets['sm_sensitivity_slider'], 'normal' if is_sm_enabled else 'disabled')
            self._set_widget_state(widgets['sm_kernel_builder'], 'normal' if is_sm_enabled else 'disabled')
        else:
            sm_frame.pack_forget()

    def _update_local_param_draw_ui(self):
        if not (0 <= self.draw_channel_index < len(params.channels)):
            self.param_draw_frame.pack_forget()
            return

        ch = params.channels[self.draw_channel_index]
        is_responsive = ch.flow_responsive_params or ch.is_sensorimotor
        if ch.has_local_params and not is_responsive:
            self.param_draw_frame.pack(fill=tk.X, pady=2)
            self._on_param_draw_target_selected()
        else:
            self.param_draw_frame.pack_forget()
            self.param_draw_target.set("Mass")
            
    def _on_param_draw_target_selected(self, event=None):
        target = self.param_draw_target.get()
        if target == "Mass":
            self.param_draw_val_slider.config(state='disabled')
            self.param_draw_val_label.config(text="N/A")
            return
        
        self.param_draw_val_slider.config(state='normal')
        ranges = {
            'mu': (0.0, 1.0), 'sigma': (0.001, 0.2),
            'dt': (0.01, 0.2), 'flow_strength': (0.0, 20.0)
        }
        from_, to_ = ranges.get(target, (0,1))
        self.param_draw_val_slider.config(from_=from_, to=to_)
        
        if (0 <= self.draw_channel_index < len(params.channels)):
            ch = params.channels[self.draw_channel_index]
            self.param_draw_value.set(getattr(ch, target))
            self._update_param_draw_slider()

    def _update_param_draw_slider(self, event=None):
        val = self.param_draw_value.get()
        self.param_draw_val_label.config(text=f"Val: {val:.2f}")

    def _on_view_mode_selected(self, event=None):
        selected = self.view_mode_var.get()
        if selected == "Final Board":
            params.view_mode = "Final Board"
        elif selected.startswith("Ch "):
            try:
                parts = selected.split(': ')
                ch_part, mode = parts[0], parts[1]
                ch_idx = int(ch_part.split(' ')[1]) - 1
                params.active_channel_idx_for_view = ch_idx
                params.view_mode = mode
            except (ValueError, IndexError):
                params.view_mode = "Final Board"

    def _update_vis_options(self):
        options = ["Final Board"]
        for i, ch in enumerate(params.channels):
            if ch.channel_type == 'environment': continue
            ch_num = i + 1
            options.extend([f"Ch {ch_num}: Potential Field", f"Ch {ch_num}: Growth Field", f"Ch {ch_num}: Flow Field"])
            if ch.has_local_params:
                for name in LOCAL_PARAM_NAMES:
                    options.append(f"Ch {ch_num}: {name}")

        self.vis_dd['values'] = options
        for dd in self.split_view_controls:
            dd['values'] = options

        if self.view_mode_var.get() not in options:
            self.view_mode_var.set("Final Board")
        for var in self.split_view_vars:
            if var.get() not in options:
                var.set("Final Board")

    def toggle_split_screen(self, initial=False):
        if not initial:
            self.is_split_screen.set(not self.is_split_screen.get())

        if self.is_split_screen.get():
            self.split_screen_button.config(text="Disable Split Screen")
            self.single_view_frame.pack_forget()
            self.split_view_frame.pack(fill=tk.X)
        else:
            self.split_screen_button.config(text="Enable Split Screen")
            self.split_view_frame.pack_forget()
            self.single_view_frame.pack(fill=tk.X)

    def _perform_organism_tracking(self):
        if len(params.channels)==0:
            self.persistent_tracked_organisms={}
            self.current_labeled_map = None
            self._base_contour_cache.clear()
            self._touches_seam_cache.clear()
            return
            
        lenia_indices = [i for i, ch in enumerate(params.channels) if ch.channel_type == 'lenia']
        if not lenia_indices: return

        mag_map = torch.sum(self.game_board[lenia_indices,:,:], dim=0).detach().cpu().numpy()
        binary_map = mag_map > 0.1

        seg_mode = "watershed" if self.use_watershed.get() else "label"
        labeled_map = toroidal_segment(binary_map, mode=seg_mode, peak_distance=self.watershed_peak_distance.get(), min_size=self.min_organism_mass.get())
        self.current_labeled_map = labeled_map

        current_props = measure.regionprops(labeled_map, intensity_image=mag_map)
        label_to_prop = {p.label: p for p in current_props}

        labels_present = np.unique(labeled_map); labels_present = labels_present[labels_present != 0]
        current_centroids = []
        label_info = {}
        H, W = GRID_DIM
        for lab in labels_present:
            rr, cc = np.nonzero(labeled_map == lab)
            wts = mag_map[rr, cc]
            cent = toroidal_weighted_centroid(rr, cc, wts, H, W)
            min_r, max_r = int(rr.min()), int(rr.max())+1
            min_c, max_c = int(cc.min()), int(cc.max())+1
            label_info[lab] = {'centroid': cent, 'mask_indices': (rr, cc), 'mass_total': float(np.sum(wts)), 'prop': label_to_prop.get(lab, None), 'bbox': (min_r, min_c, max_r, max_c)}
            current_centroids.append(cent)

            touches = (rr == 0).any() or (rr == H-1).any() or (cc == 0).any() or (cc == W-1).any()
            self._touches_seam_cache[lab] = bool(touches)

            if lab not in self._base_contour_cache:
                if label_to_prop.get(lab, None) is not None:
                    prop = label_to_prop[lab]
                    filled = binary_fill_holes(prop.image)
                    base = measure.find_contours(np.pad(filled, 1), 0.5)
                    mapped = [np.column_stack((c[:,0] + prop.bbox[0] - 1, c[:,1] + prop.bbox[1] - 1)) for c in base]
                    self._base_contour_cache[lab] = mapped
                else:
                    y0, x0, y1, x1 = label_info[lab]['bbox']
                    roi = np.zeros((y1-y0, x1-x0), dtype=bool)
                    roi[rr - y0, cc - x0] = True
                    filled = binary_fill_holes(roi)
                    base = measure.find_contours(np.pad(filled, 1), 0.5)
                    mapped = [np.column_stack((c[:,0] + y0 - 1, c[:,1] + x0 - 1)) for c in base]
                    self._base_contour_cache[lab] = mapped

        old_centroids_map = {pid: data['centroid'] for pid, data in self.persistent_tracked_organisms.items()}
        new_org_data, matched_new_labels, disappeared_ids = {}, set(), set(old_centroids_map.keys())

        if old_centroids_map and len(current_centroids) > 0:
            old_ids = list(old_centroids_map.keys())
            dist_matrix = np.zeros((len(old_ids), len(current_centroids)), dtype=np.float64)
            labels_array = np.array(list(labels_present), dtype=np.int32)

            for i, oid in enumerate(old_ids):
                oa = old_centroids_map[oid]
                for j, nb in enumerate(current_centroids):
                    dist_matrix[i, j] = toroidal_distance(oa, nb, H, W)

            for i, old_id in enumerate(old_ids):
                if dist_matrix.shape[1] > 0:
                    min_dist_idx = int(np.argmin(dist_matrix[i, :]))
                    if dist_matrix[i, min_dist_idx] < 25:
                        lab = int(labels_array[min_dist_idx])
                        if lab not in matched_new_labels:
                            info = label_info[lab]
                            prop = info['prop']
                            new_org_data[old_id] = {**self.persistent_tracked_organisms[old_id], 'skimage_props': prop, 'centroid': info['centroid'], 'label_id': lab, 'mask_indices': info['mask_indices'], 'mass': info['mass_total'], 'bbox': info['bbox']}
                            matched_new_labels.add(lab)
                            disappeared_ids.discard(old_id)
                            dist_matrix[:, min_dist_idx] = np.inf

        unmatched_labels = [lab for lab in labels_present if lab not in matched_new_labels]
        self.division_events = []

        for old_id in list(disappeared_ids):
            if old_id not in self.persistent_tracked_organisms: continue
            old_cent = self.persistent_tracked_organisms[old_id]['centroid']
            nearby = []
            for lab in unmatched_labels:
                cent = label_info[lab]['centroid']
                if toroidal_distance(old_cent, cent, H, W) < 40:
                    nearby.append(lab)
            if len(nearby) >= 2:
                div_event = {'parent_id': old_id, 'parent_mass': self.persistent_tracked_organisms[old_id]['mass'], 'children': {}}
                for lab in nearby:
                    new_id = self.next_persistent_id; self.next_persistent_id += 1
                    info = label_info[lab]
                    new_org_data[new_id] = {'id': new_id, 'parent_id': old_id, 'skimage_props': info['prop'], 'centroid': info['centroid'], 'label_id': lab, 'mask_indices': info['mask_indices'], 'mass': info['mass_total'], 'bbox': info['bbox']}
                    unmatched_labels.remove(lab)
                    div_event['children'][new_id] = info['mass_total']
                self.division_events.append(div_event)

        used_colors = {data.get('color') for data in new_org_data.values() if data.get('color')}
        for lab in unmatched_labels:
            new_id = self.next_persistent_id; self.next_persistent_id += 1
            new_color = next((c for c in self.PALETTE if c not in used_colors), self.PALETTE[new_id % len(self.PALETTE)])
            info = label_info[lab]
            new_org_data[new_id] = {'id': new_id, 'color': new_color, 'skimage_props': info['prop'], 'centroid': info['centroid'], 'label_id': lab, 'mask_indices': info['mask_indices'], 'mass': info['mass_total'], 'bbox': info['bbox']}
            used_colors.add(new_color)

        for pid, data in new_org_data.items():
            cur_cent = np.array(data['centroid'], dtype=np.float64)
            vel = np.array([0., 0.])
            if pid in self.persistent_tracked_organisms:
                prev_cent = np.array(self.persistent_tracked_organisms[pid]['centroid'], dtype=np.float64)
                dr = toroidal_delta(prev_cent[0], cur_cent[0], H)
                dc = toroidal_delta(prev_cent[1], cur_cent[1], W)
                vel = np.array([dr, dc], dtype=np.float64)
                v_alpha = 1.0 - self.velocity_smoothing_factor.get()
                prev_vel = np.array(self.persistent_tracked_organisms[pid].get('smooth_vel', vel), dtype=np.float64)
                data['smooth_vel'] = v_alpha * vel + (1.0 - v_alpha) * prev_vel
                d_alpha = 1.0 - self.direction_smoothing_factor.get()
                norm_vel = vel / (np.linalg.norm(vel) + 1e-6)
                prev_dir = np.array(self.persistent_tracked_organisms[pid].get('smooth_direction', norm_vel), dtype=np.float64)
                new_dir = d_alpha * norm_vel + (1.0 - d_alpha) * prev_dir
                data['smooth_direction'] = new_dir / (np.linalg.norm(new_dir) + 1e-6)
            else:
                data['smooth_vel'] = vel
                data['smooth_direction'] = vel / (np.linalg.norm(vel) + 1e-6)

        self.persistent_tracked_organisms = new_org_data
        if self.selected_organism_id not in self.persistent_tracked_organisms:
            self.selected_organism_id = None
        
        count = len(self.persistent_tracked_organisms)
        self.organism_count_var.set(f"Count: {count}")

    def _margin_contours(self, data, margin):
        lab = data.get('label_id')
        if lab is None or self.current_labeled_map is None:
            return []

        if margin < 0.5 and lab in self._base_contour_cache:
            return self._base_contour_cache[lab]

        rr, cc = data['mask_indices']
        H, W = GRID_DIM
        touches = self._touches_seam_cache.get(lab, False)
        r = int(np.ceil(margin))
        se = self._get_selem(r)

        if not touches:
            y0, x0, y1, x1 = data.get('bbox', (int(rr.min()), int(cc.min()), int(rr.max())+1, int(cc.max())+1))
            pad = r + 2
            y0p = max(0, y0 - pad); x0p = max(0, x0 - pad)
            y1p = min(H, y1 + pad); x1p = min(W, x1 + pad)
            roi = np.zeros((y1p - y0p, x1p - x0p), dtype=bool)
            roi[rr - y0p, cc - x0p] = True
            if r > 0:
                roi = binary_dilation(roi, structure=se)
            cnts = measure.find_contours(np.pad(roi, 1), 0.5)
            return [np.column_stack((c[:,0] + y0p - 1, c[:,1] + x0p - 1)) for c in cnts]

        big = np.zeros((H*3, W*3), dtype=bool)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                big[rr + (dy+1)*H, cc + (dx+1)*W] = True
        if r > 0:
            big = binary_dilation(big, structure=se)
        center = big[H:2*H, W:2*W]
        cnts = measure.find_contours(np.pad(center, 1), 0.5)
        return [np.column_stack((c[:,0] - 1, c[:,1] - 1)) for c in cnts]

    def _draw_tracking_overlay(self, pil_image, scale_x, scale_y, offset_x=0, offset_y=0):
        draw=ImageDraw.Draw(pil_image)
        margin=float(self.outline_margin.get())

        for pid, data in self.persistent_tracked_organisms.items():
            if self.show_outlines.get():
                contours = self._margin_contours(data, margin)
                color = data.get('color', "#FFFFFF") if pid!=self.selected_organism_id else "yellow"
                for cont in contours:
                    xs = (cont[:,1] - offset_x) * scale_x
                    ys = (cont[:,0] - offset_y) * scale_y
                    pts = np.column_stack((xs, ys)).ravel().tolist()
                    if len(pts) >= 4:
                        draw.line(pts, fill=color, width=1)

            cy,cx=data['centroid']
            sx,sy=(cx-offset_x)*scale_x,(cy-offset_y)*scale_y
            if self.show_com.get(): draw.ellipse((sx-1,sy-1,sx+1,sy+1), fill="#888888")
            if self.show_direction.get():
                vy,vx=data['smooth_vel']; vel=np.sqrt(vx**2+vy**2); direction_vec = data['smooth_direction']
                if vel > 0.01:
                    vy_dir,vx_dir=direction_vec
                    length=max(3,min(40,vel*self.velocity_sensitivity.get())); tail_len=3
                    draw.line(((sx-vx_dir*tail_len,sy-vy_dir*tail_len),(sx+vx_dir*length,sy+vy_dir*length)),fill="#888888",width=1)

    def _on_canvas_select(self, event):
        if not self.tracking_enabled.get() or not self.persistent_tracked_organisms or self.current_labeled_map is None:
            return

        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        if cw <= 1 or ch <= 1: return
        
        cy, cx = 0, 0

        if self.is_split_screen.get():
            quad_w, quad_h = cw / 2, ch / 2
            rel_x, rel_y = event.x % quad_w, event.y % quad_h
            cx = int((rel_x / quad_w) * GRID_DIM[1])
            cy = int((rel_y / quad_h) * GRID_DIM[0])
        elif self.view_is_zoomed and self.selected_organism_id in self.persistent_tracked_organisms:
            org_data = self.persistent_tracked_organisms[self.selected_organism_id]
            bbox = org_data.get('bbox')
            if bbox is None: return
            
            min_r, min_c, max_r, max_c = bbox
            pad = max((max_r - min_r), (max_c - min_c))
            
            zx0, zy0 = max(0, min_c - pad), max(0, min_r - pad)
            zx1, zy1 = min(GRID_DIM[1], max_c + pad), min(GRID_DIM[0], max_r + pad)
            
            rel_x = event.x / cw
            rel_y = event.y / ch
            cx = int(zx0 + rel_x * (zx1 - zx0))
            cy = int(zy0 + rel_y * (zy1 - zy0))
        else:
            cx = int((event.x / cw) * GRID_DIM[1])
            cy = int((event.y / ch) * GRID_DIM[0])

        lab = int(self.current_labeled_map[cy, cx]) if (0 <= cy < GRID_DIM[0] and 0 <= cx < GRID_DIM[1]) else 0
        
        new_selected_id = None
        if lab != 0:
            for pid, data in self.persistent_tracked_organisms.items():
                if data.get('label_id') == lab:
                    new_selected_id = pid
                    break
        
        if self.selected_organism_id == new_selected_id:
            self.selected_organism_id = None
        else:
            self.selected_organism_id = new_selected_id

        self._update_analysis_display()
        
    def _masses_for_label(self, label_id):
        if self.current_labeled_map is None or label_id is None or label_id == 0:
            return [0.0 for _ in range(len(params.channels))]
        rr, cc = np.nonzero(self.current_labeled_map == int(label_id))
        masses = []
        g = self.game_board.detach().cpu().numpy()
        for i in range(len(params.channels)):
            masses.append(float(np.sum(g[i, rr, cc])))
        return masses

    def _update_analysis_display(self):
        if self.selected_organism_id and self.selected_organism_id in self.persistent_tracked_organisms:
            data=self.persistent_tracked_organisms[self.selected_organism_id]
            masses = self._masses_for_label(data.get('label_id'))
            self.stats_mass_label.config(text=f"Mass (per ch): {', '.join(f'{m:.2f}' for m in masses)}")
            vy,vx=data['smooth_vel']; vel=np.sqrt(vx**2+vy**2); direction=np.degrees(np.arctan2(-vy,vx))
            self.stats_vel_label.config(text=f"Velocity: {vel:.2f} px/frame")
            self.stats_dir_label.config(text=f"Direction: {direction:.1f}°")
            return
        self.stats_mass_label.config(text="Mass (per ch): N/A"); self.stats_vel_label.config(text="Velocity: N/A"); self.stats_dir_label.config(text="Direction: N/A")
    
    def _load_presets(self):
        presets={}
        if not os.path.isdir(PRESETS_FOLDER): return presets
        for fn in os.listdir(PRESETS_FOLDER):
            if fn.endswith(".json"):
                try:
                    with open(os.path.join(PRESETS_FOLDER,fn),'r') as f: presets[fn.replace(".json","")]=json.load(f)
                except: pass
        return presets
    def _update_preset_listbox(self):
        self.preset_listbox.delete(0,tk.END)
        for name in sorted(self.organism_presets.keys()): self.preset_listbox.insert(tk.END, name)
        self._update_preset_preview()
    def save_preset(self):
        if not self.selected_organism_id or self.selected_organism_id not in self.persistent_tracked_organisms: messagebox.showwarning("Save Error", "No organism selected to save."); return
        name = simpledialog.askstring("Save Preset", "Enter preset name:", parent=self.root)
        if not name or not name.strip(): return
        self._save_preset_logic(name)
        
    def _save_preset_logic(self, name):
        if not self.selected_organism_id or self.selected_organism_id not in self.persistent_tracked_organisms: return
        org_data=self.persistent_tracked_organisms[self.selected_organism_id].get('skimage_props', None)
        if org_data is None:
            messagebox.showwarning("Save Error", "Selected organism has no bounding box to save.")
            return
            
        min_r,min_c,max_r,max_c=org_data.bbox
        tensor_slice=self.game_board[:,min_r:max_r,min_c:max_c]
        
        local_maps_data = {}
        for i, ch in enumerate(params.channels):
            if ch.has_local_params and ch.id in local_param_maps:
                channel_slices = {}
                for param_name in LOCAL_PARAM_NAMES:
                    param_map_slice = local_param_maps[ch.id][param_name][min_r:max_r, min_c:max_c]
                    channel_slices[param_name] = param_map_slice.detach().cpu().tolist()
                local_maps_data[ch.id] = channel_slices

        preset_data={'name':name, 'tensor':tensor_slice.detach().cpu().tolist(), 'params':{'channels':[c.__dict__ for c in params.channels],'interaction_matrix':params.interaction_matrix}, 'local_param_maps': local_maps_data}
        
        with open(os.path.join(PRESETS_FOLDER, f"{name}.json"),'w') as f: json.dump(preset_data, f, indent=2)
        self.organism_presets[name]=preset_data; self._update_preset_listbox()
        
    def load_preset(self):
        if not (sel:=self.preset_listbox.curselection()): return
        preset=self.organism_presets[self.preset_listbox.get(sel[0])]
        p_data=preset['params']; params.interaction_matrix=p_data['interaction_matrix']; params.channels=[]
        for d in p_data['channels']: ch=Channel(); ch.__dict__.update(d); params.channels.append(ch)
        
        self.game_board=self._clear_board()
        self._initialize_all_local_param_maps()
        
        tensor_data=torch.tensor(preset['tensor'],device=device)
        h,w=tensor_data.shape[1],tensor_data.shape[2]; ch,cw=GRID_DIM[0]//2,GRID_DIM[1]//2
        y_start, x_start = ch-h//2, cw-w//2
        y_end, x_end = y_start + h, x_start + w
        self.game_board[:, y_start:y_end, x_start:x_end]=tensor_data

        local_maps_to_load = preset.get('local_param_maps', {})
        for ch_id, channel_slices in local_maps_to_load.items():
            if ch_id in local_param_maps:
                for param_name, slice_data in channel_slices.items():
                    slice_tensor = torch.tensor(slice_data, device=device)
                    sh, sw = slice_tensor.shape
                    if sh == h and sw == w:
                        local_param_maps[ch_id][param_name][y_start:y_end, x_start:x_end] = slice_tensor

        self.draw_channel_index=0; self._build_ui(); self.update_canvas()
    def delete_preset(self):
        if not (sel:=self.preset_listbox.curselection()): return
        name=self.preset_listbox.get(sel[0])
        if name in self.organism_presets and messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete preset '{name}'?"):
            del self.organism_presets[name]; os.remove(os.path.join(PRESETS_FOLDER, f"{name}.json"))
            self._update_preset_listbox()
    def rename_preset(self):
        if not (sel:=self.preset_listbox.curselection()): return
        old_name = self.preset_listbox.get(sel[0])
        new_name = simpledialog.askstring("Rename Preset", "Enter new name:", initialvalue=old_name, parent=self.root)
        if not new_name or not new_name.strip() or new_name == old_name: return
        if new_name in self.organism_presets: messagebox.showerror("Rename Error", "A preset with that name already exists."); return
        self.organism_presets[new_name] = self.organism_presets.pop(old_name)
        os.rename(os.path.join(PRESETS_FOLDER, f"{old_name}.json"), os.path.join(PRESETS_FOLDER, f"{new_name}.json"))
        self._update_preset_listbox()
    def _update_preset_preview(self, event=None):
        if not (sel:=self.preset_listbox.curselection()): self.preset_preview_label.config(image=''); return
        name = self.preset_listbox.get(sel[0]); preset = self.organism_presets[name]
        
        preview_channels = []
        for ch_data in preset['params']['channels']:
            ch = Channel()
            ch.__dict__.update(ch_data)
            preview_channels.append(ch)

        arr = self._get_multichannel_array(torch.tensor(preset['tensor'], device=device), preview_channels)
        if arr is not None:
            img = Image.fromarray(arr).resize((100, 100), Image.NEAREST)
            self.preset_preview_photo = ImageTk.PhotoImage(image=img)
            self.preset_preview_label.config(image=self.preset_preview_photo)

    def record_organism_stats(self):
        if not self.selected_organism_id and not self.is_stats_recording: messagebox.showwarning("Recording Error", "An organism must be selected to begin recording."); return
        self.is_stats_recording = not self.is_stats_recording
        self.stats_record_button.config(text="Stop & Log Stats" if self.is_stats_recording else "Record Organism Stats")
        if self.is_stats_recording:
            session_name = simpledialog.askstring("Recording Session", "Enter a name for this session:", initialvalue=f"rec_{datetime.datetime.now():%Y%m%d_%H%M%S}", parent=self.root)
            if not session_name or not session_name.strip(): self.is_stats_recording=False; self.stats_record_button.config(text="Record Organism Stats"); return
            self._save_preset_logic(session_name)
            self.rec_dir=os.path.join(GIFS_FOLDER,session_name); self._ensure_dir(self.rec_dir)
            with open(os.path.join(self.rec_dir,"parameters.json"),'w') as f: json.dump({'channels':[c.__dict__ for c in params.channels],'interaction_matrix':params.interaction_matrix},f,indent=4)
            self.stats_log_path=os.path.join(self.rec_dir,"stats_log.csv")
            self.stats_log=[['frame',*[f'mass_ch{i+1}' for i in range(len(params.channels))],'velocity','direction','division_event','parent_id','mass_ratio']]
            self.stats_gif_writers={v:imageio.get_writer(os.path.join(self.rec_dir,f"{v.lower().replace(' ','_')}.gif"),mode='I') for v in ["Final Board","Potential Field","Growth Field","Flow Field"]}
        else:
            for writer in self.stats_gif_writers.values(): writer.close()
            pd.DataFrame(self.stats_log[1:],columns=self.stats_log[0]).to_csv(self.stats_log_path,index=False)
            self.stats_gif_writers={}; self.stats_log=[]

    def _log_stats_and_gifs(self):
        if not self.selected_organism_id or self.selected_organism_id not in self.persistent_tracked_organisms: self.record_organism_stats(); return
        org_data=self.persistent_tracked_organisms[self.selected_organism_id]
        masses = self._masses_for_label(org_data.get('label_id'))
        vy,vx=org_data['smooth_vel']; vel=np.sqrt(vx**2+vy**2); direction=np.degrees(np.arctan2(-vy,vx))
        
        div_event, parent_id, mass_ratio = 0, -1, 0.0
        if (pid:=org_data.get('parent_id')) is not None:
            for event in self.division_events:
                if event['parent_id']==pid and self.selected_organism_id in event['children']:
                    div_event=1; parent_id=pid
                    mass_ratio = event['children'][self.selected_organism_id]/event['parent_mass'] if event['parent_mass']>0 else 0
                    del event['children'][self.selected_organism_id]
                    if not event['children']: self.division_events.remove(event)
                    break
        self.stats_log.append([len(self.stats_log),*masses,vel,direction,div_event,parent_id,mass_ratio])

        bbox = org_data.get('bbox', None)
        if bbox is None and org_data.get('skimage_props') is not None:
            bbox = org_data['skimage_props'].bbox
        if bbox is None:
            crop_box=(0,0,GRID_DIM[1],GRID_DIM[0])
        else:
            min_r,min_c,max_r,max_c=bbox; pad=30
            crop_box=(max(0,min_c-pad), max(0,min_r-pad), min(GRID_DIM[1],max_c+pad), min(GRID_DIM[0],max_r+pad))
        
        original_view_mode=self.view_mode_var.get()
        for view_mode, writer in self.stats_gif_writers.items():
            arr=self._get_view_array_by_name(f"Ch {self.draw_channel_index+1}: {view_mode}")
            if arr is not None: writer.append_data(Image.fromarray(arr).crop(crop_box))
        self.view_mode_var.set(original_view_mode)
    
    def _toggle_channel_active(self, channel_id, var):
        if (channel := self._get_channel_by_id(channel_id)):
            channel.is_active = var.get()
            self._rebuild_channels_ui()
            self._rebuild_interactions_ui()

    def _toggle_layer_active(self, layer_id, var, layer_attr):
        if (layer := self._get_layer_by_id(layer_id, layer_attr)):
            layer['is_active'] = var.get()
            self._rebuild_channels_ui()
    
    def _set_widget_state(self, parent, state, exceptions=[]):
        for child in parent.winfo_children():
            if child in exceptions:
                continue
            if 'state' in child.configure():
                child.configure(state=state)
            self._set_widget_state(child, state, exceptions)

if __name__ == '__main__':
    root = tk.Tk()
    app = LeniaApp(root)
    root.mainloop()