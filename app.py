import tkinter as tk
from tkinter import ttk
import numpy as np
import torch
import os

# Configuration and Core Modules
from config import *
from simulation import *
from toroidal_helpers import *
import ui_components
import canvas_manager
import organism_tracker
import file_handler
import recording_manager
import event_handlers

class LeniaApp:
    def __init__(self, root):
        """Initializes application state, UI elements, and kicks off the update loop."""
        self.root = root
        self.root.title("Lenia Lab")
        self._ensure_dir(SETTINGS_FOLDER)
        self._ensure_dir(GIFS_FOLDER)
        self._ensure_dir(PRESETS_FOLDER)
        self._ensure_dir(SAVED_ORGANISMS_FOLDER)
        
        # --- Application and Simulation State ---
        self.sim_state = SimulationState()
        self.paused = False
        self.game_board = self._initialize_board_circle_seed()
        self.sim_fields = None
        self.last_drawn_array = None
        
        # --- Optimization Attributes ---
        self.kernel_cache = {}
        self.grid_y, self.grid_x = torch.meshgrid(
            torch.linspace(-1, 1, steps=GRID_DIM[0], device=device),
            torch.linspace(-1, 1, steps=GRID_DIM[1], device=device),
            indexing='ij'
        )

        # --- Tracking State ---
        self.tracking_enabled = tk.BooleanVar(value=False)
        self.show_outlines = tk.BooleanVar(value=True)
        self.show_com = tk.BooleanVar(value=True)
        self.show_direction = tk.BooleanVar(value=True)
        self.persistent_tracked_organisms = {}
        self.next_persistent_id = 0
        self.selected_organism_id = None
        self.organism_presets = file_handler.load_presets(self)
        self.min_organism_mass = tk.IntVar(value=20)
        self.use_watershed = tk.BooleanVar(value=False)
        self.watershed_peak_distance = tk.IntVar(value=7)
        self.division_events = []
        self.velocity_smoothing_factor = tk.DoubleVar(value=0.8)
        self.direction_smoothing_factor = tk.DoubleVar(value=0.7)
        self.velocity_sensitivity = tk.DoubleVar(value=5.0)
        self.organism_count_var = tk.StringVar(value="Count: 0")
        self.current_labeled_map = None
        self._base_contour_cache = {}
        self._touches_seam_cache = {}
        self._selem_cache = {}
        self.outline_margin = tk.DoubleVar(value=2.0)

        # --- Visualization State ---
        self.view_is_zoomed = False
        self.is_split_screen = tk.BooleanVar(value=False)
        self.split_view_vars = [tk.StringVar(value="Final Board") for _ in range(4)]
        self.view_mode_var = tk.StringVar(value="Final Board")

        # --- Recording State ---
        self.is_recording = False
        self.gif_frames = []
        self.is_stats_recording = False
        self.stats_log = []
        self.stats_gif_writers = {}
        
        # --- Drawing State ---
        self.draw_channel_index = 0
        self.draw_brush_size = 10
        self.param_draw_target = tk.StringVar(value="Mass")
        self.param_draw_value = tk.DoubleVar(value=0.5)

        self.PALETTE = ['#FF5733', '#33FF57', '#3357FF', '#FF33A1', '#A133FF', '#33FFA1', '#FFC300', '#C70039', '#900C3F', '#581845']

        # --- UI Setup ---
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(self.main_frame, width=GRID_SIZE, height=GRID_SIZE, bg='black')
        self.canvas_image_id = None  # track the single canvas image to avoid leaking items
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.ui_frame = ttk.Frame(self.main_frame, width=UI_PANEL_WIDTH)
        self.ui_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
        self.ui_frame.pack_propagate(False)
        self._build_ui()

        # --- Bindings ---
        self.root.bind("<space>", lambda e: self.toggle_pause())
        self.canvas.bind("<B1-Motion>", self.draw_on_canvas)
        self.canvas.bind("<Button-1>", self.draw_on_canvas)
        self.canvas.bind("<B3-Motion>", lambda e: self.draw_on_canvas(e, right_click=True))
        self.canvas.bind("<Button-3>", lambda e: self.draw_on_canvas(e, right_click=True))
        self.canvas.bind("<Shift-Button-1>", self._on_canvas_select)

        self._initialize_all_local_param_maps()
        self.update_all_kernels()
        self.update_loop()

    # --- Main Loop ---
    def update_loop(self):
        """Advances the simulation, updates tracking visuals, and schedules the next frame."""
        if not self.paused and len(self.sim_state.channels) > 0:
            self.sim_fields = update_multichannel_board(self.game_board, self.sim_state, self.grid_y, self.grid_x, self.kernel_cache)
            self.game_board = self.sim_fields['final_board']
        
        if self.tracking_enabled.get():
            organism_tracker.perform_organism_tracking(self)
            organism_tracker.update_analysis_display(self)
        else:
            self.persistent_tracked_organisms = {}
            self.selected_organism_id = None
            organism_tracker.update_analysis_display(self)
            self.organism_count_var.set("Count: 0")

        canvas_manager.update_canvas(self)
        
        if self.is_recording and self.last_drawn_array is not None:
            self.gif_frames.append(self.last_drawn_array)
        if self.is_stats_recording:
            recording_manager.log_stats_and_gifs(self)
            
        self.root.after(16, self.update_loop)

    # --- Kernel Caching ---
    def update_kernel(self, channel_id):
        """Regenerates a kernel and its gradients for the given channel id."""
        ch = self._get_channel_by_id(channel_id)
        if ch:
            kernel = generate_composite_kernel(ch.kernel_layers)
            grad_y, grad_x = torch.gradient(kernel)
            self.kernel_cache[ch.id] = (kernel, grad_y, grad_x)

    def update_all_kernels(self):
        """Rebuilds the kernel cache for every channel in the simulation."""
        self.kernel_cache.clear()
        for ch in self.sim_state.channels:
            self.update_kernel(ch.id)
            
    # --- Delegate Methods ---
    def _build_ui(self):
        """Builds or rebuilds the UI panels using the ui_components module."""
        ui_components._build_ui(self)

    def _rebuild_channels_ui(self):
        """Recreates channel-specific UI sections."""
        ui_components._rebuild_channels_ui(self)

    def _rebuild_interactions_ui(self):
        """Refreshes the interaction matrix UI grid."""
        ui_components._rebuild_interactions_ui(self)

    def update_canvas(self):
        """Draws the latest simulation state to the canvas."""
        canvas_manager.update_canvas(self)

    def _update_kernel_preview(self, id):
        """Refreshes the kernel preview for a channel after edits."""
        self.update_kernel(id)
        canvas_manager.update_kernel_preview(self, id)

    def draw_on_canvas(self, event, right_click=False):
        """Handles drawing interactions on the simulation canvas."""
        canvas_manager.draw_on_canvas(self, event, right_click)

    def _on_canvas_select(self, event):
        """Selects an organism when clicking the canvas while tracking is enabled."""
        organism_tracker.on_canvas_select(self, event)

    def _margin_contours(self, data, margin):
        """Returns cached contours expanded by a margin for overlay drawing."""
        return organism_tracker._margin_contours(self, data, margin)

    def save_settings(self):
        """Saves the full simulation configuration to disk."""
        file_handler.save_settings(self)

    def load_settings(self):
        """Loads simulation settings from disk and refreshes kernels."""
        file_handler.load_settings(self)
        self.update_all_kernels()

    def save_preset(self):
        """Persists the currently selected organism as a preset."""
        file_handler.save_preset(self)

    def load_preset(self):
        """Loads a preset or saved settings and rebuilds kernels."""
        file_handler.load_preset(self)
        self.update_all_kernels()

    def rename_preset(self):
        """Renames a user-created preset entry."""
        file_handler.rename_preset(self)

    def delete_preset(self):
        """Deletes a user-created preset entry."""
        file_handler.delete_preset(self)

    def _update_preset_listbox(self):
        """Refreshes the preset list UI."""
        file_handler.update_preset_listbox(self)

    def _update_preset_preview(self, event=None):
        """Updates the preview panel when the preset selection changes."""
        file_handler.update_preset_preview(self, event)

    def record_gif(self):
        """Toggles full-board GIF recording."""
        recording_manager.record_gif(self)

    def record_organism_stats(self):
        """Starts or stops organism stat logging and GIF capture."""
        recording_manager.record_organism_stats(self)

    def _update_slider_val(self, l, i, a, v, isl, isi):
        """Forwards slider change events to the shared handler."""
        event_handlers.update_slider_val(self, l, i, a, v, isl, isi)

    def _update_interaction(self, i, j, v):
        """Updates an entry in the interaction matrix via handler."""
        event_handlers.update_interaction(self, i, j, v)

    def _update_brush_size(self, v):
        """Adjusts the drawing brush size."""
        event_handlers.update_brush_size(self, v)

    def _on_draw_channel_selected(self, e):
        """Handles selection changes for the drawing channel dropdown."""
        event_handlers.on_draw_channel_selected(self, e)

    def add_channel(self):
        """Adds a new channel and rebuilds kernel cache."""
        event_handlers.add_channel(self)
        self.update_all_kernels()

    def delete_channel(self, id):
        """Deletes a channel and updates dependent state."""
        event_handlers.delete_channel(self, id)

    def duplicate_channel(self, id):
        """Duplicates a channel and refreshes kernels."""
        event_handlers.duplicate_channel(self, id)
        self.update_all_kernels()

    def add_kernel_layer(self, id):
        """Adds a kernel layer to the given channel."""
        event_handlers.add_kernel_layer(self, id)

    def clear_kernel_layers(self, id):
        """Resets kernel layers on a channel to a single default layer."""
        event_handlers.clear_kernel_layers(self, id)

    def remove_layer(self, id):
        """Removes a specific kernel layer from its channel."""
        event_handlers.remove_layer(self, id)

    def move_layer(self, id, d):
        """Moves a kernel layer up or down within the stack."""
        event_handlers.move_layer(self, id, d)

    def reset_seed(self):
        """Resets the board to the initial seeded circle pattern."""
        event_handlers.reset_seed(self)

    def randomize_board(self):
        """Randomizes the entire board contents."""
        event_handlers.randomize_board(self)

    def clear_board(self):
        """Clears the board to zeros."""
        event_handlers.clear_board(self)

    def toggle_pause(self):
        """Pauses or resumes simulation updates."""
        event_handlers.toggle_pause(self)

    def toggle_zoom_view(self):
        """Switches between zoomed and full-board views."""
        event_handlers.toggle_zoom_view(self)

    def _on_view_mode_selected(self, e=None):
        """Updates view mode selection for rendering."""
        event_handlers.on_view_mode_selected(self, e)

    def toggle_split_screen(self, initial=False):
        """Enables or disables split-screen visualization."""
        event_handlers.toggle_split_screen(self, initial=initial)

    def _toggle_local_params(self, id, en, w):
        """Turns local parameter maps on or off for a channel."""
        event_handlers.toggle_local_params(self, id, en, w)

    def _toggle_flow_responsive_params(self, id, en, w):
        """Toggles flow-responsiveness for a channel's local params."""
        event_handlers.toggle_flow_responsive_params(self, id, en, w)

    def _toggle_channel_active(self, id, var):
        """Marks a channel active or inactive and refreshes UI."""
        event_handlers.toggle_channel_active(self, id, var)

    def _toggle_layer_active(self, id, var):
        """Marks a kernel layer active or inactive."""
        event_handlers.toggle_layer_active(self, id, var)

    def _on_param_draw_target_selected(self, e=None):
        """Updates parameter drawing controls when target changes."""
        event_handlers.on_param_draw_target_selected(self, e)

    def _update_param_draw_slider(self, e=None):
        """Refreshes parameter draw slider label on movement."""
        event_handlers.update_param_draw_slider(self, e)
    
    # --- ADDED DELEGATE METHODS ---
    def _update_channel_attr(self, id, attr, val):
        """Updates a channel attribute from UI inputs."""
        event_handlers.update_channel_attr(self, id, attr, val)

    def _update_layer_attr(self, id, attr, val):
        """Updates a kernel layer attribute from UI inputs."""
        event_handlers.update_layer_attr(self, id, attr, val)

    # --- Helper & State Methods ---
    def _ensure_dir(self, dir_path):
        """Creates a directory if it does not already exist."""
        if not os.path.exists(dir_path): os.makedirs(dir_path)

    def _get_selem(self, r):
        """Returns a cached circular structuring element of radius r for morphology ops."""
        r = int(max(0, r))
        if r == 0: return np.ones((1, 1), dtype=bool)
        if r in self._selem_cache: return self._selem_cache[r]
        y, x = np.ogrid[-r:r + 1, -r:r + 1]
        self._selem_cache[r] = (x * x + y * y) <= r * r
        return self._selem_cache[r]

    def _initialize_board_circle_seed(self):
        """Initializes the board with a centered circular seed in the first channel."""
        board = torch.zeros((len(self.sim_state.channels), *GRID_DIM), dtype=torch.float32, device=device)
        if len(self.sim_state.channels) > 0:
            cy, cx = GRID_DIM[0] // 2, GRID_DIM[1] // 2
            y, x = np.ogrid[-cy:GRID_DIM[0] - cy, -cx:GRID_DIM[1] - cx]
            mask = torch.from_numpy((x * x + y * y) <= 25**2).to(device)
            board[0, mask] = 1.0
        return board
    
    def _randomize_board(self):
        """Generates a random board tensor matching the current channel count."""
        return torch.rand((len(self.sim_state.channels), *GRID_DIM), dtype=torch.float32, device=device)

    def _clear_board(self):
        """Returns an empty board tensor for the current channel count."""
        return torch.zeros((len(self.sim_state.channels), *GRID_DIM), dtype=torch.float32, device=device)
    
    def _get_channel_by_id(self, id):
        """Finds a channel by its id."""
        return next((c for c in self.sim_state.channels if c.id == id), None)

    def _get_layer_by_id(self, id):
        """Finds a kernel layer by its id across all channels."""
        return next((l for c in self.sim_state.channels for l in c.kernel_layers if l['id'] == id), None)

    def _get_channel_by_any_id(self, id):
        """Resolves an id that may refer to either a channel or one of its layers."""
        return self._get_channel_by_id(id) or next((c for c in self.sim_state.channels for l in c.kernel_layers if l['id'] == id), None)

    def _initialize_local_param_maps(self, channel):
        """Initializes local parameter maps for a channel that has them enabled."""
        local_param_maps[channel.id] = {
            name: torch.full(GRID_DIM, getattr(channel, name), dtype=torch.float32, device=device)
            for name in LOCAL_PARAM_NAMES
        }

    def _initialize_all_local_param_maps(self):
        """Prepares local parameter maps for all channels that require them."""
        local_param_maps.clear()
        for ch in self.sim_state.channels:
            if ch.has_local_params:
                self._initialize_local_param_maps(ch)

    # --- UI State Update Methods ---
    def _set_widget_state(self, parent, state, exceptions=[]):
        """Recursively sets widget state for a container, respecting exceptions."""
        for child in parent.winfo_children():
            if child in exceptions: continue
            if 'state' in child.configure(): child.configure(state=state)
            self._set_widget_state(child, state, exceptions)
            
    def _set_channel_ui_state(self, ch, widgets):
        """Enables sliders for a channel (other state managed elsewhere)."""
        for slider_frame in widgets.get('sliders', {}).values():
            self._set_widget_state(slider_frame, 'normal')

    def _update_local_param_draw_ui(self):
        """Hides parameter drawing controls when local parameters are disabled."""
        # Local parameter drawing is disabled; keep controls hidden.
        self.param_draw_frame.pack_forget()
        self.param_draw_target.set("Mass")

    def _update_draw_channel_selector(self):
        """Keeps the draw channel dropdown in sync with current channel count."""
        num_ch = len(self.sim_state.channels)
        values = ["All"] + [str(i + 1) for i in range(num_ch)] if num_ch > 0 else []
        self.draw_channel_dd['values'] = values
        if num_ch == 0:
            self.draw_channel_index = -1
            self.draw_channel_var.set("")
            return
        if self.draw_channel_index >= num_ch or self.draw_channel_index < -1:
            self.draw_channel_index = -1
        self.draw_channel_var.set("All" if self.draw_channel_index == -1 else str(self.draw_channel_index + 1))

    def _update_vis_options(self):
        """Populates visualization dropdowns based on available channels."""
        options = ["Final Board"]
        for i, ch in enumerate(self.sim_state.channels):
            options.extend([f"Ch {i+1}: Potential Field", f"Ch {i+1}: Growth Field", f"Ch {i+1}: Flow Field"])
        for widget in [self.vis_dd, *self.split_view_controls]:
            widget['values'] = options
        if self.view_mode_var.get() not in options: self.view_mode_var.set("Final Board")
        for var in self.split_view_vars:
            if var.get() not in options: var.set("Final Board")
            
if __name__ == '__main__':
    root = tk.Tk()
    app = LeniaApp(root)
    root.mainloop()
