import tkinter as tk
from tkinter import ttk
from simulation import kernel_functions, growth_functions
from config import UI_PANEL_WIDTH, KERNEL_SIZE

def _destroy_children(widget):
    for child in widget.winfo_children():
        child.destroy()

def _build_ui(app):
    _destroy_children(app.ui_frame)
    notebook = ttk.Notebook(app.ui_frame)
    notebook.pack(fill=tk.BOTH, expand=True)
    tab1 = ttk.Frame(notebook)
    tab2 = ttk.Frame(notebook)
    notebook.add(tab1, text="Channels & Kernels")
    notebook.add(tab2, text="Analysis & Tracking")
    _build_channels_tab(app, tab1)
    _build_analysis_tab(app, tab2)
    app._update_local_param_draw_ui()

def _build_channels_tab(app, parent):
    top_frame = ttk.Frame(parent)
    interactions_container = ttk.Frame(parent)
    channel_container = ttk.Frame(parent)
    bottom_frame = ttk.Frame(parent)
    top_frame.pack(fill=tk.X, pady=5, side=tk.TOP)
    interactions_container.pack(fill=tk.X, pady=5, side=tk.TOP)
    bottom_frame.pack(fill=tk.X, pady=5, side=tk.BOTTOM)
    channel_container.pack(fill=tk.BOTH, expand=True, side=tk.TOP)
    
    ttk.Button(top_frame, text="Add New Channel", command=app.add_channel).pack(fill=tk.X)
    app.interactions_frame = ttk.LabelFrame(interactions_container, text="Channel Interactions (J -> I)")
    app.interactions_frame.pack(fill=tk.X)
    _rebuild_interactions_ui(app)

    scroll_canvas = tk.Canvas(channel_container, highlightthickness=0)
    scrollbar = ttk.Scrollbar(channel_container, orient="vertical", command=scroll_canvas.yview)
    app.scrollable_frame = ttk.Frame(scroll_canvas)
    app.scrollable_frame.bind("<Configure>", lambda e: scroll_canvas.configure(scrollregion=scroll_canvas.bbox("all")))
    scroll_canvas.create_window((0, 0), window=app.scrollable_frame, anchor="nw", width=UI_PANEL_WIDTH - 25)
    scroll_canvas.configure(yscrollcommand=scrollbar.set)
    scroll_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    _rebuild_channels_ui(app)

    sim_controls = ttk.Frame(bottom_frame)
    sim_controls.pack(fill=tk.X, pady=5)
    ttk.Button(sim_controls, text="Reset Seed", command=app.reset_seed).pack(side=tk.LEFT, expand=True, fill=tk.X)
    ttk.Button(sim_controls, text="Randomize", command=app.randomize_board).pack(side=tk.LEFT, expand=True, fill=tk.X)
    ttk.Button(sim_controls, text="Clear", command=app.clear_board).pack(side=tk.LEFT, expand=True, fill=tk.X)
    
    draw_controls_frame = ttk.LabelFrame(bottom_frame, text="Drawing Controls")
    draw_controls_frame.pack(fill=tk.X, pady=(5,0), padx=2)
    
    draw_row1 = ttk.Frame(draw_controls_frame)
    draw_row1.pack(fill=tk.X, pady=2)
    ttk.Label(draw_row1, text="Draw Ch:").pack(side=tk.LEFT, padx=5)
    app.draw_channel_var = tk.StringVar()
    app.draw_channel_dd = ttk.Combobox(draw_row1, textvariable=app.draw_channel_var, state="readonly", width=3)
    app.draw_channel_dd.bind("<<ComboboxSelected>>", app._on_draw_channel_selected)
    app.draw_channel_dd.pack(side=tk.LEFT)
    app._update_draw_channel_selector()
    ttk.Label(draw_row1, text="Brush Size:").pack(side=tk.LEFT, padx=(10, 0))
    app.brush_size_var = tk.IntVar(value=app.draw_brush_size)
    ttk.Scale(draw_row1, from_=1, to=50, orient=tk.HORIZONTAL, variable=app.brush_size_var, command=app._update_brush_size).pack(side=tk.LEFT, fill=tk.X, expand=True)

    app.param_draw_frame = ttk.Frame(draw_controls_frame)
    app.param_draw_frame.pack(fill=tk.X, pady=2)
    ttk.Label(app.param_draw_frame, text="Draw Target:").pack(side=tk.LEFT, padx=5)
    app.param_draw_dd = ttk.Combobox(app.param_draw_frame, textvariable=app.param_draw_target, state="readonly", width=12, values=["Mass"])
    app.param_draw_dd.bind("<<ComboboxSelected>>", app._on_param_draw_target_selected)
    app.param_draw_dd.pack(side=tk.LEFT)
    app.param_draw_val_label = ttk.Label(app.param_draw_frame, text="Val: 0.50", width=8)
    app.param_draw_val_slider = ttk.Scale(app.param_draw_frame, from_=0, to=1, orient=tk.HORIZONTAL, variable=app.param_draw_value, command=app._update_param_draw_slider)
    app.param_draw_val_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
    app.param_draw_val_label.pack(side=tk.LEFT, padx=(5,0))
    
    app.record_button = ttk.Button(bottom_frame, text="Record Full GIF", command=app.record_gif)
    app.record_button.pack(fill=tk.X, pady=2)
    file_controls = ttk.Frame(bottom_frame)
    file_controls.pack(fill=tk.X)
    ttk.Button(file_controls, text="Save Settings", command=app.save_settings).pack(side=tk.LEFT, expand=True, fill=tk.X)
    ttk.Button(file_controls, text="Load Settings", command=app.load_settings).pack(side=tk.LEFT, expand=True, fill=tk.X)

def _build_analysis_tab(app, parent):
    tracking_frame = ttk.LabelFrame(parent, text="Organism Tracking")
    tracking_frame.pack(fill=tk.X, pady=5)
    top_track_frame = ttk.Frame(tracking_frame)
    top_track_frame.pack(fill=tk.X)
    ttk.Checkbutton(top_track_frame, text="Enable Tracking", variable=app.tracking_enabled).pack(side=tk.LEFT)
    ttk.Label(top_track_frame, text="Min Mass:").pack(side=tk.LEFT, padx=(10,2))
    ttk.Spinbox(top_track_frame, from_=5, to=1000, textvariable=app.min_organism_mass, width=5).pack(side=tk.LEFT)
    ttk.Label(top_track_frame, textvariable=app.organism_count_var).pack(side=tk.RIGHT, padx=10)

    watershed_frame = ttk.Frame(tracking_frame)
    watershed_frame.pack(fill=tk.X, padx=5, pady=(5,0))
    ttk.Checkbutton(watershed_frame, text="Use Watershed Segmentation", variable=app.use_watershed).pack(side=tk.LEFT)
    ttk.Label(watershed_frame, text="Peak Dist:").pack(side=tk.LEFT, padx=(10,2))
    ttk.Spinbox(watershed_frame, from_=2, to=20, textvariable=app.watershed_peak_distance, width=4).pack(side=tk.LEFT)

    vis_frame = ttk.Frame(tracking_frame)
    vis_frame.pack(fill=tk.X, padx=10)
    ttk.Checkbutton(vis_frame, text="Outlines", variable=app.show_outlines).pack(side=tk.LEFT, expand=True)
    ttk.Checkbutton(vis_frame, text="CoM", variable=app.show_com).pack(side=tk.LEFT, expand=True)
    ttk.Checkbutton(vis_frame, text="Direction", variable=app.show_direction).pack(side=tk.LEFT, expand=True)
    
    indicator_frame = ttk.LabelFrame(tracking_frame, text="Indicator Controls")
    indicator_frame.pack(fill=tk.X, padx=5, pady=(5,0))
    ttk.Label(indicator_frame, text="Vel. Smooth:").grid(row=0, column=0, sticky="w", padx=2)
    ttk.Scale(indicator_frame, from_=0.01, to=0.99, orient=tk.HORIZONTAL, variable=app.velocity_smoothing_factor).grid(row=0, column=1, sticky="ew")
    ttk.Label(indicator_frame, text="Dir. Smooth:").grid(row=1, column=0, sticky="w", padx=2)
    ttk.Scale(indicator_frame, from_=0.01, to=0.99, orient=tk.HORIZONTAL, variable=app.direction_smoothing_factor).grid(row=1, column=1, sticky="ew")
    ttk.Label(indicator_frame, text="Vel. Sense:").grid(row=2, column=0, sticky="w", padx=2)
    ttk.Scale(indicator_frame, from_=1.0, to=20.0, orient=tk.HORIZONTAL, variable=app.velocity_sensitivity).grid(row=2, column=1, sticky="ew")
    ttk.Label(indicator_frame, text="Out. Margin:").grid(row=3, column=0, sticky="w", padx=2)
    ttk.Scale(indicator_frame, from_=0, to=10, orient=tk.HORIZONTAL, variable=app.outline_margin).grid(row=3, column=1, sticky="ew")
    indicator_frame.grid_columnconfigure(1, weight=1)

    vis_select_frame = ttk.LabelFrame(parent, text="Canvas Visualization")
    vis_select_frame.pack(fill=tk.X, pady=5)
    
    app.split_screen_button = ttk.Button(vis_select_frame, text="Enable Split Screen", command=app.toggle_split_screen)
    app.split_screen_button.pack(fill=tk.X, pady=(0,5))
    
    app.single_view_frame = ttk.Frame(vis_select_frame)
    app.single_view_frame.pack(fill=tk.X)
    app.vis_dd = ttk.Combobox(app.single_view_frame, textvariable=app.view_mode_var, state="readonly")
    app.vis_dd.pack(fill=tk.X)
    app.vis_dd.bind("<<ComboboxSelected>>", app._on_view_mode_selected)
    
    app.split_view_frame = ttk.Frame(vis_select_frame)
    app.split_view_controls = []
    for i in range(4):
        row = i // 2; col = i % 2
        frame = ttk.Frame(app.split_view_frame)
        frame.grid(row=row, column=col, sticky="ew", padx=2, pady=2)
        dd = ttk.Combobox(frame, textvariable=app.split_view_vars[i], state="readonly")
        dd.pack(fill=tk.X, expand=True)
        app.split_view_controls.append(dd)
    app.split_view_frame.grid_columnconfigure((0,1), weight=1)

    app._update_vis_options()
    app.toggle_split_screen(initial=True)
    
    app.zoom_button = ttk.Button(vis_select_frame, text="Switch to Zoom View", command=app.toggle_zoom_view)
    app.zoom_button.pack(fill=tk.X, pady=(5,0))

    stats_frame = ttk.LabelFrame(parent, text="Selected Organism Stats")
    stats_frame.pack(fill=tk.X, pady=5)
    app.stats_mass_label = ttk.Label(stats_frame, text="Mass (per ch): N/A", wraplength=UI_PANEL_WIDTH-20); app.stats_mass_label.pack(anchor="w")
    app.stats_vel_label = ttk.Label(stats_frame, text="Velocity: N/A"); app.stats_vel_label.pack(anchor="w")
    app.stats_dir_label = ttk.Label(stats_frame, text="Direction: N/A"); app.stats_dir_label.pack(anchor="w")
    app.stats_record_button = ttk.Button(stats_frame, text="Record Organism Stats", command=app.record_organism_stats)
    app.stats_record_button.pack(fill=tk.X, pady=(5,0))

    preset_frame = ttk.LabelFrame(parent, text="Presets & Saved Items"); preset_frame.pack(fill=tk.BOTH, expand=True, pady=5)
    preset_list_frame = ttk.Frame(preset_frame); preset_list_frame.pack(fill=tk.BOTH, expand=True)
    app.preset_listbox = tk.Listbox(preset_list_frame); app.preset_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    app.preset_listbox.bind("<<ListboxSelect>>", app._update_preset_preview)
    preset_scroll = ttk.Scrollbar(preset_list_frame, orient="vertical", command=app.preset_listbox.yview)
    preset_scroll.pack(side=tk.RIGHT, fill=tk.Y); app.preset_listbox.config(yscrollcommand=preset_scroll.set)
    ttk.Label(preset_frame, text="[Preset] built-in  |  [Saved Organism] your captures  |  [Saved Settings] full configs", wraplength=UI_PANEL_WIDTH-40, justify="left").pack(fill=tk.X, padx=5, pady=(2,0))
    
    preview_frame = ttk.LabelFrame(preset_frame, text="Preview"); preview_frame.pack(fill=tk.X, pady=(5,0))
    app.preset_preview_label = ttk.Label(preview_frame); app.preset_preview_label.pack(pady=5, padx=5)
    app.preset_preview_photo = None

    preset_btn_frame = ttk.Frame(preset_frame); preset_btn_frame.pack(fill=tk.X, pady=(5,0))
    ttk.Button(preset_btn_frame, text="Save Selected", command=app.save_preset).pack(side=tk.LEFT, expand=True, fill=tk.X)
    ttk.Button(preset_btn_frame, text="Load", command=app.load_preset).pack(side=tk.LEFT, expand=True, fill=tk.X)
    ttk.Button(preset_btn_frame, text="Rename", command=app.rename_preset).pack(side=tk.LEFT, expand=True, fill=tk.X)
    ttk.Button(preset_btn_frame, text="Delete", command=app.delete_preset).pack(side=tk.LEFT, expand=True, fill=tk.X)
    app._update_preset_listbox()

def _rebuild_interactions_ui(app):
    _destroy_children(app.interactions_frame)
    num_ch = len(app.sim_state.channels)
    if num_ch == 0: return
    for i in range(num_ch):
        app.interactions_frame.grid_columnconfigure(i, weight=1)
        for j in range(num_ch):
            frame = ttk.Frame(app.interactions_frame); frame.grid(row=i, column=j, padx=2, pady=2, sticky="ew")
            label = ttk.Label(frame, text=f'{j+1}->{i+1}')
            label.pack()
            scale = ttk.Scale(frame, from_=0.0, to=1.5, orient=tk.HORIZONTAL, value=app.sim_state.interaction_matrix[i][j], command=lambda val,i=i,j=j: app._update_interaction(i,j,val))
            scale.pack(fill=tk.X, expand=True)
            if not app.sim_state.channels[i].is_active or not app.sim_state.channels[j].is_active:
                label.config(state='disabled'); scale.config(state='disabled')

def _rebuild_channels_ui(app):
    _destroy_children(app.scrollable_frame)
    app.kernel_previews = {}
    for i, ch in enumerate(app.sim_state.channels):
        ch_frame = ttk.LabelFrame(app.scrollable_frame, text=f"Channel {i+1}")
        ch_frame.pack(fill=tk.X, expand=True, padx=5, pady=5)
        _build_single_channel_ui(app, ch_frame, ch)

def _build_single_channel_ui(app, parent_frame, ch):
    widgets = {}
    top_row = ttk.Frame(parent_frame); top_row.pack(fill=tk.X, pady=2)
    ch_active_var = tk.BooleanVar(value=ch.is_active)
    widgets['ch_check'] = ttk.Checkbutton(top_row, text="Active", variable=ch_active_var, command=lambda id=ch.id, v=ch_active_var: app._toggle_channel_active(id, v))
    widgets['ch_check'].pack(side=tk.LEFT)
    controls_frame = ttk.Frame(top_row); controls_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
    ttk.Label(controls_frame, text="Color:").pack(side=tk.LEFT)
    color_var = tk.StringVar(value=ch.color_hex); color_entry = ttk.Entry(controls_frame, textvariable=color_var, width=10); color_entry.pack(side=tk.LEFT, padx=5)
    color_entry.bind("<FocusOut>", lambda e, id=ch.id, v=color_var: app._update_channel_attr(id, 'color_hex', v.get()))
    ttk.Button(controls_frame, text="Duplicate", command=lambda id=ch.id: app.duplicate_channel(id)).pack(side=tk.RIGHT, padx=2)
    ttk.Button(controls_frame, text="Delete", command=lambda id=ch.id: app.delete_channel(id)).pack(side=tk.RIGHT)

    sim_mode_frame = ttk.Frame(parent_frame); sim_mode_frame.pack(fill=tk.X, padx=5, pady=2)
    flow_active_var = tk.BooleanVar(value=ch.use_flow)
    widgets['flow_check'] = ttk.Checkbutton(sim_mode_frame, text="Use Flow", variable=flow_active_var, command=lambda id=ch.id, v=flow_active_var: app._update_channel_attr(id, 'use_flow', v.get()))
    widgets['flow_check'].pack(side=tk.LEFT)
    
    widgets['sliders'] = {}
    param_list = [("Mu", 'mu', 0.0, 1.0), ("Sigma", 'sigma', 0.001, 0.2), ("DT", 'dt', 0.01, 0.2), ("Flow", 'flow_strength', 0.0, 20.0)]
    for label, attr, min_val, max_val in param_list:
        widgets['sliders'][attr] = _create_slider(app, parent_frame, label, ch.id, attr, min_val, max_val)

    growth_frame = ttk.Frame(parent_frame); growth_frame.pack(fill=tk.X, padx=5, pady=2); ttk.Label(growth_frame, text="Growth Func:", width=10).pack(side=tk.LEFT)
    growth_dd = ttk.Combobox(growth_frame, values=list(growth_functions.keys()), state="readonly"); growth_dd.set(ch.growth_func_name)
    growth_dd.bind("<<ComboboxSelected>>", lambda e, id=ch.id: app._update_channel_attr(id, 'growth_func_name', e.widget.get())); growth_dd.pack(side=tk.LEFT, fill=tk.X, expand=True)
    
    kb_frame = ttk.LabelFrame(parent_frame, text="Kernel Builder"); kb_frame.pack(fill=tk.X, expand=True, pady=5, padx=5)
    preview_label = ttk.Label(kb_frame); preview_label.pack(side=tk.RIGHT, padx=5); app.kernel_previews[ch.id] = preview_label
    kernel_controls_frame = ttk.Frame(kb_frame); kernel_controls_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
    kb_buttons_frame = ttk.Frame(kernel_controls_frame); kb_buttons_frame.pack(fill=tk.X)
    ttk.Button(kb_buttons_frame, text="Add Layer", command=lambda id=ch.id: app.add_kernel_layer(id)).pack(side=tk.LEFT, expand=True, fill=tk.X)
    ttk.Button(kb_buttons_frame, text="Clear Layers", command=lambda id=ch.id: app.clear_kernel_layers(id)).pack(side=tk.LEFT, expand=True, fill=tk.X)
    
    for k_idx, layer in enumerate(ch.kernel_layers): _build_kernel_layer_ui(app, kernel_controls_frame, ch.id, layer, k_idx, len(ch.kernel_layers))
    app._update_kernel_preview(ch.id)
    
    app._set_channel_ui_state(ch, widgets)
    if not ch.is_active: app._set_widget_state(parent_frame, 'disabled', exceptions=[widgets['ch_check']])

def _build_flow_responsive_ui(app, parent, ch):
    frame = ttk.LabelFrame(parent, text="Flow Response Settings")
    _create_slider(app, frame, "Sensitivity", ch.id, 'flow_sensitivity', 1.0, 50.0)
    _create_range_slider(app, frame, "Mu Range", ch.id, 'mu_range', 0.0, 1.0)
    _create_range_slider(app, frame, "Sigma Range", ch.id, 'sigma_range', 0.001, 0.2)
    _create_range_slider(app, frame, "DT Range", ch.id, 'dt_range', 0.01, 0.2)
    _create_range_slider(app, frame, "Flow Str Range", ch.id, 'flow_strength_range', 0.0, 20.0)
    return frame

def _create_range_slider(app, parent, label_text, ch_id, attr, from_, to_):
    frame = ttk.Frame(parent); frame.pack(fill=tk.X, padx=5, pady=2)
    ttk.Label(frame, text=label_text, width=12).pack(side=tk.LEFT)
    ch = app._get_channel_by_id(ch_id)
    current_range = getattr(ch, attr, [from_, to_])
    min_var = tk.DoubleVar(value=current_range[0])
    max_var = tk.DoubleVar(value=current_range[1])
    label = ttk.Label(frame, text=f"[{current_range[0]:.2f}, {current_range[1]:.2f}]", width=12)
    def _update(v=None):
        min_val = min_var.get(); max_val = max_var.get()
        if min_val > max_val: min_var.set(max_val); min_val = max_val
        label.config(text=f"[{min_val:.2f}, {max_val:.2f}]")
        app._update_channel_attr(ch_id, attr, [min_val, max_val])
    min_scale = ttk.Scale(frame, from_=from_, to=to_, orient=tk.HORIZONTAL, variable=min_var, command=_update)
    max_scale = ttk.Scale(frame, from_=from_, to=to_, orient=tk.HORIZONTAL, variable=max_var, command=_update)
    min_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
    max_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
    label.pack(side=tk.LEFT, padx=(5,0))
    return frame
        
def _build_kernel_layer_ui(app, parent, ch_id, layer, k_idx, total_layers):
    layer_frame = ttk.Frame(parent, borderwidth=1, relief="solid"); layer_frame.pack(fill=tk.X, pady=3, padx=2)
    row1 = ttk.Frame(layer_frame); row1.pack(fill=tk.X)
    layer_active_var = tk.BooleanVar(value=layer.get('is_active', True))
    layer_check = ttk.Checkbutton(row1, text="Active", variable=layer_active_var, command=lambda l_id=layer['id'], v=layer_active_var: app._toggle_layer_active(l_id, v))
    layer_check.pack(side=tk.LEFT)
    controls_frame = ttk.Frame(row1); controls_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
    if k_idx > 0:
        op_dd = ttk.Combobox(controls_frame, values=['+', '-', '*'], state="readonly", width=3); op_dd.set(layer.get('op', '+'))
        op_dd.bind("<<ComboboxSelected>>", lambda e,l_id=layer['id']: app._update_layer_attr(l_id,'op',e.widget.get())); op_dd.pack(side=tk.LEFT)
    else: ttk.Label(controls_frame, text="Base:").pack(side=tk.LEFT)
    type_dd = ttk.Combobox(controls_frame, values=list(kernel_functions.keys()), state="readonly"); type_dd.set(layer['type'])
    type_dd.bind("<<ComboboxSelected>>", lambda e,l_id=layer['id']: app._update_layer_attr(l_id,'type',e.widget.get())); type_dd.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
    btn_frame = ttk.Frame(controls_frame); btn_frame.pack(side=tk.LEFT)
    up_btn = ttk.Button(btn_frame, text="Up", width=3, command=lambda l_id=layer['id']: app.move_layer(l_id, -1)); up_btn.pack(side=tk.LEFT); up_btn['state'] = 'disabled' if k_idx==0 else 'normal'
    down_btn = ttk.Button(btn_frame, text="Dn", width=3, command=lambda l_id=layer['id']: app.move_layer(l_id, 1)); down_btn.pack(side=tk.LEFT); down_btn['state'] = 'disabled' if k_idx==total_layers-1 else 'normal'
    ttk.Button(btn_frame, text="Del", width=4, command=lambda l_id=layer['id']: app.remove_layer(l_id)).pack(side=tk.LEFT)
    _create_slider(app, layer_frame, "Radius", layer['id'], 'radius', 1, KERNEL_SIZE//2, is_layer=True, is_int=True)
    _create_slider(app, layer_frame, "Weight", layer['id'], 'weight', -2.0, 2.0, is_layer=True)
    if not layer.get('is_active', True):
        app._set_widget_state(layer_frame, 'disabled', exceptions=[layer_check])

def _create_slider(app, parent, label_text, item_id, attr, from_, to, is_layer=False, is_int=False):
    frame = ttk.Frame(parent); frame.pack(fill=tk.X, padx=5, pady=2)
    ttk.Label(frame, text=label_text, width=12).pack(side=tk.LEFT)
    if is_layer: item=app._get_layer_by_id(item_id); val=item.get(attr)
    else: item=app._get_channel_by_id(item_id); val=getattr(item, attr)
    if val is None: val=0
    var=tk.DoubleVar(value=val); label=ttk.Label(frame, text=f"{val:.2f}" if not is_int else f"{int(val)}", width=5)
    scale=ttk.Scale(frame, from_=from_, to=to, orient=tk.HORIZONTAL, variable=var, command=lambda v,lbl=label,id=item_id,a=attr: app._update_slider_val(lbl,id,a,v,is_layer,is_int))
    scale.pack(side=tk.LEFT, fill=tk.X, expand=True); label.pack(side=tk.LEFT)
    return frame
