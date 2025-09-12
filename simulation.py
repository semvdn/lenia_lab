import torch
import uuid
from config import device, KERNEL_SIZE, GRID_DIM

# --- DATA STRUCTURES & SIMULATION CORE ---
class Channel:
    """
    Represents a single channel in the simulation, holding all its parameters.

    Attributes:
        id (str): A unique identifier for the channel.
        color_hex (str): The color of the channel in hexadecimal format.
        mu (float): The center of the growth function.
        sigma (float): The width of the growth function.
        dt (float): The time step for the simulation update.
        flow_strength (float): The strength of the flow field.
        use_flow (bool): Whether to use the flow field for advection.
        has_local_params (bool): Whether the channel uses local parameter maps.
        flow_responsive_params (bool): Whether local parameters respond to flow magnitude.
        flow_sensitivity (float): The sensitivity of local parameters to the flow field.
        mu_range (list[float]): The range [min, max] for mu when using flow-responsive parameters.
        sigma_range (list[float]): The range [min, max] for sigma when using flow-responsive parameters.
        dt_range (list[float]): The range [min, max] for dt when using flow-responsive parameters.
        flow_strength_range (list[float]): The range [min, max] for flow_strength when using flow-responsive parameters.
        growth_func_name (str): The name of the growth function to use.
        kernel_layers (list[dict]): A list of dictionaries, each defining a layer of the composite kernel.
        is_active (bool): Whether the channel is currently active in the simulation.
    """
    def __init__(self, **kwargs):
        self.id = str(uuid.uuid4())
        self.color_hex = '#FF00FF'
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
        self.__dict__.update(kwargs)

class SimulationState:
    """
    A container for all simulation parameters.

    Attributes:
        channels (list[Channel]): A list of Channel objects used in the simulation.
        interaction_matrix (list[list[float]]): A matrix defining the interaction weights between channels.
        view_mode (str): The current view mode for rendering (e.g., "Final Board").
        active_channel_idx_for_view (int): The index of the channel to be displayed in certain view modes.
    """
    def __init__(self):
        self.channels = [Channel(color_hex='#00FFFF')]
        self.interaction_matrix = [[1.0]]
        self.view_mode = "Final Board"
        self.active_channel_idx_for_view = 0

# Global dictionary to store local parameter maps for channels that have them enabled.
local_param_maps = {}
# Names of parameters that can be localized.
LOCAL_PARAM_NAMES = ['mu', 'sigma', 'dt', 'flow_strength']

def kernel_gaussian_ring(r, ks):
    """
    Generates a Gaussian ring kernel.

    Args:
        r (int): The radius of the ring.
        ks (int): The size of the kernel.

    Returns:
        torch.Tensor: A 2D tensor representing the Gaussian ring kernel.
    """
    x, y = torch.meshgrid(torch.arange(ks, device=device), torch.arange(ks, device=device), indexing='ij'); c=ks//2
    d = torch.sqrt((x-c)**2+(y-c)**2); s = torch.exp(-((d-r)**2)/(2*(max(1,r)/3)**2)); return s*(d>0)

def kernel_filled_gaussian(r, ks):
    """
    Generates a filled Gaussian kernel (a Gaussian blob).

    Args:
        r (int): The radius (standard deviation) of the Gaussian.
        ks (int): The size of the kernel.

    Returns:
        torch.Tensor: A 2D tensor representing the filled Gaussian kernel.
    """
    x, y = torch.meshgrid(torch.arange(ks, device=device), torch.arange(ks, device=device), indexing='ij'); c=ks//2
    d = torch.sqrt((x-c)**2+(y-c)**2); return torch.exp(-(d**2)/(2*(max(1,r)/2)**2))

def kernel_square(r, ks):
    """
    Generates a square-shaped kernel.

    Args:
        r (int): The half-width of the square.
        ks (int): The size of the kernel.

    Returns:
        torch.Tensor: A 2D tensor representing the square kernel.
    """
    x, y = torch.meshgrid(torch.arange(ks, device=device), torch.arange(ks, device=device), indexing='ij'); c=ks//2
    m = (torch.abs(x-c)<r)&(torch.abs(y-c)<r); return m.float()

kernel_functions = {"Gaussian Ring": kernel_gaussian_ring, "Filled Gaussian": kernel_filled_gaussian, "Square": kernel_square}

def growth_gaussian_bell(x, mu, sigma):
    """
    A Gaussian bell-shaped growth function.

    Args:
        x (torch.Tensor): The input potential field.
        mu (float or torch.Tensor): The center of the Gaussian function.
        sigma (float or torch.Tensor): The width of the Gaussian function.

    Returns:
        torch.Tensor: The growth value, ranging from -1 to 1.
    """
    clamped_sigma = torch.clamp(torch.as_tensor(sigma), min=1e-6)
    return torch.exp(-((x - mu)**2) / (2 * clamped_sigma**2)) * 2 - 1

def growth_step_function(x, mu, sigma):
    """
    A step growth function.

    Args:
        x (torch.Tensor): The input potential field.
        mu (float or torch.Tensor): The center of the step.
        sigma (float or torch.Tensor): The half-width of the step.

    Returns:
        torch.Tensor: The growth value, either -1 or 1.
    """
    return (((x>mu-sigma)&(x<mu+sigma)).float()*2-1)

growth_functions = { "Gaussian Bell": growth_gaussian_bell, "Step Function": growth_step_function }

def generate_composite_kernel(kernel_layers):
    """
    Generates a composite kernel by combining multiple kernel layers.

    Args:
        kernel_layers (list[dict]): A list of layer definitions. Each layer is a dictionary
                                     specifying 'type', 'radius', 'weight', 'op', and 'is_active'.

    Returns:
        torch.Tensor: A 2D tensor representing the final composite kernel.
    """
    active_layers = [l for l in kernel_layers if l.get('is_active', True)]
    if not active_layers: return torch.zeros(KERNEL_SIZE, KERNEL_SIZE, device=device)
    
    # Initialize with the first active layer
    k = kernel_functions.get(active_layers[0]['type'])(active_layers[0]['radius'], KERNEL_SIZE)
    if torch.sum(k)>0: k/=torch.sum(k) # Normalize
    k*=active_layers[0]['weight']
    
    # Combine subsequent layers
    for layer in active_layers[1:]:
        op=layer.get('op','+'); 
        sub_k=kernel_functions.get(layer['type'])(layer['radius'], KERNEL_SIZE)
        if torch.sum(sub_k)>0: sub_k/=torch.sum(sub_k) # Normalize
        sub_k*=layer['weight']
        
        if op=='+':k+=sub_k
        elif op=='-':k-=sub_k
        elif op=='*':k*=sub_k
            
    return k

def update_multichannel_board(game_board, sim_state, grid_y, grid_x, kernel_cache):
    """
    Updates the simulation state for one time step.

    This function calculates the potential and growth for each channel, applies advection based on
    the flow field, and updates the board state. It handles multiple channels and their interactions.

    Args:
        game_board (torch.Tensor): The current state of the simulation board (num_channels, height, width).
        sim_state (SimulationState): The container for all simulation parameters.
        grid_y (torch.Tensor): A meshgrid of y-coordinates for flow field calculations.
        grid_x (torch.Tensor): A meshgrid of x-coordinates for flow field calculations.
        kernel_cache (dict): A cache of pre-computed kernels and their gradients.

    Returns:
        dict[str, torch.Tensor]: A dictionary containing various simulation fields like 'potential',
                                 'growth', 'flow_y', 'flow_x', and the 'final_board'.
    """
    channels = sim_state.channels
    interaction_matrix = sim_state.interaction_matrix
    
    num_channels = len(channels)
    new_board = game_board.clone()
    sim_fields = {'potential':torch.zeros_like(game_board), 'growth':torch.zeros_like(game_board), 'flow_y':torch.zeros_like(game_board), 'flow_x':torch.zeros_like(game_board)}
    padding = KERNEL_SIZE//2
    
    active_indices = [i for i, ch in enumerate(channels) if ch.is_active]

    for i in active_indices:
        channel_params = channels[i]
        kernel, kernel_grad_y, kernel_grad_x = kernel_cache[channel_params.id]

        # Calculate the influenced slice based on the interaction matrix
        influenced_slice=torch.zeros((1,1,*GRID_DIM),device=device)
        for j in range(num_channels):
            if channels[j].is_active and (w:=interaction_matrix[i][j])!=0: 
                influenced_slice += game_board[j,:,:].unsqueeze(0).unsqueeze(0) * w
        influenced_slice=torch.clamp(influenced_slice,0,1)
        
        # Calculate potential field via convolution
        padded_board=torch.nn.functional.pad(influenced_slice,(padding,)*4,mode='circular')
        potential=torch.nn.functional.conv2d(padded_board, kernel.unsqueeze(0).unsqueeze(0)).squeeze()
        sim_fields['potential'][i]=potential

        # Calculate flow field from the gradient of the kernel
        original_slice=game_board[i,:,:].unsqueeze(0).unsqueeze(0)
        padded_original=torch.nn.functional.pad(original_slice,(padding,)*4,mode='circular')
        flow_y=-torch.nn.functional.conv2d(padded_original, kernel_grad_y.unsqueeze(0).unsqueeze(0)).squeeze()
        flow_x=-torch.nn.functional.conv2d(padded_original, kernel_grad_x.unsqueeze(0).unsqueeze(0)).squeeze()
        sim_fields['flow_y'][i],sim_fields['flow_x'][i]=flow_y,flow_x

        # Update local parameters based on flow magnitude if enabled
        if channel_params.has_local_params and channel_params.flow_responsive_params and channel_params.id in local_param_maps:
            p_maps = local_param_maps[channel_params.id]
            flow_mag = torch.sqrt(flow_x**2 + flow_y**2)
            response_factor = torch.clamp(flow_mag / (channel_params.flow_sensitivity + 1e-6), 0.0, 1.0)
            p_maps['mu'] = channel_params.mu_range[0] + response_factor * (channel_params.mu_range[1] - channel_params.mu_range[0])
            p_maps['sigma'] = channel_params.sigma_range[0] + response_factor * (channel_params.sigma_range[1] - channel_params.sigma_range[0])
            p_maps['dt'] = channel_params.dt_range[0] + response_factor * (channel_params.dt_range[1] - channel_params.dt_range[0])
            p_maps['flow_strength'] = channel_params.flow_strength_range[0] + response_factor * (channel_params.flow_strength_range[1] - channel_params.flow_strength_range[0])

        # Get parameters for growth calculation (either global or local)
        if channel_params.has_local_params and channel_params.id in local_param_maps:
            p_maps = local_param_maps[channel_params.id]
            mu, sigma, dt, flow_strength = p_maps['mu'], p_maps['sigma'], p_maps['dt'], p_maps['flow_strength']
        else:
            mu, sigma, dt, flow_strength = channel_params.mu, channel_params.sigma, channel_params.dt, channel_params.flow_strength
        
        # Calculate growth
        growth=growth_functions.get(channel_params.growth_func_name)(potential, mu, sigma)
        sim_fields['growth'][i]=growth

        if channel_params.use_flow:
            # Advect the board slice and local parameter maps using the flow field
            scaled_flow_x=flow_x*(2.0/GRID_DIM[1])*flow_strength
            scaled_flow_y=flow_y*(2.0/GRID_DIM[0])*flow_strength
            sampling_grid=torch.stack((grid_x-scaled_flow_x, grid_y-scaled_flow_y),dim=2)
            sampling_grid = torch.remainder(sampling_grid + 1.0, 2.0) - 1.0 # Ensure grid values are in [-1, 1]

            if channel_params.has_local_params and channel_params.id in local_param_maps:
                # Stack the board and parameter maps for simultaneous advection
                maps_to_advect = [game_board[i,:,:]] + [p_maps[name] for name in LOCAL_PARAM_NAMES]
                stacked_maps = torch.stack(maps_to_advect, dim=0).unsqueeze(0)
                advected_stack = torch.nn.functional.grid_sample(stacked_maps, sampling_grid.unsqueeze(0), mode='bicubic', padding_mode='zeros', align_corners=True).squeeze(0)
                advected_board = advected_stack[0,:,:]
                # Unstack the advected parameter maps
                for idx, name in enumerate(LOCAL_PARAM_NAMES):
                    local_param_maps[channel_params.id][name] = advected_stack[idx+1,:,:]
            else:
                advected_board=torch.nn.functional.grid_sample(original_slice, sampling_grid.unsqueeze(0), mode='bicubic', padding_mode='zeros', align_corners=True).squeeze()
            
            new_board[i,:,:]=torch.clamp(advected_board + dt * growth, 0, 1)
        else:
            # Update without flow (advection)
            sim_fields['flow_y'][i].zero_(); sim_fields['flow_x'][i].zero_()
            current_slice = game_board[i,:,:]
            new_board[i,:,:]=torch.clamp(current_slice + dt * growth, 0, 1)
    
    sim_fields['final_board']=new_board
    return sim_fields