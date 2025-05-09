import numpy as np
import matplotlib.pyplot as plt
from ABCD_fitting import MicrowaveNetworkCalculator
from matplotlib import cm
from matplotlib.colors import Normalize

def sweep_magnon_omega_m(network, omega_m_range, frequency_range, magnon_index=None, 
                        plot_params=['S21'], db_scale=True, plot_type='waterfall'):
    """
    Sweep the omega_m parameter of a magnon element and plot the resulting S-parameters.
    
    Args:
        network: List of dictionaries describing the network elements
        omega_m_range: Array of omega_m values to sweep (in Hz)
        frequency_range: Array of frequencies to evaluate (in Hz)
        magnon_index: Index of the magnon element to sweep (if None, finds the first magnon)
        plot_params: List of S-parameters to plot (e.g., ['S11', 'S21'])
        db_scale: Whether to use dB scale for magnitude
        plot_type: Type of plot ('waterfall', 'heatmap', or 'lines')
        
    Returns:
        Tuple of (frequencies, omega_m_values, s_params_array)
    """
    # Find the magnon element if index not provided
    if magnon_index is None:
        for i, element in enumerate(network):
            if element['type'] == 'magnon':
                magnon_index = i
                break
        if magnon_index is None:
            raise ValueError("No magnon element found in the network")
    
    # Verify the element is a magnon
    if network[magnon_index]['type'] != 'magnon':
        raise ValueError(f"Element at index {magnon_index} is not a magnon")
    
    # Create calculator
    calculator = MicrowaveNetworkCalculator(frequency_range)
    
    # Store original omega_m value
    original_omega_m = network[magnon_index]['omega_m']
    
    # Initialize array to store S-parameters for each omega_m value
    s_params_array = []
    
    # Sweep omega_m and calculate S-parameters
    for omega_m in omega_m_range:
        # Update the magnon's omega_m
        network[magnon_index]['omega_m'] = omega_m
        
        # Calculate S-parameters
        _, s_params = calculator.calculate_network(network)
        s_params_array.append(s_params)
    
    # Restore original omega_m value
    network[magnon_index]['omega_m'] = original_omega_m
    
    # Convert to numpy array
    s_params_array = np.array(s_params_array)
    
    # Plot the results
    if plot_type == 'waterfall':
        plot_waterfall(frequency_range, omega_m_range, s_params_array, plot_params, db_scale)
    elif plot_type == 'heatmap':
        plot_heatmap(frequency_range, omega_m_range, s_params_array, plot_params, db_scale)
    elif plot_type == 'lines':
        plot_lines(frequency_range, omega_m_range, s_params_array, plot_params, db_scale)
    else:
        raise ValueError(f"Unknown plot_type: {plot_type}")
    
    return frequency_range, omega_m_range, s_params_array

def plot_waterfall(frequency_range, omega_m_range, s_params_array, plot_params=['S21'], db_scale=True):
    """
    Create a 3D waterfall plot of S-parameters vs frequency and omega_m.
    
    Args:
        frequency_range: Array of frequencies (in Hz)
        omega_m_range: Array of omega_m values (in Hz)
        s_params_array: Array of S-parameter matrices [omega_m, freq, 2, 2]
        plot_params: List of S-parameters to plot (e.g., ['S11', 'S21'])
        db_scale: Whether to use dB scale for magnitude
    """
    indices = {
        'S11': (0, 0),
        'S12': (0, 1),
        'S21': (1, 0),
        'S22': (1, 1)
    }
    
    freq_ghz = frequency_range / 1e9  # Convert to GHz
    omega_m_ghz = omega_m_range / 1e9  # Convert to GHz
    
    # Create figure
    fig = plt.figure(figsize=(12, 8))
    
    for param_idx, param_name in enumerate(plot_params):
        i, j = indices[param_name]
        
        # Create 3D axis
        if len(plot_params) > 1:
            ax = fig.add_subplot(1, len(plot_params), param_idx + 1, projection='3d')
        else:
            ax = fig.add_subplot(111, projection='3d')
        
        # Create meshgrid for 3D surface
        X, Y = np.meshgrid(freq_ghz, omega_m_ghz)
        
        # Extract magnitude data
        Z = np.zeros((len(omega_m_range), len(frequency_range)))
        for om_idx, _ in enumerate(omega_m_range):
            for f_idx, _ in enumerate(frequency_range):
                mag = np.abs(s_params_array[om_idx, f_idx, i, j])
                if db_scale:
                    Z[om_idx, f_idx] = 20 * np.log10(mag)
                else:
                    Z[om_idx, f_idx] = mag
        
        # Plot the surface
        surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis, linewidth=0, antialiased=True)
        
        # Add colorbar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, 
                    label=f"|{param_name}| {'(dB)' if db_scale else ''}")
        
        # Set labels
        ax.set_xlabel('Frequency (GHz)')
        ax.set_ylabel('Magnon $\omega_m$ (GHz)')
        ax.set_zlabel(f"|{param_name}| {'(dB)' if db_scale else ''}")
        ax.set_title(f"{param_name} vs Frequency and $\omega_m$")
    
    plt.tight_layout()
    plt.show()

def plot_heatmap(frequency_range, omega_m_range, s_params_array, plot_params=['S21'], db_scale=True):
    """
    Create a heatmap plot of S-parameters vs frequency and omega_m.
    
    Args:
        frequency_range: Array of frequencies (in Hz)
        omega_m_range: Array of omega_m values (in Hz)
        s_params_array: Array of S-parameter matrices [omega_m, freq, 2, 2]
        plot_params: List of S-parameters to plot (e.g., ['S11', 'S21'])
        db_scale: Whether to use dB scale for magnitude
    """
    indices = {
        'S11': (0, 0),
        'S12': (0, 1),
        'S21': (1, 0),
        'S22': (1, 1)
    }
    
    freq_ghz = frequency_range / 1e9  # Convert to GHz
    omega_m_ghz = omega_m_range / 1e9  # Convert to GHz
    
    # Create figure
    fig, axs = plt.subplots(1, len(plot_params), figsize=(6*len(plot_params), 5))
    
    if len(plot_params) == 1:
        axs = [axs]
    
    for param_idx, param_name in enumerate(plot_params):
        i, j = indices[param_name]
        ax = axs[param_idx]
        
        # Extract magnitude data
        Z = np.zeros((len(omega_m_range), len(frequency_range)))
        for om_idx, _ in enumerate(omega_m_range):
            for f_idx, _ in enumerate(frequency_range):
                mag = np.abs(s_params_array[om_idx, f_idx, i, j])
                if db_scale:
                    Z[om_idx, f_idx] = 20 * np.log10(mag)
                else:
                    Z[om_idx, f_idx] = mag
        
        # Plot heatmap
        im = ax.imshow(Z, aspect='auto', origin='lower', 
                      extent=[freq_ghz.min(), freq_ghz.max(), omega_m_ghz.min(), omega_m_ghz.max()],
                      cmap='viridis')
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(f"|{param_name}| {'(dB)' if db_scale else ''}")
        
        # Set labels
        ax.set_xlabel('Frequency (GHz)')
        ax.set_ylabel('Magnon $\omega_m$ (GHz)')
        ax.set_title(f"{param_name} vs Frequency and $\omega_m$")
    
    plt.tight_layout()
    plt.show()

def plot_lines(frequency_range, omega_m_range, s_params_array, plot_params=['S21'], db_scale=True):
    """
    Create a line plot of S-parameters vs frequency for different omega_m values.
    
    Args:
        frequency_range: Array of frequencies (in Hz)
        omega_m_range: Array of omega_m values (in Hz)
        s_params_array: Array of S-parameter matrices [omega_m, freq, 2, 2]
        plot_params: List of S-parameters to plot (e.g., ['S11', 'S21'])
        db_scale: Whether to use dB scale for magnitude
    """
    indices = {
        'S11': (0, 0),
        'S12': (0, 1),
        'S21': (1, 0),
        'S22': (1, 1)
    }
    
    freq_ghz = frequency_range / 1e9  # Convert to GHz
    
    # Create figure
    fig, axs = plt.subplots(len(plot_params), 1, figsize=(10, 4*len(plot_params)))
    
    if len(plot_params) == 1:
        axs = [axs]
    
    # Create colormap for omega_m values
    norm = Normalize(vmin=omega_m_range.min()/1e9, vmax=omega_m_range.max()/1e9)
    cmap = cm.viridis
    
    for param_idx, param_name in enumerate(plot_params):
        i, j = indices[param_name]
        ax = axs[param_idx]
        
        # Plot lines for each omega_m value
        for om_idx, omega_m in enumerate(omega_m_range):
            # Extract magnitude data for this omega_m
            mags = []
            for f_idx, _ in enumerate(frequency_range):
                mag = np.abs(s_params_array[om_idx, f_idx, i, j])
                if db_scale:
                    mags.append(20 * np.log10(mag))
                else:
                    mags.append(mag)
            
            # Plot with color based on omega_m value
            color = cmap(norm(omega_m/1e9))
            ax.plot(freq_ghz, mags, color=color, alpha=0.7)
        
        # Set labels
        ax.set_xlabel('Frequency (GHz)')
        ax.set_ylabel(f"|{param_name}| {'(dB)' if db_scale else ''}")
        ax.set_title(f"{param_name} vs Frequency for different $\omega_m$ values")
        ax.grid(True)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axs)
    cbar.set_label('Magnon $\omega_m$ (GHz)')
    
    plt.tight_layout()
    plt.show()

def example_magnon_sweep():
    """
    Example of sweeping magnon omega_m parameter and plotting the results.
    """
    print("=== Example: Magnon Omega_m Sweep ===")
    
    # Define frequency range
    f_start = 1e9   # 1 GHz
    f_stop = 10e9   # 10 GHz
    f_points = 201
    frequencies = np.linspace(f_start, f_stop, f_points)
    
    # Define omega_m range to sweep
    omega_m_start = 3e9   # 3 GHz
    omega_m_stop = 8e9    # 8 GHz
    omega_m_points = 21
    omega_m_range = np.linspace(omega_m_start, omega_m_stop, omega_m_points)
    '''
    network = [
        {'type': 'tline', 'z0': 50, 'length': 0.010},   # Known transmission line
        
        {'type': 'magnon', 'omega_m': 4e9, 'gamma': 2e7, 'kappa': 5e7},  # Initial guess for magnon
        {'type': 'tline', 'z0': 50, 'length': 0.010},   # Known transmission line
    ]

    '''
    # Define network with magnon and transmission lines
    network = [
        {'type': 'tline', 'z0': 125, 'length': 0.034},   # 15mm transmission line
        {'type': 'tline', 'z0': 28, 'length': 0.007565},   # 15mm transmission line
        {'type': 'tline', 'z0': 125, 'length': 0.034},   # 15mm transmission line
        {'type': 'tline', 'z0': 28, 'length': 0.007565},   # 15mm transmission line
        {'type': 'tline', 'z0': 125, 'length': 0.034},   # 15mm transmission line
        {'type': 'tline', 'z0': 28, 'length': 0.007565},   # 15mm transmission line
        {'type': 'tline', 'z0': 125, 'length': 0.034},   # 15mm transmission line
        {'type': 'tline', 'z0': 28, 'length': 0.007565},   # 15mm transmission line
        {'type': 'tline', 'z0': 125, 'length': 0.034},   # 15mm transmission line
        {'type': 'tline', 'z0': 28, 'length': 0.007565},   # 15mm transmission line
        {'type': 'tline', 'z0': 125, 'length': 0.034},   # 15mm transmission line
        {'type': 'tline', 'z0': 28, 'length': 0.007565},   # 15mm transmission line
        {'type': 'tline', 'z0': 125, 'length': 0.034},   # 15mm transmission line
        {'type': 'tline', 'z0': 28, 'length': 0.007565},   # 15mm transmission line
        {'type': 'tline', 'z0': 125, 'length': 0.034},   # 15mm transmission line
        {'type': 'tline', 'z0': 28, 'length': 0.007565},   # 15mm transmission line
        {'type': 'tline', 'z0': 125, 'length': 0.034},   # 15mm transmission line
        {'type': 'tline', 'z0': 28, 'length': 0.007565},   # 15mm transmission line

        {'type': 'magnon', 'omega_m': 5e9, 'gamma': 5e6, 'kappa': 8e7},  # Magnon element

        {'type': 'tline', 'z0': 125, 'length': 0.034},   # 15mm transmission line
        {'type': 'tline', 'z0': 28, 'length': 0.007565},   # 15mm transmission line
        {'type': 'tline', 'z0': 125, 'length': 0.034},   # 15mm transmission line
        {'type': 'tline', 'z0': 28, 'length': 0.007565},   # 15mm transmission line
        {'type': 'tline', 'z0': 125, 'length': 0.034},   # 15mm transmission line
        {'type': 'tline', 'z0': 28, 'length': 0.007565},   # 15mm transmission line
        {'type': 'tline', 'z0': 125, 'length': 0.034},   # 15mm transmission line
        {'type': 'tline', 'z0': 28, 'length': 0.007565},   # 15mm transmission line
        {'type': 'tline', 'z0': 125, 'length': 0.034},   # 15mm transmission line
        {'type': 'tline', 'z0': 28, 'length': 0.007565},   # 15mm transmission line
        {'type': 'tline', 'z0': 125, 'length': 0.034},   # 15mm transmission line
        {'type': 'tline', 'z0': 28, 'length': 0.007565},   # 15mm transmission line
        {'type': 'tline', 'z0': 125, 'length': 0.034},   # 15mm transmission line
        {'type': 'tline', 'z0': 28, 'length': 0.007565},   # 15mm transmission line
        {'type': 'tline', 'z0': 125, 'length': 0.034},   # 15mm transmission line
        {'type': 'tline', 'z0': 28, 'length': 0.007565},   # 15mm transmission line
        {'type': 'tline', 'z0': 125, 'length': 0.034},   # 15mm transmission line
    ]
    
    print(f"Network structure:")
    for i, element in enumerate(network):
        if element['type'] == 'tline':
            print(f"  Element {i+1}: Transmission line with Z0 = {element['z0']} Ω, length = {element['length']*1000:.2f} mm")
        elif element['type'] == 'magnon':
            print(f"  Element {i+1}: Magnon with parameters:")
            print(f"    - omega_m = {element['omega_m']/1e9:.2f} GHz")
            print(f"    - gamma = {element['gamma']/1e6:.2f} MHz")
            print(f"    - kappa = {element['kappa']/1e6:.2f} MHz")
    
    print(f"\nSweeping omega_m from {omega_m_start/1e9:.2f} GHz to {omega_m_stop/1e9:.2f} GHz")
    
    # Perform the sweep and plot results
    print("\n1. Waterfall plot:")
    sweep_magnon_omega_m(network, omega_m_range, frequencies, 
                        plot_params=['S21'], db_scale=True, plot_type='waterfall')
    
    print("\n2. Heatmap plot:")
    sweep_magnon_omega_m(network, omega_m_range, frequencies,  
                        plot_params=['S11', 'S21'], db_scale=True, plot_type='heatmap')
    
    print("\n3. Line plot:")
    sweep_magnon_omega_m(network, omega_m_range, frequencies, 
                        plot_params=['S21'], db_scale=True, plot_type='lines')

if __name__ == "__main__":
    example_magnon_sweep()
