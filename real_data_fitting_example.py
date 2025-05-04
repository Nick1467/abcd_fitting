import numpy as np
import matplotlib.pyplot as plt
from ABCD_fitting import MicrowaveNetworkCalculator
import os

def fit_real_data():
    """
    Example of fitting real S-parameter data to extract transmission line lengths
    and magnon parameters.
    
    This is a template that can be adapted to work with your specific data.
    """
    print("=== Fitting Real S-parameter Data ===")
    
    # Step 1: Define your frequency range
    # If loading from a file, you'll get this from your data
    # For this example, we'll define it manually
    f_start = 1e9   # 1 GHz
    f_stop = 10e9   # 10 GHz
    f_points = 401
    frequencies = np.linspace(f_start, f_stop, f_points)
    
    # Step 2: Load your measured S-parameter data
    # This is where you would load your actual measured data
    # For example, from a Touchstone file or CSV file
    measured_s_params = load_s_parameter_data(frequencies)
    
    # Step 3: Create the calculator with your frequency range
    calculator = MicrowaveNetworkCalculator(frequencies)
    
    # Step 4: Define your network template
    # This is where you specify your cable arrangement and Z0 values
    # The structure should match your physical setup
    # Initial parameter values are your best guesses
    network_template = [
        {'type': 'tline', 'z0': 50, 'length': 0.010},   # Initial guess for first transmission line
        {'type': 'tline', 'z0': 75, 'length': 0.020},   # Initial guess for second transmission line
        {'type': 'magnon', 'omega_m': 5e9, 'gamma': 1e7, 'kappa': 1e8},  # Initial guess for magnon
        {'type': 'tline', 'z0': 50, 'length': 0.015},   # Initial guess for third transmission line
    ]
    
    print("\nNetwork template structure:")
    for i, element in enumerate(network_template):
        if element['type'] == 'tline':
            print(f"  Element {i+1}: Transmission line with Z0 = {element['z0']} Ω, initial length = {element['length']*1000:.2f} mm")
        elif element['type'] == 'magnon':
            print(f"  Element {i+1}: Magnon element with initial parameters:")
            print(f"    - omega_m = {element['omega_m']/1e9:.2f} GHz")
            print(f"    - gamma = {element['gamma']/1e6:.2f} MHz")
            print(f"    - kappa = {element['kappa']/1e6:.2f} MHz")
    
    # Step 5: Perform the fitting
    print("\nPerforming fitting...")
    
    # Option 1: Fit only transmission line lengths
    print("\n1. Fitting only transmission line lengths...")
    fitted_network_tlines = calculator.fit_transmission_line_lengths(
        measured_s_params, network_template, verbose=1
    )
    
    # Calculate S-parameters for the fitted network (transmission lines only)
    _, fitted_s_params_tlines = calculator.calculate_network(fitted_network_tlines)
    
    # Option 2: Fit only magnon parameters (assuming transmission line lengths are known)
    print("\n2. Fitting only magnon parameters...")
    # Use the fitted transmission line lengths from the previous step
    magnon_template = fitted_network_tlines.copy()
    fitted_network_magnon = calculator.fit_magnon_parameters(
        measured_s_params, magnon_template, magnon_indices=[2], verbose=1
    )
    
    # Calculate S-parameters for the fitted network (magnon parameters)
    _, fitted_s_params_magnon = calculator.calculate_network(fitted_network_magnon)
    
    # Option 3: Fit all parameters simultaneously
    print("\n3. Fitting all parameters simultaneously...")
    fitted_network_all = calculator.fit_all_parameters(
        measured_s_params, network_template, verbose=1
    )
    
    # Calculate S-parameters for the fitted network (all parameters)
    _, fitted_s_params_all = calculator.calculate_network(fitted_network_all)
    
    # Step 6: Compare the results
    print("\nFitted Parameters:")
    
    # Print fitted transmission line lengths
    print("\nFitted Transmission Line Lengths:")
    for i, element in enumerate(fitted_network_all):
        if element['type'] == 'tline':
            print(f"  Line {i+1}: length = {element['length']*1000:.3f} mm, Z0 = {element['z0']:.1f} Ω")
    
    # Print fitted magnon parameters
    print("\nFitted Magnon Parameters:")
    for i, element in enumerate(fitted_network_all):
        if element['type'] == 'magnon':
            print(f"  Magnon {i+1}:")
            print(f"    - omega_m = {element['omega_m']/1e9:.3f} GHz")
            print(f"    - gamma = {element['gamma']/1e6:.3f} MHz")
            print(f"    - kappa = {element['kappa']/1e6:.3f} MHz")
    
    # Step 7: Plot the comparison between measured and fitted S-parameters
    print("\nPlotting comparison between measured and fitted S-parameters...")
    
    # Compare the results from the three fitting approaches
    plt.figure(figsize=(12, 8))
    
    # Plot S11 magnitude
    plt.subplot(2, 2, 1)
    plot_s_parameter_comparison(calculator, frequencies, measured_s_params, 
                               [fitted_s_params_tlines, fitted_s_params_magnon, fitted_s_params_all],
                               ['S11'], db_scale=True, 
                               labels=['Measured', 'Fitted (T-lines only)', 'Fitted (Magnon only)', 'Fitted (All)'])
    plt.title('S11 Magnitude')
    
    # Plot S21 magnitude
    plt.subplot(2, 2, 2)
    plot_s_parameter_comparison(calculator, frequencies, measured_s_params, 
                               [fitted_s_params_tlines, fitted_s_params_magnon, fitted_s_params_all],
                               ['S21'], db_scale=True, 
                               labels=['Measured', 'Fitted (T-lines only)', 'Fitted (Magnon only)', 'Fitted (All)'])
    plt.title('S21 Magnitude')
    
    # Plot S11 phase
    plt.subplot(2, 2, 3)
    plot_s_parameter_comparison(calculator, frequencies, measured_s_params, 
                               [fitted_s_params_tlines, fitted_s_params_magnon, fitted_s_params_all],
                               ['S11'], plot_magnitude=False, plot_phase=True,
                               labels=['Measured', 'Fitted (T-lines only)', 'Fitted (Magnon only)', 'Fitted (All)'])
    plt.title('S11 Phase')
    
    # Plot S21 phase
    plt.subplot(2, 2, 4)
    plot_s_parameter_comparison(calculator, frequencies, measured_s_params, 
                               [fitted_s_params_tlines, fitted_s_params_magnon, fitted_s_params_all],
                               ['S21'], plot_magnitude=False, plot_phase=True,
                               labels=['Measured', 'Fitted (T-lines only)', 'Fitted (Magnon only)', 'Fitted (All)'])
    plt.title('S21 Phase')
    
    plt.tight_layout()
    plt.show()
    
    # Also show the detailed comparison for the best fit (all parameters)
    calculator.compare_s_parameters(measured_s_params, fitted_s_params_all, param_names=['S11', 'S21'])
    
    print("\nFitting completed!")
    print("The plots show the comparison between measured and fitted S-parameters.")
    print("You can use the fitted parameters in your further analysis.")

def load_s_parameter_data(frequencies):
    """
    Load S-parameter data from a file or generate synthetic data for testing.
    
    In a real application, you would replace this with code to load your actual measured data.
    
    Args:
        frequencies: Array of frequencies in Hz
        
    Returns:
        Array of S-parameter matrices
    """
    # Check if we have a real data file
    # If you have real data, uncomment and modify this section
    """
    data_file = "your_s_parameter_data.s2p"  # Touchstone file
    if os.path.exists(data_file):
        # Load data from Touchstone file
        # This is just a placeholder - you'll need to implement the actual loading
        # based on your file format
        s_params = load_touchstone_file(data_file, frequencies)
        return s_params
    """
    
    # If no real data is available, generate synthetic data for testing
    print("No real data file found. Generating synthetic data for testing...")
    
    # Define a "true" network for generating synthetic data
    true_network = [
        {'type': 'tline', 'z0': 50, 'length': 0.018},
        {'type': 'tline', 'z0': 75, 'length': 0.022},
        {'type': 'magnon', 'omega_m': 5.5e9, 'gamma': 7e6, 'kappa': 9e7},
        {'type': 'tline', 'z0': 50, 'length': 0.015},
    ]
    
    # Create calculator and generate clean S-parameters
    calculator = MicrowaveNetworkCalculator(frequencies)
    _, clean_s_params = calculator.calculate_network(true_network)
    
    # Add some noise to simulate real measurements
    noisy_s_params = add_noise_to_s_params(clean_s_params, noise_level=0.03)
    
    return noisy_s_params

def add_noise_to_s_params(s_params, noise_level=0.05):
    """
    Add random noise to S-parameters to simulate measurement noise.
    
    Args:
        s_params: Array of S-parameter matrices
        noise_level: Level of random noise to add (0.0 to 1.0)
        
    Returns:
        Array of S-parameter matrices with added noise
    """
    noisy_s_params = []
    for s_matrix in s_params:
        noisy_matrix = np.zeros_like(s_matrix, dtype=complex)
        for i in range(s_matrix.shape[0]):
            for j in range(s_matrix.shape[1]):
                # Add noise to magnitude and phase separately
                mag = np.abs(s_matrix[i, j])
                phase = np.angle(s_matrix[i, j])
                
                # Add noise to magnitude (ensure it stays positive)
                mag_noise = mag * (1 + noise_level * (np.random.random() - 0.5))
                mag_noise = max(0.001, mag_noise)  # Ensure magnitude is positive
                
                # Add noise to phase
                phase_noise = phase + noise_level * np.pi * (np.random.random() - 0.5)
                
                # Convert back to complex
                noisy_matrix[i, j] = mag_noise * np.exp(1j * phase_noise)
        
        noisy_s_params.append(noisy_matrix)
    
    return np.array(noisy_s_params)

def plot_s_parameter_comparison(calculator, frequencies, measured_s_params, fitted_s_params_list, 
                               param_names=['S11'], plot_magnitude=True, plot_phase=False,
                               db_scale=True, labels=None):
    """
    Plot comparison of measured vs multiple fitted S-parameters.
    
    Args:
        calculator: MicrowaveNetworkCalculator instance
        frequencies: Array of frequencies in Hz
        measured_s_params: Array of measured S-parameter matrices
        fitted_s_params_list: List of arrays of fitted S-parameter matrices
        param_names: List of S-parameter names to plot (e.g., ['S11', 'S21'])
        plot_magnitude: Whether to plot magnitude
        plot_phase: Whether to plot phase
        db_scale: Whether to use dB scale for magnitude
        labels: List of labels for the data series
    """
    indices = {
        'S11': (0, 0),
        'S12': (0, 1),
        'S21': (1, 0),
        'S22': (1, 1)
    }
    
    freq_ghz = frequencies / 1e9  # Convert to GHz for plotting
    
    if labels is None:
        labels = ['Measured'] + [f'Fitted {i+1}' for i in range(len(fitted_s_params_list))]
    
    for name in param_names:
        i, j = indices[name]
        
        if plot_magnitude:
            # Plot measured data
            measured_mag = np.abs([s[i, j] for s in measured_s_params])
            
            if db_scale:
                measured_mag_db = 20 * np.log10(measured_mag)
                plt.plot(freq_ghz, measured_mag_db, 'b-', label=labels[0])
            else:
                plt.plot(freq_ghz, measured_mag, 'b-', label=labels[0])
            
            # Plot fitted data
            for idx, fitted_s_params in enumerate(fitted_s_params_list):
                fitted_mag = np.abs([s[i, j] for s in fitted_s_params])
                
                if db_scale:
                    fitted_mag_db = 20 * np.log10(fitted_mag)
                    plt.plot(freq_ghz, fitted_mag_db, '--', label=labels[idx+1])
                else:
                    plt.plot(freq_ghz, fitted_mag, '--', label=labels[idx+1])
            
            if db_scale:
                plt.ylabel('Magnitude (dB)')
            else:
                plt.ylabel('Magnitude')
        
        if plot_phase:
            # Plot measured data
            measured_phase = np.angle([s[i, j] for s in measured_s_params]) * 180 / np.pi
            plt.plot(freq_ghz, measured_phase, 'b-', label=labels[0])
            
            # Plot fitted data
            for idx, fitted_s_params in enumerate(fitted_s_params_list):
                fitted_phase = np.angle([s[i, j] for s in fitted_s_params]) * 180 / np.pi
                plt.plot(freq_ghz, fitted_phase, '--', label=labels[idx+1])
            
            plt.ylabel('Phase (degrees)')
        
        plt.grid(True)
        plt.legend()
        plt.xlabel('Frequency (GHz)')

if __name__ == "__main__":
    fit_real_data()
