import numpy as np
import matplotlib.pyplot as plt
from ABCD_fitting import MicrowaveNetworkCalculator

def main():
    """
    Simple example demonstrating how to fit transmission line lengths and magnon parameters
    from S-parameter data.
    """
    # Define frequency range
    f_start = 1e9   # 1 GHz
    f_stop = 10e9   # 10 GHz
    f_points = 2001
    frequencies = np.linspace(f_start, f_stop, f_points)
    
    # Create calculator
    calculator = MicrowaveNetworkCalculator(frequencies)
    
    # Load measured S-parameter data
    # In a real scenario, you would load your measured data here
    # For this example, we'll generate synthetic data
    
    # Define a "true" network that represents our device under test
    true_network = [
        {'type': 'tline', 'z0': 50, 'length': 0.015},   # 15mm transmission line
        {'type': 'tline', 'z0': 75, 'length': 0.025},   # 25mm transmission line
        {'type': 'magnon', 'omega_m': 5e9, 'gamma': 5e6, 'kappa': 8e7},  # Magnon element
        {'type': 'tline', 'z0': 50, 'length': 0.020},   # 20mm transmission line
    ]
    
    # Generate synthetic "measured" data
    _, measured_s_params = calculator.calculate_network(true_network)
    
    # Add some noise to simulate real measurements
    noisy_s_params = add_noise_to_s_params(measured_s_params, noise_level=0.01)
    
    # Create a template network with initial guesses for the parameters
    # This is where you define your cable arrangement and Z0 values
    template_network = [
        {'type': 'tline', 'z0': 50, 'length': 0.010},   # Initial guess
        {'type': 'tline', 'z0': 75, 'length': 0.020},   # Initial guess
        {'type': 'magnon', 'omega_m': 4.5e9, 'gamma': 1e7, 'kappa': 5e7},  # Initial guess
        {'type': 'tline', 'z0': 50, 'length': 0.015},   # Initial guess
    ]
    
    print("=== Fitting Example ===")
    print("\nNetwork structure:")
    for i, element in enumerate(template_network):
        if element['type'] == 'tline':
            print(f"  Element {i+1}: Transmission line with Z0 = {element['z0']} Ω")
        elif element['type'] == 'magnon':
            print(f"  Element {i+1}: Magnon element")
    
    print("\nInitial parameter values:")
    for i, element in enumerate(template_network):
        if element['type'] == 'tline':
            print(f"  Transmission line {i+1}: length = {element['length']*1000:.2f} mm")
        elif element['type'] == 'magnon':
            print(f"  Magnon {i+1}: omega_m = {element['omega_m']/1e9:.2f} GHz, gamma = {element['gamma']/1e6:.2f} MHz, kappa = {element['kappa']/1e6:.2f} MHz")
    
    # Option 1: Fit only transmission line lengths
    print("\n1. Fitting only transmission line lengths...")
    fitted_network_tlines = calculator.fit_transmission_line_lengths(
        noisy_s_params, template_network, verbose=1
    )
    
    # Option 2: Fit only magnon parameters
    print("\n2. Fitting only magnon parameters...")
    fitted_network_magnon = calculator.fit_magnon_parameters(
        noisy_s_params, template_network, magnon_indices=[2], verbose=1
    )
    
    # Option 3: Fit all parameters
    print("\n3. Fitting all parameters (transmission lines and magnon)...")
    fitted_network_all = calculator.fit_all_parameters(
        noisy_s_params, template_network, verbose=1
    )
    
    # Calculate S-parameters for the fitted network
    _, fitted_s_params = calculator.calculate_network(fitted_network_all)
    
    # Compare the results
    print("\nTrue vs. Fitted Parameters:")
    
    # Transmission lines
    for i in [0, 1, 3]:
        true_length = true_network[i]['length']
        fitted_length = fitted_network_all[i]['length']
        error_percent = abs(true_length - fitted_length) / true_length * 100
        print(f"  Line {i+1}: True = {true_length*1000:.2f} mm, Fitted = {fitted_length*1000:.2f} mm, Error = {error_percent:.2f}%")
    
    # Magnon
    true_magnon = true_network[2]
    fitted_magnon = fitted_network_all[2]
    print(f"  Magnon: omega_m: True = {true_magnon['omega_m']/1e9:.3f} GHz, Fitted = {fitted_magnon['omega_m']/1e9:.3f} GHz")
    print(f"          gamma: True = {true_magnon['gamma']/1e6:.3f} MHz, Fitted = {fitted_magnon['gamma']/1e6:.3f} MHz")
    print(f"          kappa: True = {true_magnon['kappa']/1e6:.3f} MHz, Fitted = {fitted_magnon['kappa']/1e6:.3f} MHz")
    
    # Plot the comparison
    calculator.compare_s_parameters(noisy_s_params, fitted_s_params, param_names=['S11', 'S21'])
    
    print("\nFitting completed! The plots show the comparison between measured and fitted S-parameters.")

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

if __name__ == "__main__":
    main()
