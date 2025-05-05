import numpy as np
import matplotlib.pyplot as plt
from ABCD_fitting import MicrowaveNetworkCalculator

def generate_synthetic_data(frequency_range, network, noise_level=0.05, termination=None):
    """
    Generate synthetic S-parameter data with some noise to simulate measured data.
    
    Args:
        frequency_range: Array of frequencies in Hz
        network: List of network elements
        noise_level: Level of random noise to add (0.0 to 1.0)
        termination: Optional termination type
        
    Returns:
        Array of S-parameter matrices with added noise
    """
    # Create calculator and get clean S-parameters
    calculator = MicrowaveNetworkCalculator(frequency_range)
    _, clean_s_params = calculator.calculate_network(network, termination)
    
    # Add random noise to the S-parameters
    noisy_s_params = []
    for s_matrix in clean_s_params:
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

def example_fit_magnon_parameters():
    """
    Example of fitting magnon parameters with improved error control.
    """
    print("\n=== Example: Fitting Magnon Parameters with Improved Error Control ===")
    
    # Define frequency range
    f_start = 2e9  # 2 GHz
    f_stop = 8e9   # 8 GHz
    f_points = 301
    frequencies = np.linspace(f_start, f_stop, f_points)
    
    # Define the true network (this will be our reference)
    true_network = [
        {'type': 'tline', 'z0': 50, 'length': 0.010},   # 10mm transmission line
        {'type': 'magnon', 'omega_m': 5e9, 'gamma': 5e6, 'kappa': 1e8},  # Magnon element
        {'type': 'tline', 'z0': 50, 'length': 0.010},   # 10mm transmission line
    ]
    
    # Generate synthetic "measured" data with some noise
    measured_s_params = generate_synthetic_data(frequencies, true_network, noise_level=0.03)
    
    # Create a template network with initial guesses for the parameters
    # These initial guesses are intentionally far from the true values
    template_network = [
        {'type': 'tline', 'z0': 50, 'length': 0.010},   # Known transmission line
        {'type': 'magnon', 'omega_m': 4e9, 'gamma': 2e7, 'kappa': 5e7},  # Initial guess for magnon
        {'type': 'tline', 'z0': 50, 'length': 0.010},   # Known transmission line
    ]
    
    # Create calculator
    calculator = MicrowaveNetworkCalculator(frequencies)
    
    print("\nTrue magnon parameters:")
    true_magnon = true_network[1]
    print(f"  omega_m = {true_magnon['omega_m']/1e9:.3f} GHz")
    print(f"  gamma = {true_magnon['gamma']/1e6:.3f} MHz")
    print(f"  kappa = {true_magnon['kappa']/1e6:.3f} MHz")
    
    print("\nInitial guess for magnon parameters:")
    initial_magnon = template_network[1]
    print(f"  omega_m = {initial_magnon['omega_m']/1e9:.3f} GHz")
    print(f"  gamma = {initial_magnon['gamma']/1e6:.3f} MHz")
    print(f"  kappa = {initial_magnon['kappa']/1e6:.3f} MHz")
    
    # Try fitting with different error thresholds
    error_thresholds = [10.0, 5.0, 3.0, 1.0]
    
    for threshold in error_thresholds:
        print(f"\n--- Fitting with error threshold: {threshold}% ---")
        
        # Set up parameters to fit
        params_to_fit = {1: ['omega_m', 'gamma', 'kappa']}
        
        # Set reasonable bounds for magnon parameters
        bounds = {
            'magnon_omega_m': (1e9, 10e9),    # 1 GHz to 10 GHz
            'magnon_gamma': (1e6, 1e8),       # 1 MHz to 100 MHz
            'magnon_kappa': (1e7, 1e9)        # 10 MHz to 1 GHz
        }
        
        # Perform the fitting
        fitted_network, final_error = calculator.fit_network(
            measured_s_params, 
            template_network, 
            params_to_fit, 
            bounds=bounds,
            method='trf',
            max_error_percent=threshold,
            max_iterations=5,
            verbose=1
        )
        
        # Calculate S-parameters for the fitted network
        _, fitted_s_params = calculator.calculate_network(fitted_network)
        
        # Compare the results
        print("\nFitted vs. True Magnon Parameters:")
        fitted_magnon = fitted_network[1]
        
        omega_m_error = abs(fitted_magnon['omega_m'] - true_magnon['omega_m']) / true_magnon['omega_m'] * 100
        gamma_error = abs(fitted_magnon['gamma'] - true_magnon['gamma']) / true_magnon['gamma'] * 100
        kappa_error = abs(fitted_magnon['kappa'] - true_magnon['kappa']) / true_magnon['kappa'] * 100
        
        print(f"  omega_m: Fitted = {fitted_magnon['omega_m']/1e9:.3f} GHz, True = {true_magnon['omega_m']/1e9:.3f} GHz, Error = {omega_m_error:.2f}%")
        print(f"  gamma: Fitted = {fitted_magnon['gamma']/1e6:.3f} MHz, True = {true_magnon['gamma']/1e6:.3f} MHz, Error = {gamma_error:.2f}%")
        print(f"  kappa: Fitted = {fitted_magnon['kappa']/1e6:.3f} MHz, True = {true_magnon['kappa']/1e6:.3f} MHz, Error = {kappa_error:.2f}%")
        
        # Plot the comparison
        if threshold == 5.0:  # Only plot for the 5% threshold case
            calculator.compare_s_parameters(measured_s_params, fitted_s_params, param_names=['S11', 'S21'])

def example_fit_complex_network():
    """
    Example of fitting a complex network with multiple elements.
    """
    print("\n=== Example: Fitting Complex Network with Multiple Elements ===")
    
    # Define frequency range
    f_start = 1e9   # 1 GHz
    f_stop = 10e9   # 10 GHz
    f_points = 401
    frequencies = np.linspace(f_start, f_stop, f_points)
    
    # Define the true network (this will be our reference)
    true_network = [
        {'type': 'tline', 'z0': 50, 'length': 0.015},   # 15mm transmission line
        {'type': 'tline', 'z0': 75, 'length': 0.025},   # 25mm transmission line
        {'type': 'magnon', 'omega_m': 6e9, 'gamma': 8e6, 'kappa': 1.2e8},  # Magnon element
        {'type': 'tline', 'z0': 50, 'length': 0.020},   # 20mm transmission line
    ]
    
    # Generate synthetic "measured" data with some noise
    measured_s_params = generate_synthetic_data(frequencies, true_network, noise_level=0.02)
    
    # Create a template network with initial guesses for the parameters
    template_network = [
        {'type': 'tline', 'z0': 50, 'length': 0.010},   # Initial guess
        {'type': 'tline', 'z0': 75, 'length': 0.020},   # Initial guess
        {'type': 'magnon', 'omega_m': 5.5e9, 'gamma': 1e7, 'kappa': 1e8},  # Initial guess
        {'type': 'tline', 'z0': 50, 'length': 0.015},   # Initial guess
    ]
    
    # Create calculator
    calculator = MicrowaveNetworkCalculator(frequencies)
    
    print("\nTrue network parameters:")
    for i, element in enumerate(true_network):
        if element['type'] == 'tline':
            print(f"  Transmission line {i+1}: length = {element['length']*1000:.2f} mm, Z0 = {element['z0']:.1f} Ω")
        elif element['type'] == 'magnon':
            print(f"  Magnon {i+1}: omega_m = {element['omega_m']/1e9:.3f} GHz, gamma = {element['gamma']/1e6:.3f} MHz, kappa = {element['kappa']/1e6:.3f} MHz")
    
    print("\nInitial guesses:")
    for i, element in enumerate(template_network):
        if element['type'] == 'tline':
            print(f"  Transmission line {i+1}: length = {element['length']*1000:.2f} mm, Z0 = {element['z0']:.1f} Ω")
        elif element['type'] == 'magnon':
            print(f"  Magnon {i+1}: omega_m = {element['omega_m']/1e9:.3f} GHz, gamma = {element['gamma']/1e6:.3f} MHz, kappa = {element['kappa']/1e6:.3f} MHz")
    
    # Perform the fitting with 5% error threshold
    print("\n--- Fitting all parameters with 5% error threshold ---")
    
    # Set up parameters to fit
    params_to_fit = {
        0: ['length'],
        1: ['length'],
        2: ['omega_m', 'gamma', 'kappa'],
        3: ['length']
    }
    
    # Set reasonable bounds
    bounds = {
        'tline_length': (0.001, 0.1),      # Between 1mm and 10cm
        'magnon_omega_m': (1e9, 10e9),     # 1 GHz to 10 GHz
        'magnon_gamma': (1e6, 1e8),        # 1 MHz to 100 MHz
        'magnon_kappa': (1e7, 1e9)         # 10 MHz to 1 GHz
    }
    
    # Perform the fitting
    fitted_network, final_error = calculator.fit_network(
        measured_s_params, 
        template_network, 
        params_to_fit, 
        bounds=bounds,
        method='trf',
        max_error_percent=5.0,
        max_iterations=5,
        verbose=1
    )
    
    # Calculate S-parameters for the fitted network
    _, fitted_s_params = calculator.calculate_network(fitted_network)
    
    # Compare the results
    print("\nFitted vs. True Parameters:")
    
    # Transmission lines
    for i in [0, 1, 3]:
        true_length = true_network[i]['length']
        fitted_length = fitted_network[i]['length']
        error_percent = abs(true_length - fitted_length) / true_length * 100
        print(f"  Line {i+1}: Fitted = {fitted_length*1000:.2f} mm, True = {true_length*1000:.2f} mm, Error = {error_percent:.2f}%")
    
    # Magnon
    true_magnon = true_network[2]
    fitted_magnon = fitted_network[2]
    
    omega_m_error = abs(fitted_magnon['omega_m'] - true_magnon['omega_m']) / true_magnon['omega_m'] * 100
    gamma_error = abs(fitted_magnon['gamma'] - true_magnon['gamma']) / true_magnon['gamma'] * 100
    kappa_error = abs(fitted_magnon['kappa'] - true_magnon['kappa']) / true_magnon['kappa'] * 100
    
    print(f"  Magnon: omega_m: Fitted = {fitted_magnon['omega_m']/1e9:.3f} GHz, True = {true_magnon['omega_m']/1e9:.3f} GHz, Error = {omega_m_error:.2f}%")
    print(f"          gamma: Fitted = {fitted_magnon['gamma']/1e6:.3f} MHz, True = {true_magnon['gamma']/1e6:.3f} MHz, Error = {gamma_error:.2f}%")
    print(f"          kappa: Fitted = {fitted_magnon['kappa']/1e6:.3f} MHz, True = {true_magnon['kappa']/1e6:.3f} MHz, Error = {kappa_error:.2f}%")
    
    # Plot the comparison
    calculator.compare_s_parameters(measured_s_params, fitted_s_params, param_names=['S11', 'S21'])

if __name__ == "__main__":
    # Run the examples
    example_fit_magnon_parameters()
    example_fit_complex_network()
