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

def example_fit_transmission_line_lengths():
    """
    Example of fitting transmission line lengths in a network.
    """
    print("\n=== Example: Fitting Transmission Line Lengths ===")
    
    # Define frequency range
    f_start = 1e9  # 1 GHz
    f_stop = 10e9  # 10 GHz
    f_points = 201
    frequencies = np.linspace(f_start, f_stop, f_points)
    
    # Define the true network (this will be our reference)
    true_network = [
        {'type': 'tline', 'z0': 75, 'length': 0.015},   # 15mm transmission line
        {'type': 'tline', 'z0': 50, 'length': 0.025},   # 25mm transmission line
        {'type': 'tline', 'z0': 75, 'length': 0.020},   # 20mm transmission line
    ]
    
    # Generate synthetic "measured" data with some noise
    measured_s_params = generate_synthetic_data(frequencies, true_network, noise_level=0.03)
    
    # Create a template network with initial guesses for the lengths
    template_network = [
        {'type': 'tline', 'z0': 75, 'length': 0.010},   # Initial guess: 10mm
        {'type': 'tline', 'z0': 50, 'length': 0.020},   # Initial guess: 20mm
        {'type': 'tline', 'z0': 75, 'length': 0.015},   # Initial guess: 15mm
    ]
    
    # Create calculator and perform the fitting
    calculator = MicrowaveNetworkCalculator(frequencies)
    fitted_network = calculator.fit_transmission_line_lengths(
        measured_s_params, template_network, verbose=1
    )
    
    # Calculate S-parameters for the fitted network
    _, fitted_s_params = calculator.calculate_network(fitted_network)
    
    # Compare the results
    print("\nTrue vs. Fitted Transmission Line Lengths:")
    for i in range(len(true_network)):
        true_length = true_network[i]['length']
        fitted_length = fitted_network[i]['length']
        error_percent = abs(true_length - fitted_length) / true_length * 100
        print(f"  Line {i+1}: True = {true_length*1000:.2f} mm, Fitted = {fitted_length*1000:.2f} mm, Error = {error_percent:.2f}%")
    
    # Plot the comparison
    calculator.compare_s_parameters(measured_s_params, fitted_s_params, param_names=['S11', 'S21'])

def example_fit_magnon_parameters():
    """
    Example of fitting magnon parameters in a network.
    """
    print("\n=== Example: Fitting Magnon Parameters ===")
    
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
    measured_s_params = generate_synthetic_data(frequencies, true_network, noise_level=0.02)
    
    # Create a template network with initial guesses for the parameters
    template_network = [
        {'type': 'tline', 'z0': 50, 'length': 0.010},   # Known transmission line
        {'type': 'magnon', 'omega_m': 4.5e9, 'gamma': 1e7, 'kappa': 5e7},  # Initial guess for magnon
        {'type': 'tline', 'z0': 50, 'length': 0.010},   # Known transmission line
    ]
    
    # Create calculator and perform the fitting
    calculator = MicrowaveNetworkCalculator(frequencies)
    fitted_network = calculator.fit_magnon_parameters(
        measured_s_params, template_network, magnon_indices=[1], verbose=1
    )
    
    # Calculate S-parameters for the fitted network
    _, fitted_s_params = calculator.calculate_network(fitted_network)
    
    # Compare the results
    print("\nTrue vs. Fitted Magnon Parameters:")
    true_magnon = true_network[1]
    fitted_magnon = fitted_network[1]
    
    print(f"  omega_m: True = {true_magnon['omega_m']/1e9:.3f} GHz, Fitted = {fitted_magnon['omega_m']/1e9:.3f} GHz")
    print(f"  gamma: True = {true_magnon['gamma']/1e6:.3f} MHz, Fitted = {fitted_magnon['gamma']/1e6:.3f} MHz")
    print(f"  kappa: True = {true_magnon['kappa']/1e6:.3f} MHz, Fitted = {fitted_magnon['kappa']/1e6:.3f} MHz")
    
    # Plot the comparison
    calculator.compare_s_parameters(measured_s_params, fitted_s_params, param_names=['S11', 'S21'])

def example_fit_rlc_parameters():
    """
    Example of fitting RLC parameters in a network.
    """
    print("\n=== Example: Fitting RLC Parameters ===")
    
    # Define frequency range
    f_start = 0.5e9  # 0.5 GHz
    f_stop = 5e9     # 5 GHz
    f_points = 201
    frequencies = np.linspace(f_start, f_stop, f_points)
    
    # Define the true network (this will be our reference)
    true_network = [
        {'type': 'tline', 'z0': 50, 'length': 0.020},   # 20mm transmission line
        {'type': 'series_rlc', 'r': 10, 'l': 5e-9, 'c': 2e-12},  # Series RLC
        {'type': 'tline', 'z0': 50, 'length': 0.020},   # 20mm transmission line
    ]
    
    # Generate synthetic "measured" data with some noise
    measured_s_params = generate_synthetic_data(frequencies, true_network, noise_level=0.03)
    
    # Create a template network with initial guesses for the parameters
    template_network = [
        {'type': 'tline', 'z0': 50, 'length': 0.020},   # Known transmission line
        {'type': 'series_rlc', 'r': 5, 'l': 1e-9, 'c': 1e-12},  # Initial guess for RLC
        {'type': 'tline', 'z0': 50, 'length': 0.020},   # Known transmission line
    ]
    
    # Create calculator and perform the fitting
    calculator = MicrowaveNetworkCalculator(frequencies)
    fitted_network = calculator.fit_rlc_parameters(
        measured_s_params, template_network, rlc_indices=[1], verbose=1
    )
    
    # Calculate S-parameters for the fitted network
    _, fitted_s_params = calculator.calculate_network(fitted_network)
    
    # Compare the results
    print("\nTrue vs. Fitted RLC Parameters:")
    true_rlc = true_network[1]
    fitted_rlc = fitted_network[1]
    
    print(f"  R: True = {true_rlc['r']:.3f} Ω, Fitted = {fitted_rlc['r']:.3f} Ω")
    print(f"  L: True = {true_rlc['l']*1e9:.3f} nH, Fitted = {fitted_rlc['l']*1e9:.3f} nH")
    print(f"  C: True = {true_rlc['c']*1e12:.3f} pF, Fitted = {fitted_rlc['c']*1e12:.3f} pF")
    
    # Plot the comparison
    calculator.compare_s_parameters(measured_s_params, fitted_s_params, param_names=['S11', 'S21'])

def example_fit_all_parameters():
    """
    Example of fitting all parameters in a complex network.
    """
    print("\n=== Example: Fitting All Parameters in a Complex Network ===")
    
    # Define frequency range
    f_start = 1e9   # 1 GHz
    f_stop = 10e9   # 10 GHz
    f_points = 401
    frequencies = np.linspace(f_start, f_stop, f_points)
    
    # Define the true network (this will be our reference)
    true_network = [
        {'type': 'tline', 'z0': 50, 'length': 0.015},   # 15mm transmission line
        {'type': 'series_rlc', 'r': 5, 'l': 2e-9, 'c': 1e-12},  # Series RLC
        {'type': 'tline', 'z0': 75, 'length': 0.025},   # 25mm transmission line
        {'type': 'magnon', 'omega_m': 6e9, 'gamma': 8e6, 'kappa': 1.2e8},  # Magnon element
        {'type': 'tline', 'z0': 50, 'length': 0.020},   # 20mm transmission line
    ]
    
    # Generate synthetic "measured" data with some noise
    measured_s_params = generate_synthetic_data(frequencies, true_network, noise_level=0.02)
    
    # Create a template network with initial guesses for the parameters
    template_network = [
        {'type': 'tline', 'z0': 50, 'length': 0.010},   # Initial guess
        {'type': 'series_rlc', 'r': 10, 'l': 1e-9, 'c': 2e-12},  # Initial guess
        {'type': 'tline', 'z0': 75, 'length': 0.020},   # Initial guess
        {'type': 'magnon', 'omega_m': 5.5e9, 'gamma': 1e7, 'kappa': 1e8},  # Initial guess
        {'type': 'tline', 'z0': 50, 'length': 0.015},   # Initial guess
    ]
    
    # Create calculator and perform the fitting
    calculator = MicrowaveNetworkCalculator(frequencies)
    fitted_network = calculator.fit_all_parameters(
        measured_s_params, template_network, verbose=1
    )
    
    # Calculate S-parameters for the fitted network
    _, fitted_s_params = calculator.calculate_network(fitted_network)
    
    # Compare the results
    print("\nTrue vs. Fitted Parameters:")
    
    # Transmission lines
    for i in [0, 2, 4]:
        true_length = true_network[i]['length']
        fitted_length = fitted_network[i]['length']
        error_percent = abs(true_length - fitted_length) / true_length * 100
        print(f"  Line {i+1}: True = {true_length*1000:.2f} mm, Fitted = {fitted_length*1000:.2f} mm, Error = {error_percent:.2f}%")
    
    # RLC
    true_rlc = true_network[1]
    fitted_rlc = fitted_network[1]
    print(f"  RLC: R: True = {true_rlc['r']:.3f} Ω, Fitted = {fitted_rlc['r']:.3f} Ω")
    print(f"       L: True = {true_rlc['l']*1e9:.3f} nH, Fitted = {fitted_rlc['l']*1e9:.3f} nH")
    print(f"       C: True = {true_rlc['c']*1e12:.3f} pF, Fitted = {fitted_rlc['c']*1e12:.3f} pF")
    
    # Magnon
    true_magnon = true_network[3]
    fitted_magnon = fitted_network[3]
    print(f"  Magnon: omega_m: True = {true_magnon['omega_m']/1e9:.3f} GHz, Fitted = {fitted_magnon['omega_m']/1e9:.3f} GHz")
    print(f"          gamma: True = {true_magnon['gamma']/1e6:.3f} MHz, Fitted = {fitted_magnon['gamma']/1e6:.3f} MHz")
    print(f"          kappa: True = {true_magnon['kappa']/1e6:.3f} MHz, Fitted = {fitted_magnon['kappa']/1e6:.3f} MHz")
    
    # Plot the comparison
    calculator.compare_s_parameters(measured_s_params, fitted_s_params, param_names=['S11', 'S21'])

def example_fit_real_data():
    """
    Example of how to fit real measured data.
    This is a template for users to adapt to their own data.
    """
    print("\n=== Template for Fitting Real Measured Data ===")
    
    # Step 1: Load your measured S-parameter data
    # This is just a placeholder - replace with your actual data loading code
    # For example, you might load from a Touchstone file or CSV
    
    # Example of loading frequency and S-parameter data (replace with your actual data)
    # frequencies = np.loadtxt('your_frequency_data.txt')  # in Hz
    # s11_real = np.loadtxt('your_s11_real_data.txt')
    # s11_imag = np.loadtxt('your_s11_imag_data.txt')
    # s21_real = np.loadtxt('your_s21_real_data.txt')
    # s21_imag = np.loadtxt('your_s21_imag_data.txt')
    
    # For this example, we'll generate synthetic data
    f_start = 1e9
    f_stop = 10e9
    f_points = 201
    frequencies = np.linspace(f_start, f_stop, f_points)
    
    # Define a network to generate synthetic "measured" data
    true_network = [
        {'type': 'tline', 'z0': 50, 'length': 0.018},
        {'type': 'tline', 'z0': 75, 'length': 0.022},
        {'type': 'magnon', 'omega_m': 5.5e9, 'gamma': 7e6, 'kappa': 9e7},
        {'type': 'tline', 'z0': 50, 'length': 0.015},
    ]
    
    # Generate synthetic data (replace with your real data)
    measured_s_params = generate_synthetic_data(frequencies, true_network, noise_level=0.03)
    
    # Step 2: Create your network template with initial parameter guesses
    # This is where you define the structure of your network and initial guesses for parameters
    network_template = [
        {'type': 'tline', 'z0': 50, 'length': 0.015},  # Initial guess
        {'type': 'tline', 'z0': 75, 'length': 0.020},  # Initial guess
        {'type': 'magnon', 'omega_m': 5e9, 'gamma': 1e7, 'kappa': 1e8},  # Initial guess
        {'type': 'tline', 'z0': 50, 'length': 0.010},  # Initial guess
    ]
    
    # Step 3: Create the calculator with your frequency range
    calculator = MicrowaveNetworkCalculator(frequencies)
    
    # Step 4: Perform the fitting
    # You can choose to fit all parameters or specific ones
    
    # Option 1: Fit only transmission line lengths
    print("\nFitting only transmission line lengths...")
    fitted_network_tlines = calculator.fit_transmission_line_lengths(
        measured_s_params, network_template, verbose=1
    )
    
    # Option 2: Fit only magnon parameters
    print("\nFitting only magnon parameters...")
    fitted_network_magnon = calculator.fit_magnon_parameters(
        measured_s_params, network_template, magnon_indices=[2], verbose=1
    )
    
    # Option 3: Fit all parameters
    print("\nFitting all parameters...")
    fitted_network_all = calculator.fit_all_parameters(
        measured_s_params, network_template, verbose=1
    )
    
    # Step 5: Calculate S-parameters for the fitted network
    _, fitted_s_params = calculator.calculate_network(fitted_network_all)
    
    # Step 6: Compare the measured and fitted S-parameters
    calculator.compare_s_parameters(measured_s_params, fitted_s_params, param_names=['S11', 'S21'])
    
    # Step 7: Print the fitted parameters
    print("\nFitted Parameters:")
    for i, element in enumerate(fitted_network_all):
        element_type = element['type']
        if element_type == 'tline':
            print(f"  Transmission Line {i+1}: length = {element['length']*1000:.3f} mm, Z0 = {element['z0']:.1f} Ω")
        elif element_type == 'magnon':
            print(f"  Magnon {i+1}: omega_m = {element['omega_m']/1e9:.3f} GHz, gamma = {element['gamma']/1e6:.3f} MHz, kappa = {element['kappa']/1e6:.3f} MHz")
        elif element_type in ['series_rlc', 'parallel_rlc']:
            print(f"  {element_type.capitalize()} {i+1}: R = {element['r']:.3f} Ω, L = {element['l']*1e9:.3f} nH, C = {element['c']*1e12:.3f} pF")

if __name__ == "__main__":
    # Run the examples
    example_fit_transmission_line_lengths()
    example_fit_magnon_parameters()
    example_fit_rlc_parameters()
    example_fit_all_parameters()
    example_fit_real_data()
