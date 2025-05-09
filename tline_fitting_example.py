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

def example_fit_single_tline():
    """
    Example of fitting a single transmission line length.
    """
    print("\n=== Example: Fitting a Single Transmission Line ===")
    
    # Define frequency range
    f_start = 1e9  # 1 GHz
    f_stop = 10e9  # 10 GHz
    f_points = 201
    frequencies = np.linspace(f_start, f_stop, f_points)
    
    # Define the true network (this will be our reference)
    true_network = [
        {'type': 'tline', 'z0': 50, 'length': 0.025},   # 25mm transmission line
    ]
    
    # Generate synthetic "measured" data with some noise
    measured_s_params = generate_synthetic_data(frequencies, true_network, noise_level=0.02)
    
    # Create a template network with initial guess for the length
    template_network = [
        {'type': 'tline', 'z0': 50, 'length': 0.015},   # Initial guess: 15mm
    ]
    
    # Create calculator
    calculator = MicrowaveNetworkCalculator(frequencies)
    
    print("\nTrue transmission line parameters:")
    print(f"  Length = {true_network[0]['length']*1000:.2f} mm")
    print(f"  Z0 = {true_network[0]['z0']:.1f} Ω")
    
    print("\nInitial guess:")
    print(f"  Length = {template_network[0]['length']*1000:.2f} mm")
    print(f"  Z0 = {template_network[0]['z0']:.1f} Ω")
    
    # Perform the fitting with 5% error threshold
    print("\n--- Fitting transmission line length with 5% error threshold ---")
    
    # Set up parameters to fit
    params_to_fit = {0: ['length']}
    
    # Set reasonable bounds for transmission line length
    bounds = {
        'tline_length': (0.001, 0.1)  # Between 1mm and 10cm
    }
    
    # Perform the fitting
    fitted_network, final_error = calculator.fit_network(
        measured_s_params, 
        template_network, 
        params_to_fit, 
        bounds=bounds,
        method='trf',
        max_error_percent=5.0,
        max_iterations=3,
        verbose=1
    )
    
    # Calculate S-parameters for the fitted network
    _, fitted_s_params = calculator.calculate_network(fitted_network)
    
    # Compare the results
    true_length = true_network[0]['length']
    fitted_length = fitted_network[0]['length']
    error_percent = abs(true_length - fitted_length) / true_length * 100
    
    print("\nFitted vs. True Parameters:")
    print(f"  Length: Fitted = {fitted_length*1000:.2f} mm, True = {true_length*1000:.2f} mm, Error = {error_percent:.2f}%")
    
    # Plot the comparison
    calculator.compare_s_parameters(measured_s_params, fitted_s_params, param_names=['S11', 'S21'])

def example_fit_multiple_tlines():
    """
    Example of fitting multiple transmission line lengths in a cascade.
    """
    print("\n=== Example: Fitting Multiple Transmission Lines in Cascade ===")
    
    # Define frequency range
    f_start = 1e9  # 1 GHz
    f_stop = 10e9  # 10 GHz
    f_points = 201
    frequencies = np.linspace(f_start, f_stop, f_points)
    
    # Define the true network (this will be our reference)
    true_network = [
        {'type': 'tline', 'z0': 50, 'length': 0.015},   # 15mm transmission line
        {'type': 'tline', 'z0': 75, 'length': 0.025},   # 25mm transmission line
        {'type': 'tline', 'z0': 50, 'length': 0.020},   # 20mm transmission line
    ]
    
    # Generate synthetic "measured" data with some noise
    measured_s_params = generate_synthetic_data(frequencies, true_network, noise_level=0.02)
    
    # Create a template network with initial guesses for the lengths
    template_network = [
        {'type': 'tline', 'z0': 50, 'length': 0.010},   # Initial guess: 10mm
        {'type': 'tline', 'z0': 75, 'length': 0.020},   # Initial guess: 20mm
        {'type': 'tline', 'z0': 50, 'length': 0.015},   # Initial guess: 15mm
    ]
    
    # Create calculator
    calculator = MicrowaveNetworkCalculator(frequencies)
    
    print("\nTrue transmission line parameters:")
    for i, tline in enumerate(true_network):
        print(f"  Line {i+1}: Length = {tline['length']*1000:.2f} mm, Z0 = {tline['z0']:.1f} Ω")
    
    print("\nInitial guesses:")
    for i, tline in enumerate(template_network):
        print(f"  Line {i+1}: Length = {tline['length']*1000:.2f} mm, Z0 = {tline['z0']:.1f} Ω")
    
    # Perform the fitting with 5% error threshold
    print("\n--- Fitting transmission line lengths with 5% error threshold ---")
    
    # Set up parameters to fit
    params_to_fit = {
        0: ['length'],
        1: ['length'],
        2: ['length']
    }
    
    # Set reasonable bounds for transmission line lengths
    bounds = {
        'tline_length': (0.001, 0.1)  # Between 1mm and 10cm
    }
    
    # Perform the fitting
    fitted_network, final_error = calculator.fit_network(
        measured_s_params, 
        template_network, 
        params_to_fit, 
        bounds=bounds,
        method='trf',
        max_error_percent=5.0,
        max_iterations=3,
        verbose=1
    )
    
    # Calculate S-parameters for the fitted network
    _, fitted_s_params = calculator.calculate_network(fitted_network)
    
    # Compare the results
    print("\nFitted vs. True Parameters:")
    for i in range(len(true_network)):
        true_length = true_network[i]['length']
        fitted_length = fitted_network[i]['length']
        error_percent = abs(true_length - fitted_length) / true_length * 100
        print(f"  Line {i+1}: Fitted = {fitted_length*1000:.2f} mm, True = {true_length*1000:.2f} mm, Error = {error_percent:.2f}%")
    
    # Plot the comparison
    calculator.compare_s_parameters(measured_s_params, fitted_s_params, param_names=['S11', 'S21'])

def example_fit_tline_with_different_z0():
    """
    Example of fitting transmission lines with different characteristic impedances (Z0).
    """
    print("\n=== Example: Fitting Transmission Lines with Different Z0 Values ===")
    
    # Define frequency range
    f_start = 1e9  # 1 GHz
    f_stop = 10e9  # 10 GHz
    f_points = 201
    frequencies = np.linspace(f_start, f_stop, f_points)
    
    # Define the true network (this will be our reference)
    true_network = [
        {'type': 'tline', 'z0': 50, 'length': 0.015},   # 15mm transmission line with Z0=50Ω
        {'type': 'tline', 'z0': 75, 'length': 0.025},   # 25mm transmission line with Z0=75Ω
        {'type': 'tline', 'z0': 100, 'length': 0.020},  # 20mm transmission line with Z0=100Ω
    ]
    
    # Generate synthetic "measured" data with some noise
    measured_s_params = generate_synthetic_data(frequencies, true_network, noise_level=0.02)
    
    # Create a template network with initial guesses for the lengths
    # Note: We assume Z0 values are known and only fit the lengths
    template_network = [
        {'type': 'tline', 'z0': 50, 'length': 0.010},   # Initial guess: 10mm
        {'type': 'tline', 'z0': 75, 'length': 0.020},   # Initial guess: 20mm
        {'type': 'tline', 'z0': 100, 'length': 0.015},  # Initial guess: 15mm
    ]
    
    # Create calculator
    calculator = MicrowaveNetworkCalculator(frequencies)
    
    print("\nTrue transmission line parameters:")
    for i, tline in enumerate(true_network):
        print(f"  Line {i+1}: Length = {tline['length']*1000:.2f} mm, Z0 = {tline['z0']:.1f} Ω")
    
    print("\nInitial guesses:")
    for i, tline in enumerate(template_network):
        print(f"  Line {i+1}: Length = {tline['length']*1000:.2f} mm, Z0 = {tline['z0']:.1f} Ω")
    
    # Perform the fitting with 5% error threshold
    print("\n--- Fitting transmission line lengths with 5% error threshold ---")
    
    # Set up parameters to fit
    params_to_fit = {
        0: ['length'],
        1: ['length'],
        2: ['length']
    }
    
    # Set reasonable bounds for transmission line lengths
    bounds = {
        'tline_length': (0.001, 0.1)  # Between 1mm and 10cm
    }
    
    # Perform the fitting
    fitted_network, final_error = calculator.fit_network(
        measured_s_params, 
        template_network, 
        params_to_fit, 
        bounds=bounds,
        method='trf',
        max_error_percent=5.0,
        max_iterations=3,
        verbose=1
    )
    
    # Calculate S-parameters for the fitted network
    _, fitted_s_params = calculator.calculate_network(fitted_network)
    
    # Compare the results
    print("\nFitted vs. True Parameters:")
    for i in range(len(true_network)):
        true_length = true_network[i]['length']
        fitted_length = fitted_network[i]['length']
        error_percent = abs(true_length - fitted_length) / true_length * 100
        print(f"  Line {i+1}: Fitted = {fitted_length*1000:.2f} mm, True = {true_length*1000:.2f} mm, Z0 = {fitted_network[i]['z0']:.1f} Ω, Error = {error_percent:.2f}%")
    
    # Plot the comparison
    calculator.compare_s_parameters(measured_s_params, fitted_s_params, param_names=['S11', 'S21'])

def example_fit_tline_with_termination():
    """
    Example of fitting transmission lines with different terminations.
    """
    print("\n=== Example: Fitting Transmission Lines with Different Terminations ===")
    
    # Define frequency range
    f_start = 1e9  # 1 GHz
    f_stop = 10e9  # 10 GHz
    f_points = 201
    frequencies = np.linspace(f_start, f_stop, f_points)
    
    # Create calculator
    calculator = MicrowaveNetworkCalculator(frequencies)
    
    # Define the true network (this will be our reference)
    true_network = [
        {'type': 'tline', 'z0': 50, 'length': 0.015},   # 15mm transmission line
        {'type': 'tline', 'z0': 75, 'length': 0.025},   # 25mm transmission line
    ]
    
    # Try different terminations
    terminations = [None, 'open', 'short']
    termination_names = ['Matched (50Ω)', 'Open', 'Short']
    
    for term_idx, termination in enumerate(terminations):
        print(f"\n--- Termination: {termination_names[term_idx]} ---")
        
        # Generate synthetic "measured" data with some noise
        measured_s_params = generate_synthetic_data(frequencies, true_network, noise_level=0.02, termination=termination)
        
        # Create a template network with initial guesses for the lengths
        template_network = [
            {'type': 'tline', 'z0': 50, 'length': 0.010},   # Initial guess: 10mm
            {'type': 'tline', 'z0': 75, 'length': 0.020},   # Initial guess: 20mm
        ]
        
        print("\nTrue transmission line parameters:")
        for i, tline in enumerate(true_network):
            print(f"  Line {i+1}: Length = {tline['length']*1000:.2f} mm, Z0 = {tline['z0']:.1f} Ω")
        
        print("\nInitial guesses:")
        for i, tline in enumerate(template_network):
            print(f"  Line {i+1}: Length = {tline['length']*1000:.2f} mm, Z0 = {tline['z0']:.1f} Ω")
        
        # Perform the fitting with 5% error threshold
        print(f"\nFitting transmission line lengths with {termination_names[term_idx]} termination...")
        
        # Set up parameters to fit
        params_to_fit = {
            0: ['length'],
            1: ['length']
        }
        
        # Set reasonable bounds for transmission line lengths
        bounds = {
            'tline_length': (0.001, 0.1)  # Between 1mm and 10cm
        }
        
        # Perform the fitting
        fitted_network, final_error = calculator.fit_network(
            measured_s_params, 
            template_network, 
            params_to_fit, 
            termination=termination,
            bounds=bounds,
            method='trf',
            max_error_percent=5.0,
            max_iterations=3,
            verbose=1
        )
        
        # Calculate S-parameters for the fitted network
        _, fitted_s_params = calculator.calculate_network(fitted_network, termination=termination)
        
        # Compare the results
        print("\nFitted vs. True Parameters:")
        for i in range(len(true_network)):
            true_length = true_network[i]['length']
            fitted_length = fitted_network[i]['length']
            error_percent = abs(true_length - fitted_length) / true_length * 100
            print(f"  Line {i+1}: Fitted = {fitted_length*1000:.2f} mm, True = {true_length*1000:.2f} mm, Error = {error_percent:.2f}%")
        
        # Plot the comparison
        if term_idx == 0:  # Only plot for the matched termination case
            calculator.compare_s_parameters(measured_s_params, fitted_s_params, param_names=['S11', 'S21'])

if __name__ == "__main__":
    # Run the examples
    example_fit_single_tline()
    example_fit_multiple_tlines()
    example_fit_tline_with_different_z0()
    example_fit_tline_with_termination()
