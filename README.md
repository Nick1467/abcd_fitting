# ABCD Matrix Fitting

This project provides tools for fitting microwave network parameters using ABCD matrices. It allows you to:

1. Fit transmission line lengths given a cable arrangement and characteristic impedance (Z0)
2. Fit RLC circuit parameters (resistance, inductance, capacitance)
3. Fit magnon parameters (resonance frequency, damping rates)
4. Fit any combination of the above parameters

## Overview

The code is based on the ABCD matrix formalism for microwave networks. It calculates S-parameters from ABCD matrices and uses optimization techniques to fit model parameters to match measured S-parameter data.

## Files

- `ABCD_fitting.py`: The main library containing the `MicrowaveNetworkCalculator` class with all the fitting functionality
- `fitting_example.py`: Examples showing how to use the fitting functionality for different scenarios

## How to Use

### Basic Usage

1. Create a `MicrowaveNetworkCalculator` instance with your frequency range
2. Define your network structure as a list of dictionaries
3. Use one of the fitting methods to fit parameters to your measured data
4. Compare the fitted results with your measured data

### Network Definition

Networks are defined as a list of dictionaries, where each dictionary represents a network element:

```python
network = [
    {'type': 'tline', 'z0': 50, 'length': 0.01},  # 10mm transmission line with Z0=50Ω
    {'type': 'series_rlc', 'r': 10, 'l': 1e-9, 'c': 1e-12},  # Series RLC circuit
    {'type': 'magnon', 'omega_m': 5e9, 'gamma': 1e7, 'kappa': 1e8},  # Magnon element
]
```

### Supported Element Types

1. **Transmission Line** (`'tline'`):
   - `'z0'`: Characteristic impedance in ohms
   - `'length'`: Physical length in meters

2. **Series RLC** (`'series_rlc'`):
   - `'r'`: Resistance in ohms
   - `'l'`: Inductance in henries
   - `'c'`: Capacitance in farads

3. **Parallel RLC** (`'parallel_rlc'`):
   - `'r'`: Resistance in ohms
   - `'l'`: Inductance in henries
   - `'c'`: Capacitance in farads

4. **Magnon Element** (`'magnon'`):
   - `'omega_m'`: Resonance frequency in Hz
   - `'gamma'`: Intrinsic damping rate in Hz
   - `'kappa'`: External damping rate in Hz

5. **Lumped Element** (`'lumped'`):
   - `'element_type'`: Type of element ('R', 'L', 'C', 'R_parallel', 'L_parallel', 'C_parallel')
   - `'value'`: Element value (ohms for R, henries for L, farads for C)

### Fitting Methods

The `MicrowaveNetworkCalculator` class provides several methods for fitting:

1. **`fit_transmission_line_lengths`**: Fits only the lengths of transmission lines
2. **`fit_rlc_parameters`**: Fits RLC parameters (R, L, C) for specified elements
3. **`fit_magnon_parameters`**: Fits magnon parameters (omega_m, gamma, kappa) for specified elements
4. **`fit_all_parameters`**: Fits all parameters in the network
5. **`fit_network`**: Low-level fitting function that allows you to specify exactly which parameters to fit

## Examples

### Fitting Transmission Line Lengths

```python
# Create calculator with your frequency range
calculator = MicrowaveNetworkCalculator(frequencies)

# Define your network template with initial guesses
network_template = [
    {'type': 'tline', 'z0': 75, 'length': 0.010},  # Initial guess: 10mm
    {'type': 'tline', 'z0': 50, 'length': 0.020},  # Initial guess: 20mm
]

# Perform the fitting
fitted_network = calculator.fit_transmission_line_lengths(
    measured_s_params,  # Your measured S-parameter data
    network_template,
    verbose=1
)

# Calculate S-parameters for the fitted network
_, fitted_s_params = calculator.calculate_network(fitted_network)

# Compare measured and fitted S-parameters
calculator.compare_s_parameters(measured_s_params, fitted_s_params)
```

### Fitting Magnon Parameters

```python
# Define your network template
network_template = [
    {'type': 'tline', 'z0': 50, 'length': 0.010},  # Known transmission line
    {'type': 'magnon', 'omega_m': 4.5e9, 'gamma': 1e7, 'kappa': 5e7},  # Initial guess
    {'type': 'tline', 'z0': 50, 'length': 0.010},  # Known transmission line
]

# Perform the fitting
fitted_network = calculator.fit_magnon_parameters(
    measured_s_params,
    network_template,
    magnon_indices=[1],  # Index of the magnon element to fit
    verbose=1
)
```

### Fitting RLC Parameters

```python
# Define your network template
network_template = [
    {'type': 'tline', 'z0': 50, 'length': 0.020},  # Known transmission line
    {'type': 'series_rlc', 'r': 5, 'l': 1e-9, 'c': 1e-12},  # Initial guess
    {'type': 'tline', 'z0': 50, 'length': 0.020},  # Known transmission line
]

# Perform the fitting
fitted_network = calculator.fit_rlc_parameters(
    measured_s_params,
    network_template,
    rlc_indices=[1],  # Index of the RLC element to fit
    verbose=1
)
```

### Fitting All Parameters

```python
# Define your network template with initial guesses
network_template = [
    {'type': 'tline', 'z0': 50, 'length': 0.010},  # Initial guess
    {'type': 'series_rlc', 'r': 10, 'l': 1e-9, 'c': 2e-12},  # Initial guess
    {'type': 'magnon', 'omega_m': 5.5e9, 'gamma': 1e7, 'kappa': 1e8},  # Initial guess
]

# Perform the fitting
fitted_network = calculator.fit_all_parameters(
    measured_s_params,
    network_template,
    verbose=1
)
```

## Working with Real Data

See the `example_fit_real_data()` function in `fitting_example.py` for a template on how to work with real measured data.

## Improved Fitting with Error Control

The fitting functions now include enhanced error control to ensure the error stays below a specified threshold:

```python
# Fit with error control to ensure error is below 5%
fitted_network, final_error = calculator.fit_network(
    measured_s_params,
    network_template,
    params_to_fit,
    bounds=bounds,
    method='trf',               # Optimization method ('trf', 'dogbox', 'lm', or 'Nelder-Mead')
    max_error_percent=5.0,      # Maximum allowed error percentage (default: 5.0%)
    max_iterations=5,           # Maximum number of optimization iterations to try
    verbose=1                   # Verbosity level
)
```

The improved fitting algorithm:
1. Uses percentage error instead of absolute error
2. Tries multiple optimization methods if the first one doesn't achieve the desired error threshold
3. Perturbs parameters between iterations to help escape local minima
4. Provides detailed reporting of the fitting process

See `improved_fitting_example.py` for examples of using the enhanced fitting functionality.

## Parameter Bounds

The fitting functions use reasonable default bounds for parameters, but you can also specify custom bounds:

```python
bounds = {
    'tline_length': (0.001, 0.5),      # Between 1mm and 50cm
    'magnon_omega_m': (1e9, 20e9),     # 1 GHz to 20 GHz
    'magnon_gamma': (1e6, 1e8),        # 1 MHz to 100 MHz
    'magnon_kappa': (1e6, 1e9)         # 1 MHz to 1 GHz
}

fitted_network, error = calculator.fit_network(
    measured_s_params,
    network_template,
    params_to_fit,
    bounds=bounds
)
```

## Visualization

The library provides functions to visualize S-parameters:

1. **`plot_s_parameters`**: Plots S-parameters (magnitude and phase) for a single dataset
2. **`compare_s_parameters`**: Plots measured vs. fitted S-parameters for comparison

## Dependencies

- NumPy
- SciPy (for optimization)
- Matplotlib (for plotting)
