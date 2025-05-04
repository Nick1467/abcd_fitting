import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Union, Tuple
import cmath
from scipy.optimize import minimize, least_squares
import copy

class MicrowaveNetworkCalculator:
    def __init__(self, frequency_range: np.ndarray, z0: float = 50.0):
        """
        Initialize the microwave network calculator.
        
        Args:
            frequency_range: Array of frequencies in Hz at which to evaluate the network
            z0: Characteristic impedance of the system (default: 50 ohms)
        """
        self.frequency_range = frequency_range
        self.z0 = z0
        self.omega = 2 * np.pi * frequency_range
    
    def transmission_line_abcd(self, z_line: float, length: float, freq: float) -> np.ndarray:
        """
        Calculate ABCD matrix for a transmission line segment.
        
        Args:
            z_line: Characteristic impedance of the line in ohms
            length: Physical length of the line in meters
            freq: Frequency in Hz
        
        Returns:
            2x2 ABCD matrix as numpy array
        """
        # Calculate propagation constant (assuming lossless line)
        c = 3e7  # Speed of light in m/s
        beta = 2 * np.pi * freq / c
        
        # Calculate ABCD parameters
        A = np.cos(beta * length)
        B = 1j * z_line * np.sin(beta * length)
        C = 1j * np.sin(beta * length) / z_line
        D = np.cos(beta * length)
        
        return np.array([[A, B], [C, D]])
    
    def rlc_series_abcd(self, r: float, l: float, c: float, freq: float) -> np.ndarray:
        """
        Calculate ABCD matrix for a series RLC resonator.
        
        Args:
            r: Resistance in ohms
            l: Inductance in henries
            c: Capacitance in farads
            freq: Frequency in Hz
        
        Returns:
            2x2 ABCD matrix as numpy array
        """
        omega = 2 * np.pi * freq
        
        # Calculate impedance of the series RLC
        if c == 0:  # Handle series RL case
            z = r + 1j * omega * l
        else:
            z = r + 1j * omega * l + 1 / (1j * omega * c)
        
        # ABCD matrix for series impedance
        A = 1
        B = z
        C = 0
        D = 1
        
        return np.array([[A, B], [C, D]])
    
    def rlc_parallel_abcd(self, r: float, l: float, c: float, freq: float) -> np.ndarray:
        """
        Calculate ABCD matrix for a parallel RLC resonator.
        
        Args:
            r: Resistance in ohms
            l: Inductance in henries
            c: Capacitance in farads
            freq: Frequency in Hz
        
        Returns:
            2x2 ABCD matrix as numpy array
        """
        omega = 2 * np.pi * freq
        
        # Calculate admittance of the parallel RLC
        y_r = 1/r if r != 0 else 0
        y_l = 1/(1j * omega * l) if l != 0 else 0
        y_c = 1j * omega * c if c != 0 else 0
        
        y = y_r + y_l + y_c
        
        # ABCD matrix for parallel admittance
        A = 1
        B = 0
        C = y
        D = 1
        
        return np.array([[A, B], [C, D]])
    
    def magnon_element_abcd(self, omega_m: float, gamma: float, kappa: float, freq: float, Z0 = 50 ) -> np.array:
        """
        Calculate ABCD matrix for a magnon.
        
        Args:
            omega_m : resonance frequency
            gamma : disappative (intrinsic damping) rate 
            kappa : radiative(external) duamping rate
            
        
        Returns:
            2x2 ABCD matrix as numpy array
        """

        

        omega =  2 * np.pi*freq
        omega_m =2 * np.pi*  omega_m

        L = Z0 / (4*kappa)
        R = 2*L*gamma
        C = 1/(omega_m**2 *L)

        Z = R + 1j*omega*L + 1/(1j* omega*C)

        A = 1
        B = 0
        C = 1/Z
        D = 1

        return np.array([[A, B], [C, D]])



    def lumped_element_abcd(self, element_type: str, value: float, freq: float) -> np.ndarray:
        """
        Calculate ABCD matrix for common lumped elements.
        
        Args:
            element_type: Type of element ('R', 'L', 'C', 'R_parallel', 'L_parallel', 'C_parallel')
            value: Element value (ohms for R, henries for L, farads for C)
            freq: Frequency in Hz
        
        Returns:
            2x2 ABCD matrix as numpy array
        """
        omega = 2 * np.pi * freq
        
        if element_type == 'R':
            # Series resistor
            return np.array([[1, value], [0, 1]])
        elif element_type == 'L':
            # Series inductor
            return np.array([[1, 1j * omega * value], [0, 1]])
        elif element_type == 'C':
            # Series capacitor
            return np.array([[1, 1 / (1j * omega * value)], [0, 1]])
        elif element_type == 'R_parallel':
            # Parallel resistor
            return np.array([[1, 0], [1 / value, 1]])
        elif element_type == 'L_parallel':
            # Parallel inductor
            return np.array([[1, 0], [1 / (1j * omega * value), 1]])
        elif element_type == 'C_parallel':
            # Parallel capacitor
            return np.array([[1, 0], [1j * omega * value, 1]])
        else:
            raise ValueError(f"Unknown element type: {element_type}")
    
    def cascade_abcd(self, abcd_matrices: List[np.ndarray]) -> np.ndarray:
        """
        Calculate the overall ABCD matrix for a cascade of networks.
        
        Args:
            abcd_matrices: List of ABCD matrices
        
        Returns:
            Combined 2x2 ABCD matrix as numpy array
        """
        result = np.identity(2)
        for matrix in abcd_matrices:
            result = np.matmul(result, matrix)
        return result
    
    def abcd_to_s_parameters(self, abcd: np.ndarray, z0: float = None) -> np.ndarray:
        """
        Convert ABCD matrix to S-parameters.
        
        Args:
            abcd: 2x2 ABCD matrix
            z0: Characteristic impedance (uses self.z0 if None)
        
        Returns:
            2x2 S-parameter matrix as numpy array
        """
        if z0 is None:
            z0 = self.z0
        
        A, B, C, D = abcd[0, 0], abcd[0, 1], abcd[1, 0], abcd[1, 1]
        
        denominator = A + B/z0 + C*z0 + D
        
        S11 = (A + B/z0 - C*z0 - D) / denominator
        S12 = 2 * (A*D - B*C) / denominator
        S21 = 2 / denominator
        S22 = (-A + B/z0 - C*z0 + D) / denominator
        
        return np.array([[S11, S12], [S21, S22]])
    
    def short_circuit_abcd(self) -> np.ndarray:
        """
        Calculate ABCD matrix for a short circuit (ground) termination.
        
        Returns:
            2x2 ABCD matrix as numpy array
        """
        # For a short circuit termination, Z=0
        # This is equivalent to a shunt with Z=0, or Y→∞
        # Mathematically, the ABCD matrix would have B=0, C=∞
        # But we can't directly represent infinity in our calculations
        # Instead, we'll use a very small resistance as an approximation
        very_small_r = 1e-10
        
        # ABCD matrix for a shunt with very small resistance
        return np.array([[1, 0], [1/very_small_r, 1]])
    
    def open_circuit_abcd(self) -> np.ndarray:
        """
        Calculate ABCD matrix for an open circuit termination.
        
        Returns:
            2x2 ABCD matrix as numpy array
        """
        # For an open circuit termination, Z=∞
        # This is equivalent to a shunt with Z=∞, or Y=0
        # The ABCD matrix is:
        return np.array([[1, 0], [0, 1]])
    
    def calculate_network(self, network_elements: List[Dict], termination: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate S-parameters for a network over the frequency range.
        
        Args:
            network_elements: List of dictionaries describing each element
                Each dict should have:
                    'type': 'tline' | 'series_rlc' | 'parallel_rlc' | 'lumped'
                    Plus type-specific parameters:
                        For 'tline': 'z0', 'length'
                        For 'series_rlc': 'r', 'l', 'c'
                        For 'parallel_rlc': 'r', 'l', 'c'
                        For 'lumped': 'element_type', 'value'
            termination: Optional termination type ('short', 'open', or None for matched load)
        
        Returns:
            Tuple of (frequencies, S-parameter matrices)
        """
        s_matrices = []
        
        for freq in self.frequency_range:
            abcd_matrices = []
            
            for element in network_elements:
                element_type = element['type']
                
                if element_type == 'tline':
                    abcd = self.transmission_line_abcd(
                        element['z0'], element['length'], freq
                    )
                elif element_type == 'series_rlc':
                    abcd = self.rlc_series_abcd(
                        element.get('r', 0), element.get('l', 0), element.get('c', 0), freq
                    )
                elif element_type == 'parallel_rlc':
                    abcd = self.rlc_parallel_abcd(
                        element.get('r', 0), element.get('l', 0), element.get('c', 0), freq
                    )
                elif element_type == 'lumped':
                    abcd = self.lumped_element_abcd(
                        element['element_type'], element['value'], freq
                    )
                elif element_type == 'magnon':
                    abcd = self.magnon_element_abcd(
                        element.get('omega_m', 0), element.get('gamma', 0), element.get('kappa', 0), freq
                    )
                else:
                    raise ValueError(f"Unknown element type: {element_type}")
                
                abcd_matrices.append(abcd)
            
            # Calculate overall ABCD matrix
            total_abcd = self.cascade_abcd(abcd_matrices)
            
            # Add termination if specified
            if termination == 'short':
                short_abcd = self.short_circuit_abcd()
                total_abcd = np.matmul(total_abcd, short_abcd)
            elif termination == 'open':
                open_abcd = self.open_circuit_abcd()
                total_abcd = np.matmul(total_abcd, open_abcd)
            
            # Convert to S-parameters
            s_matrix = self.abcd_to_s_parameters(total_abcd)
            s_matrices.append(s_matrix)
        
        return self.frequency_range, np.array(s_matrices)

    def plot_s_parameters(self, s_params: np.ndarray, param_names: List[str] = None, 
                          plot_magnitude: bool = True, plot_phase: bool = True,
                          db_scale: bool = True):
        """
        Plot S-parameters vs frequency.
        
        Args:
            s_params: Array of S-parameter matrices
            param_names: List of S-parameter names to plot (e.g., ['S11', 'S21'])
            plot_magnitude: Whether to plot magnitude
            plot_phase: Whether to plot phase
            db_scale: Whether to use dB scale for magnitude
        """
        if param_names is None:
            param_names = ['S11', 'S21', 'S12', 'S22']
        
        indices = {
            'S11': (0, 0),
            'S12': (0, 1),
            'S21': (1, 0),
            'S22': (1, 1)
        }
        
        freq_ghz = self.frequency_range / 1e9  # Convert to GHz for plotting
        
        plt.figure(figsize=(6, 4))
        
        # Plot magnitude
        if plot_magnitude:
            plt.subplot(2 if plot_phase else 1, 1, 1)
            for name in param_names:
                i, j = indices[name]
                magnitude = np.abs([s[i, j] for s in s_params])
                
                if db_scale:
                    magnitude_db = 20 * np.log10(magnitude)
                    plt.plot(freq_ghz, magnitude_db, label=f"|{name}| (dB)")
                    plt.ylabel('Magnitude (dB)')
                else:
                    plt.plot(freq_ghz, magnitude, label=f"|{name}|")
                    plt.ylabel('Magnitude')
            
            plt.grid(True)
            plt.legend()
            plt.title('S-Parameter Magnitude')
            plt.xlabel('Frequency (GHz)')
        
        # Plot phase
        if plot_phase:
            plt.subplot(2, 1, 2 if plot_magnitude else 1)
            for name in param_names:
                i, j = indices[name]
                phase_deg = np.angle([s[i, j] for s in s_params]) * 180 / np.pi
                plt.plot(freq_ghz, phase_deg, label=f"∠{name} (deg)")
            
            plt.grid(True)
            plt.legend()
            plt.title('S-Parameter Phase')
            plt.xlabel('Frequency (GHz)')
            plt.ylabel('Phase (degrees)')
        
        plt.tight_layout()
        plt.show()

    def compare_s_parameters(self, measured_s_params: np.ndarray, fitted_s_params: np.ndarray, 
                            param_names: List[str] = None, db_scale: bool = True):
        """
        Plot measured vs fitted S-parameters for comparison.
        
        Args:
            measured_s_params: Array of measured S-parameter matrices
            fitted_s_params: Array of fitted S-parameter matrices
            param_names: List of S-parameter names to plot (e.g., ['S11', 'S21'])
            db_scale: Whether to use dB scale for magnitude
        """
        if param_names is None:
            param_names = ['S11', 'S21']
        
        indices = {
            'S11': (0, 0),
            'S12': (0, 1),
            'S21': (1, 0),
            'S22': (1, 1)
        }
        
        freq_ghz = self.frequency_range / 1e9  # Convert to GHz for plotting
        
        # Create subplots for each S-parameter
        fig, axs = plt.subplots(len(param_names), 2, figsize=(12, 4*len(param_names)))
        
        for idx, name in enumerate(param_names):
            i, j = indices[name]
            
            # Get magnitudes
            measured_mag = np.abs([s[i, j] for s in measured_s_params])
            fitted_mag = np.abs([s[i, j] for s in fitted_s_params])
            
            # Get phases
            measured_phase = np.angle([s[i, j] for s in measured_s_params]) * 180 / np.pi
            fitted_phase = np.angle([s[i, j] for s in fitted_s_params]) * 180 / np.pi
            
            # Plot magnitude
            if len(param_names) == 1:
                ax_mag = axs[0]
                ax_phase = axs[1]
            else:
                ax_mag = axs[idx, 0]
                ax_phase = axs[idx, 1]
            
            if db_scale:
                measured_mag_db = 20 * np.log10(measured_mag)
                fitted_mag_db = 20 * np.log10(fitted_mag)
                ax_mag.plot(freq_ghz, measured_mag_db, 'b-', label=f"Measured |{name}| (dB)")
                ax_mag.plot(freq_ghz, fitted_mag_db, 'r--', label=f"Fitted |{name}| (dB)")
                ax_mag.set_ylabel('Magnitude (dB)')
            else:
                ax_mag.plot(freq_ghz, measured_mag, 'b-', label=f"Measured |{name}|")
                ax_mag.plot(freq_ghz, fitted_mag, 'r--', label=f"Fitted |{name}|")
                ax_mag.set_ylabel('Magnitude')
            
            ax_mag.grid(True)
            ax_mag.legend()
            ax_mag.set_title(f'{name} Magnitude')
            ax_mag.set_xlabel('Frequency (GHz)')
            
            # Plot phase
            ax_phase.plot(freq_ghz, measured_phase, 'b-', label=f"Measured ∠{name} (deg)")
            ax_phase.plot(freq_ghz, fitted_phase, 'r--', label=f"Fitted ∠{name} (deg)")
            ax_phase.grid(True)
            ax_phase.legend()
            ax_phase.set_title(f'{name} Phase')
            ax_phase.set_xlabel('Frequency (GHz)')
            ax_phase.set_ylabel('Phase (degrees)')
        
        plt.tight_layout()
        plt.show()

    def fit_network(self, measured_s_params: np.ndarray, network_template: List[Dict], 
                   params_to_fit: Dict[int, List[str]], termination: str = None,
                   bounds: Dict = None, method: str = 'trf', verbose: int = 1) -> Tuple[List[Dict], float]:
        """
        Fit network parameters to match measured S-parameters.
        
        Args:
            measured_s_params: Array of measured S-parameter matrices
            network_template: List of dictionaries describing the network structure
            params_to_fit: Dictionary mapping element indices to lists of parameter names to fit
                           e.g., {0: ['length'], 5: ['omega_m', 'gamma', 'kappa']}
            termination: Optional termination type ('short', 'open', or None for matched load)
            bounds: Dictionary of parameter bounds {param_name: (lower_bound, upper_bound)}
            method: Optimization method ('trf', 'dogbox', or 'lm')
            verbose: Verbosity level (0: silent, 1: progress, 2: detailed)
            
        Returns:
            Tuple of (fitted_network, final_error)
        """
        # Create a deep copy of the network template to avoid modifying the original
        network = copy.deepcopy(network_template)
        
        # Extract parameters to fit and their initial values
        param_names = []
        initial_values = []
        param_mapping = []  # Maps flat parameter index to (element_idx, param_name)
        
        for element_idx, param_list in params_to_fit.items():
            element = network[element_idx]
            for param_name in param_list:
                param_names.append(f"{element['type']}_{element_idx}_{param_name}")
                initial_values.append(element[param_name])
                param_mapping.append((element_idx, param_name))
        
        # Set up bounds if provided
        if bounds:
            lb = []
            ub = []
            for element_idx, param_name in param_mapping:
                element_type = network[element_idx]['type']
                bound_key = f"{element_type}_{param_name}"
                if bound_key in bounds:
                    lb.append(bounds[bound_key][0])
                    ub.append(bounds[bound_key][1])
                else:
                    # Default bounds
                    if param_name == 'length':
                        lb.append(0.001)  # 1mm minimum length
                        ub.append(1.0)    # 1m maximum length
                    elif param_name in ['r', 'l', 'c']:
                        lb.append(1e-12)  # Small positive value
                        ub.append(1e3)    # Large value
                    elif param_name == 'omega_m':
                        lb.append(1e8)    # 100 MHz
                        ub.append(1e11)   # 100 GHz
                    elif param_name in ['gamma', 'kappa']:
                        lb.append(1e5)    # Small damping
                        ub.append(1e9)    # Large damping
                    else:
                        lb.append(1e-6)   # Default lower bound
                        ub.append(1e6)    # Default upper bound
            
            optimization_bounds = (lb, ub)
        else:
            optimization_bounds = (-np.inf, np.inf)
        
        # Define the error function to minimize
        def error_function(params):
            # Update network with current parameter values
            for i, (element_idx, param_name) in enumerate(param_mapping):
                network[element_idx][param_name] = params[i]
            
            # Calculate S-parameters for the current network
            _, calculated_s_params = self.calculate_network(network, termination)
            
            # Calculate error between measured and calculated S-parameters
            error = []
            for i in range(len(self.frequency_range)):
                # Add errors for S11 (magnitude and phase)
                s11_measured = measured_s_params[i][0, 0]
                s11_calculated = calculated_s_params[i][0, 0]
                error.append(np.abs(np.abs(s11_measured) - np.abs(s11_calculated)))
                
                # Add errors for S21 (magnitude and phase)
                s21_measured = measured_s_params[i][1, 0]
                s21_calculated = calculated_s_params[i][1, 0]
                error.append(np.abs(np.abs(s21_measured) - np.abs(s21_calculated)))
                
                # Optionally add S12 and S22 errors if needed
                # error.append(np.abs(np.abs(measured_s_params[i][0, 1]) - np.abs(calculated_s_params[i][0, 1])))
                # error.append(np.abs(np.abs(measured_s_params[i][1, 1]) - np.abs(calculated_s_params[i][1, 1])))
            
            return np.array(error)
        
        # Run the optimization
        if verbose >= 1:
            print(f"Starting optimization with {len(initial_values)} parameters to fit...")
            print(f"Initial parameters: {dict(zip(param_names, initial_values))}")
        
        result = least_squares(
            error_function, 
            initial_values, 
            bounds=optimization_bounds,
            method=method,
            verbose=2 if verbose >= 2 else 0
        )
        
        # Update network with final fitted parameters
        for i, (element_idx, param_name) in enumerate(param_mapping):
            network[element_idx][param_name] = result.x[i]
        
        # Calculate final error
        final_error = np.sum(result.fun**2)
        
        if verbose >= 1:
            print("\nFitting completed!")
            print(f"Final error: {final_error}")
            print("\nFitted parameters:")
            for i, (element_idx, param_name) in enumerate(param_mapping):
                element_type = network[element_idx]['type']
                print(f"  {element_type}_{element_idx}_{param_name}: {result.x[i]}")
        
        return network, final_error

    def fit_transmission_line_lengths(self, measured_s_params: np.ndarray, network_template: List[Dict], 
                                     termination: str = None, verbose: int = 1) -> List[Dict]:
        """
        Fit only the lengths of transmission lines in the network.
        
        Args:
            measured_s_params: Array of measured S-parameter matrices
            network_template: List of dictionaries describing the network structure
            termination: Optional termination type ('short', 'open', or None for matched load)
            verbose: Verbosity level
            
        Returns:
            Fitted network with optimized transmission line lengths
        """
        # Identify all transmission line elements and their indices
        params_to_fit = {}
        for i, element in enumerate(network_template):
            if element['type'] == 'tline':
                params_to_fit[i] = ['length']
        
        # Set reasonable bounds for transmission line lengths
        bounds = {
            'tline_length': (0.001, 0.5)  # Between 1mm and 50cm
        }
        
        # Perform the fitting
        fitted_network, _ = self.fit_network(
            measured_s_params, 
            network_template, 
            params_to_fit, 
            termination, 
            bounds, 
            verbose=verbose
        )
        
        return fitted_network

    def fit_rlc_parameters(self, measured_s_params: np.ndarray, network_template: List[Dict], 
                          rlc_indices: List[int], termination: str = None, verbose: int = 1) -> List[Dict]:
        """
        Fit RLC parameters in the network.
        
        Args:
            measured_s_params: Array of measured S-parameter matrices
            network_template: List of dictionaries describing the network structure
            rlc_indices: Indices of RLC elements to fit
            termination: Optional termination type ('short', 'open', or None for matched load)
            verbose: Verbosity level
            
        Returns:
            Fitted network with optimized RLC parameters
        """
        # Set up parameters to fit
        params_to_fit = {}
        for idx in rlc_indices:
            element = network_template[idx]
            if element['type'] in ['series_rlc', 'parallel_rlc']:
                params_to_fit[idx] = ['r', 'l', 'c']
        
        # Set reasonable bounds for RLC parameters
        bounds = {
            'series_rlc_r': (0.1, 1000),      # 0.1 to 1000 ohms
            'series_rlc_l': (1e-12, 1e-6),    # 1pH to 1µH
            'series_rlc_c': (1e-15, 1e-9),    # 1fF to 1nF
            'parallel_rlc_r': (1, 10000),     # 1 to 10k ohms
            'parallel_rlc_l': (1e-12, 1e-6),  # 1pH to 1µH
            'parallel_rlc_c': (1e-15, 1e-9)   # 1fF to 1nF
        }
        
        # Perform the fitting
        fitted_network, _ = self.fit_network(
            measured_s_params, 
            network_template, 
            params_to_fit, 
            termination, 
            bounds, 
            verbose=verbose
        )
        
        return fitted_network

    def fit_magnon_parameters(self, measured_s_params: np.ndarray, network_template: List[Dict], 
                             magnon_indices: List[int], termination: str = None, verbose: int = 1) -> List[Dict]:
        """
        Fit magnon parameters in the network.
        
        Args:
            measured_s_params: Array of measured S-parameter matrices
            network_template: List of dictionaries describing the network structure
            magnon_indices: Indices of magnon elements to fit
            termination: Optional termination type ('short', 'open', or None for matched load)
            verbose: Verbosity level
            
        Returns:
            Fitted network with optimized magnon parameters
        """
        # Set up parameters to fit
        params_to_fit = {}
        for idx in magnon_indices:
            element = network_template[idx]
            if element['type'] == 'magnon':
                params_to_fit[idx] = ['omega_m', 'gamma', 'kappa']
        
        # Set reasonable bounds for magnon parameters
        bounds = {
            'magnon_omega_m': (1e9, 20e9),    # 1 GHz to 20 GHz
            'magnon_gamma': (1e6, 1e8),       # 1 MHz to 100 MHz
            'magnon_kappa': (1e6, 1e9)        # 1 MHz to 1 GHz
        }
        
        # Perform the fitting
        fitted_network, _ = self.fit_network(
            measured_s_params, 
            network_template, 
            params_to_fit, 
            termination, 
            bounds, 
            verbose=verbose
        )
        
        return fitted_network

    def fit_all_parameters(self, measured_s_params: np.ndarray, network_template: List[Dict], 
                          termination: str = None, verbose: int = 1) -> List[Dict]:
        """
        Fit all parameters in the network (transmission line lengths, RLC parameters, magnon parameters).
        
        Args:
            measured_s_params: Array of measured S-parameter matrices
            network_template: List of dictionaries describing the network structure
            termination: Optional termination type ('short', 'open', or None for matched load)
            verbose: Verbosity level
            
        Returns:
            Fitted network with all optimized parameters
        """
        # Set up parameters to fit
        params_to_fit = {}
        
        for i, element in enumerate(network_template):
            element_type = element['type']
            
            if element_type == 'tline':
                params_to_fit[i] = ['length']
            elif element_type in ['series_rlc', 'parallel_rlc']:
                params_to_fit[i] = ['r', 'l', 'c']
            elif element_type == 'magnon':
                params_to_fit[i] = ['omega_m', 'gamma', 'kappa']
        
        # Set reasonable bounds for all parameters
        bounds = {
            'tline_length': (0.001, 0.5),      # Between 1mm and 50cm
            'series_rlc_r': (0.1, 1000),       # 0.1 to 1000 ohms
            'series_rlc_l': (1e-12, 1e-6),     # 1pH to 1µH
            'series_rlc_c': (1e-15, 1e-9),     # 1fF to 1nF
            'parallel_rlc_r': (1, 10000),      # 1 to 10k ohms
            'parallel_rlc_l': (1e-12, 1e-6),   # 1pH to 1µH
            'parallel_rlc_c': (1e-15, 1e-9),   # 1fF to 1nF
            'magnon_omega_m': (1e9, 20e9),     # 1 GHz to 20 GHz
            'magnon_gamma': (1e6, 1e8),        # 1 MHz to 100 MHz
            'magnon_kappa': (1e6, 1e9)         # 1 MHz to 1 GHz
        }
        
        # Perform the fitting
        fitted_network, _ = self.fit_network(
            measured_s_params, 
            network_template, 
            params_to_fit, 
            termination, 
            bounds, 
            verbose=verbose
        )
        
        return fitted_network
