# src/fiberis/moose/study.py
# This module provides high-level tools for sensitivity analysis and optimization using MOOSE.
# Shenyao Jin, shenyaojin@mines.edu, 2025-08-10
# Not tested.
import os
import shutil
import numpy as np
from typing import List, Dict, Any, Callable, Optional, Tuple

from .input_editor import MooseInputEditor
from .runner import MooseRunner
from .postprocessor import MoosePointSamplerSet

# Make scipy an optional dependency
try:
    from scipy.optimize import minimize, OptimizeResult
    _scipy_available = True
except ImportError:
    _scipy_available = False

class MooseStudy:
    """
    Manages sensitivity analysis and optimization workflows for MOOSE simulations.

    This class orchestrates a series of MOOSE runs by repeatedly modifying a template
    input file, executing the simulation, and evaluating the results against an
    objective function.

    Workflow:
    1. Initialize `MooseStudy`, providing either a path to a template `.i` file or a
       Python function that can generate it using `ModelBuilder`.
    2. Add parameters to be varied using `add_parameter()`.
    3. Define the goal of the study by providing a custom `objective_function` that
       calculates a misfit score from simulation results.
    4. Run a sensitivity analysis using `run_sensitivity()` or a full optimization
       using `run_optimization()`.
    """

    def __init__(self,
                 study_name: str,
                 moose_executable_path: str,
                 output_base_dir: str,
                 template_input_file: Optional[str] = None,
                 builder_function: Optional[Callable[[str], None]] = None,
                 builder_args: Optional[Dict[str, Any]] = None):
        """
        Initializes the MooseStudy.

        Args:
            study_name (str): A name for the study.
            moose_executable_path (str): Absolute path to the MOOSE executable.
            output_base_dir (str): The directory where all simulation runs and logs will be stored.
            template_input_file (Optional[str]): Path to an existing MOOSE `.i` file to be used as a template.
            builder_function (Optional[Callable]): A function that generates the template `.i` file.
                                                  It must accept one argument: the output file path.
            builder_args (Optional[Dict[str, Any]]): A dictionary of keyword arguments to pass to the builder_function.

        Raises:
            ValueError: If neither or both `template_input_file` and `builder_function` are provided.
            FileNotFoundError: If the provided `template_input_file` does not exist.
        """
        if not ((template_input_file is None) ^ (builder_function is None)):
            raise ValueError("You must provide exactly one of `template_input_file` or `builder_function`.")

        self.study_name = study_name
        self.output_base_dir = os.path.abspath(output_base_dir)
        os.makedirs(self.output_base_dir, exist_ok=True)

        self.moose_runner = MooseRunner(moose_executable_path)
        self.parameters: List[Dict[str, Any]] = []
        self.param_order: List[str] = []
        self.objective_function: Optional[Callable[[str], float]] = None
        self._run_counter = 0

        if builder_function:
            self.template_input_file = os.path.join(self.output_base_dir, f"{self.study_name}_template.i")
            print(f"Generating template input file using provided builder function...")
            args = builder_args or {}
            builder_function(self.template_input_file, **args)
            print(f"Template file generated at: {self.template_input_file}")
        elif template_input_file:
            if not os.path.exists(template_input_file):
                raise FileNotFoundError(f"Template input file not found: {template_input_file}")
            self.template_input_file = os.path.abspath(template_input_file)

    def add_parameter(self, name: str, node_path: str, param_name: str, bounds: Optional[Tuple[float, float]] = None):
        """
        Registers a parameter to be varied in the study.

        Args:
            name (str): A unique, user-friendly name for the parameter (e.g., 'fracture_perm').
            node_path (str): The path to the block in the MOOSE input file (e.g., '/Materials/permeability_fracture').
            param_name (str): The name of the parameter within the block (e.g., 'permeability').
            bounds (Optional[Tuple[float, float]]): The (min, max) bounds for the parameter, required for optimization.
        """
        if name in self.param_order:
            raise ValueError(f"Parameter with name '{name}' already exists.")
        self.parameters.append({
            "name": name,
            "node_path": node_path,
            "param_name": param_name,
            "bounds": bounds
        })
        self.param_order.append(name)
        print(f"Added parameter '{name}' targeting '{node_path}[{param_name}]'.")

    def set_objective_function(self, objective_func: Callable[[str], float]):
        """
        Sets the user-defined function to calculate the objective/misfit.

        The provided function must take one argument:
            - output_dir (str): The path to the directory of a completed simulation run.
        It must return a single float value representing the misfit or objective score.

        Example:
            def calculate_misfit(output_dir):
                samplers = MoosePointSamplerSet(output_dir)
                sim_pressure = samplers.get_sampler('pp_inj')
                # ... compare to real data ...
                return np.sum((sim_pressure.data - real_data)**2)
        """
        self.objective_function = objective_func
        print("Objective function has been set.")

    def run_single_case(self, param_values: Dict[str, float], run_name: str) -> float:
        """
        Runs a single MOOSE simulation with a specific set of parameter values.

        Args:
            param_values (Dict[str, float]): A dictionary mapping parameter names to their values for this run.
            run_name (str): A unique name for this specific run.

        Returns:
            float: The misfit value calculated by the objective function.
        """
        if self.objective_function is None:
            raise RuntimeError("Cannot run case: Objective function not set. Use set_objective_function().")

        run_dir = os.path.join(self.output_base_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)
        modified_input_path = os.path.join(run_dir, "input.i")

        # Edit the template file with the new parameter values
        editor = MooseInputEditor(self.template_input_file)
        for param_config in self.parameters:
            p_name = param_config["name"]
            if p_name in param_values:
                editor.set_param_value(
                    node_path=param_config["node_path"],
                    param_name=param_config["param_name"],
                    new_value=param_values[p_name]
                )
        editor.write_file(modified_input_path)

        # Run the simulation
        success, _, stderr = self.moose_runner.run(
            input_file_path=modified_input_path,
            output_directory=run_dir,
            log_file_name="moose_log.txt"
        )

        if not success:
            print(f"Warning: Simulation run '{run_name}' failed. Returning infinity as misfit.")
            print(f"STDERR: {stderr}")
            return float('inf')

        # Calculate and return the objective
        misfit = self.objective_function(run_dir)
        print(f"Run '{run_name}' completed with misfit: {misfit:.6e}")
        return misfit

    def _objective_wrapper(self, x: np.ndarray) -> float:
        """
        The bridge function for scipy.optimize.minimize.
        It maps a numpy array from scipy to a simulation run and returns the misfit.
        """
        self._run_counter += 1
        params_dict = {name: val for name, val in zip(self.param_order, x)}
        run_name = f"opt_run_{self._run_counter:04d}"
        
        # Log the parameters being tested
        param_str = ", ".join([f"{k}={v:.4e}" for k, v in params_dict.items()])
        print(f"--- Starting {run_name} with params: {param_str} ---")

        return self.run_single_case(params_dict, run_name)

    def run_optimization(self, initial_guess: Dict[str, float], method: str = 'L-BFGS-B', options: Optional[Dict] = None) -> 'OptimizeResult':
        """
        Performs optimization to find parameters that minimize the objective function.

        Args:
            initial_guess (Dict[str, float]): A dictionary of starting values for each parameter.
            method (str): The optimization algorithm to use (passed to scipy.optimize.minimize).
            options (Optional[Dict]): A dictionary of options for the optimizer.

        Returns:
            scipy.optimize.OptimizeResult: The result object from the scipy optimizer.
        """
        if not _scipy_available:
            raise ImportError("scipy is required for optimization. Please install it via 'pip install scipy'.")
        if not self.parameters:
            raise RuntimeError("No parameters have been added to the study.")

        print(f"--- Starting Optimization using '{method}' ---")
        self._run_counter = 0

        # Prepare initial guess and bounds in the format scipy expects
        x0 = [initial_guess[name] for name in self.param_order]
        bounds = [p["bounds"] for p in self.parameters if p["name"] in self.param_order]

        if any(b is None for b in bounds):
            print("Warning: Some parameters have no bounds. Using an unconstrained optimization method may be necessary.")

        result = minimize(
            fun=self._objective_wrapper,
            x0=np.array(x0),
            bounds=bounds,
            method=method,
            options=options
        )

        print("--- Optimization Finished ---")
        print(f"Success: {result.success}")
        print(f"Message: {result.message}")
        print(f"Final Misfit: {result.fun:.6e}")
        final_params = {name: val for name, val in zip(self.param_order, result.x)}
        print("Optimal Parameters:")
        for name, val in final_params.items():
            print(f"  - {name}: {val:.6e}")

        return result

    def run_sensitivity(self, param_name: str, values: List[float]) -> Dict[float, float]:
        """
        Performs a sensitivity analysis for a single parameter.

        Args:
            param_name (str): The name of the parameter to vary.
            values (List[float]): A list of values to test for the parameter.

        Returns:
            Dict[float, float]: A dictionary mapping each parameter value to the resulting misfit.
        """
        if param_name not in self.param_order:
            raise ValueError(f"Parameter '{param_name}' not found in the study.")

        print(f"--- Starting Sensitivity Analysis for '{param_name}' ---")
        results = {}
        for i, value in enumerate(values):
            run_name = f"sensitivity_{param_name}_{i:03d}"
            params_dict = {param_name: value}
            print(f"--- Running case {i+1}/{len(values)}: {param_name} = {value:.4e} ---")
            misfit = self.run_single_case(params_dict, run_name)
            results[value] = misfit

        print("--- Sensitivity Analysis Finished ---")
        return results
