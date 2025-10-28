# src/fiberis/moose/optimizer.py
# Description: This file contains the Optimizer class, designed to automate parameter
# optimization and history matching studies using the fiberis library.
# Shenyao Jin, 10/27/2025

from __future__ import annotations
import os
import datetime
import numpy as np
from typing import List, Dict, Any, Callable, Optional, Tuple, TYPE_CHECKING

# Third-party libraries for optimization
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args

# Import necessary modules from fiberis
from fiberis.moose.editor import MooseModelEditor
from fiberis.moose.runner import MooseRunner
from fiberis.moose.postprocessor import MoosePointSamplerSet
from fiberis.analyzer.Data1D.Data1D_MOOSEps import Data1D_MOOSEps

if TYPE_CHECKING:
    from fiberis.moose.model_builder import ModelBuilder


class Optimizer:
    """
    Orchestrates the optimization process by linking a MOOSE model with an optimization
    algorithm. It uses a pre-configured ModelBuilder instance, modifies it with a
    MooseModelEditor, runs simulations, and uses user-defined functions to calculate
    misfit against observation data.
    """

    def __init__(
        self,
        base_model: ModelBuilder,
        runner: MooseRunner,
        parameter_space: Dict[str, Dict],
        misfit_function: Callable,
        data_loader_function: Optional[Callable] = None,
        constraints: Optional[List[Callable]] = None,
        working_directory: str = "optimizer_runs",
    ):
        """
        Initializes the Optimizer.

        Args:
            base_model (ModelBuilder): A fully configured ModelBuilder instance that serves
                                       as the template for all simulations.
            runner (MooseRunner): An instance of the MooseRunner to execute simulations.
            parameter_space (Dict): A dictionary defining the parameters to optimize, their
                                    search space, and their location in the ModelBuilder.
                                    Example:
                                    {
                                        'perm': {
                                            'dimension': Real(1e-15, 1e-12, name='perm'),
                                            'location': {
                                                'block_path': ['Materials', 'permeability_hf'],
                                                'parameter_name': 'permeability'
                                            }
                                        }
                                    }
            misfit_function (Callable): A user-defined function that takes simulation data
                                        and observation data and returns a float misfit value.
            data_loader_function (Optional[Callable]): A user-defined function to load simulation
                                                       results from an output directory. If None, a
                                                       default loader for point samplers is used.
            constraints (Optional[List[Callable]]): A list of functions that take a dictionary
                                                    of parameters and return True if constraints
                                                    are met, False otherwise.
            working_directory (str): The root directory to store all simulation runs and logs.
        """
        self.base_model = base_model
        self.runner = runner
        self.parameter_space = parameter_space
        self.misfit_function = misfit_function
        self.data_loader_function = data_loader_function
        self.constraints = constraints if constraints is not None else []
        self.results: List[Dict[str, Any]] = []

        # Internal components
        self._editor = MooseModelEditor(self.base_model)
        self._working_directory = os.path.abspath(working_directory)
        os.makedirs(self._working_directory, exist_ok=True)
        print(f"Optimizer initialized. Working directory: {self._working_directory}")

    def _default_point_sampler_loader(self, output_directory: str, **kwargs) -> List[Data1D_MOOSEps]:
        """
        A default data loader that reads all point sampler outputs from a directory
        using the MoosePointSamplerSet class.

        Args:
            output_directory (str): The directory where the simulation output CSVs are stored.
            **kwargs: Additional arguments, expecting 'start_time' for timestamp correction.

        Returns:
            List[Data1D_MOOSEps]: A list of Data1D objects, one for each point sampler variable.
        """
        try:
            sampler_set = MoosePointSamplerSet(folder=output_directory)
            loaded_data = list(sampler_set.samplers.values())

            if not loaded_data:
                print("Warning: Default loader found no point sampler data.")
                return []

            # Apply timestamp correction if provided
            start_time = kwargs.get('start_time')
            if start_time:
                for sim_data in loaded_data:
                    sim_data.start_time = start_time
            else:
                print("Warning: 'start_time' not provided to default loader. Timestamps may be incorrect.")

            return loaded_data

        except Exception as e:
            print(f"Error during default data loading with MoosePointSamplerSet: {e}")
            # Return an empty list on failure to prevent crashing the optimization
            return []

    def _run_single_instance(
        self,
        instance_id: str,
        params: Dict[str, Any],
        clean_output_dir: bool = True
    ) -> Tuple[bool, Optional[str]]:
        """
        Manages the process of running a single MOOSE simulation instance.

        Args:
            instance_id (str): A unique identifier for this simulation run.
            params (Dict[str, Any]): A dictionary of parameter names and their values for this run.
            clean_output_dir (bool): If True, the output directory will be cleaned before the run.

        Returns:
            Tuple[bool, Optional[str]]: A tuple containing a success flag and the path to the
                                        output directory if successful, otherwise None.
        """
        # 1. Create a dedicated directory for this run
        run_dir = os.path.join(self._working_directory, instance_id)
        # The runner will handle directory creation and cleaning
        
        # 2. Apply the new parameters to the model using the editor
        for name, value in params.items():
            if name in self.parameter_space:
                loc = self.parameter_space[name]['location']
                self._editor.set_parameter(
                    block_path=loc['block_path'],
                    parameter_name=loc['parameter_name'],
                    new_value=value
                )
        
        # 3. Generate the MOOSE input file
        input_filepath = os.path.join(run_dir, "input.i")
        # Ensure directory exists before writing the input file, especially if not cleaning
        os.makedirs(run_dir, exist_ok=True)
        self.base_model.generate_input_file(input_filepath)

        # 4. Run the simulation
        success, _, _ = self.runner.run(
            input_file_path=input_filepath,
            output_directory=run_dir,
            num_processors=20, # Example value, could be configurable
            log_file_name=f"simulation_{instance_id}.log",
            clean_output_dir=clean_output_dir
        )

        if success:
            return True, run_dir
        else:
            print(f"Warning: Simulation failed for instance {instance_id}.")
            return False, run_dir

    def test_misfit_calculation(
        self,
        test_params: Dict[str, Any],
        misfit_args: Dict[str, Any],
        loader_args: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Performs a pre-flight check to test the misfit calculation with a single run.

        Args:
            test_params (Dict[str, Any]): Parameter values for the test run.
            misfit_args (Dict[str, Any]): Arguments for the misfit function (e.g., observation data).
            loader_args (Optional[Dict[str, Any]]): Arguments for the data loader (e.g., start_time).
        """
        if loader_args is None:
            loader_args = {}
        print("\n--- Starting Misfit Function Test ---")
        instance_id = "misfit_test"
        test_dir = os.path.join(self._working_directory, instance_id)

        # Robust check for existing output files
        simulation_needed = True
        run_dir = test_dir
        success = False

        # The input file is named 'input.i', so the default CSV output will be 'input_csv.csv'
        expected_csv = os.path.join(test_dir, "input_csv.csv")

        if os.path.exists(expected_csv) and os.path.getsize(expected_csv) > 0:
            print(f"Found existing, non-empty output file: '{expected_csv}'. Skipping simulation.")
            simulation_needed = False
            success = True
        else:
            # Check if any PointValue samplers are configured, which would generate the CSV.
            ps_configs = [pp for pp in self.base_model.postprocessor_info.get('postprocessors', [])
                          if pp.get('pp_type') == 'PointValue']
            if not ps_configs:
                print("Warning: No PointValue postprocessors found in the model. The simulation may not generate the expected CSV output for the default loader.")


        if simulation_needed:
            print("No complete previous test output found. Running a single simulation for the test...")
            # Run the simulation but DO NOT clean the directory, to preserve results
            success, run_dir = self._run_single_instance(
                instance_id,
                test_params,
                clean_output_dir=False
            )

        if not success or not run_dir:
            print("ERROR: Misfit test failed because the simulation run was unsuccessful.")
            return

        # Load the data using the appropriate loader
        print("Loading simulation data...")
        if self.data_loader_function:
            sim_data = self.data_loader_function(run_dir, **loader_args)
        else:
            sim_data = self._default_point_sampler_loader(run_dir, **loader_args)

        if not sim_data:
            print("ERROR: Misfit test failed because the data loader returned no data.")
            return

        # Try to run the misfit function and catch any errors
        try:
            print("Executing misfit function...")
            misfit_value = self.misfit_function(simulation_results=sim_data, **misfit_args)

            if np.isnan(misfit_value):
                print("ERROR: Misfit function returned NaN.")
            else:
                print(f"SUCCESS: Misfit function test completed successfully. Returned value: {misfit_value:.4e}")

        except Exception as e:
            print(f"ERROR: Misfit function failed with an exception: {e}")

        print("--- Misfit Function Test Finished ---\n")

    def run_optimization(
        self,
        n_calls: int = 50,
        n_initial_points: int = 10,
        misfit_args: Optional[Dict[str, Any]] = None,
        loader_args: Optional[Dict[str, Any]] = None
    ):
        """
        Starts the Bayesian optimization process.

        Args:
            n_calls (int): The total number of simulations to run.
            n_initial_points (int): The number of initial random points to explore.
            misfit_args (Optional[Dict[str, Any]]): Fixed arguments for the misfit function.
            loader_args (Optional[Dict[str, Any]]): Fixed arguments for the data loader.
        """
        if misfit_args is None:
            misfit_args = {}
        if loader_args is None:
            loader_args = {}
            
        # Extract the skopt dimension objects from the parameter_space dictionary
        dimensions = [v['dimension'] for v in self.parameter_space.values()]

        # This is the core function that skopt will call to evaluate a set of parameters
        @use_named_args(dimensions=dimensions)
        def objective(**params):
            # 1. Check constraints
            for constraint_func in self.constraints:
                if not constraint_func(params):
                    print(f"Constraint '{constraint_func.__name__}' violated. Penalizing.")
                    return 1e10  # Return a large penalty value

            # 2. Run the simulation
            instance_id = f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            success, run_dir = self._run_single_instance(
                instance_id,
                params,
                clean_output_dir=True  # Ensure clean runs for optimization iterations
            )

            if not success or not run_dir:
                return 1e10 # Penalize failed runs

            # 3. Load data
            if self.data_loader_function:
                sim_data = self.data_loader_function(run_dir, **loader_args)
            else:
                sim_data = self._default_point_sampler_loader(run_dir, **loader_args)
            
            if not sim_data:
                print("Warning: Data loader returned no data. Penalizing.")
                return 1e10

            # 4. Calculate misfit
            try:
                misfit = self.misfit_function(simulation_results=sim_data, **misfit_args)
                if np.isnan(misfit):
                    print("Warning: Misfit function returned NaN. Penalizing.")
                    misfit = 1e10
            except Exception as e:
                print(f"Warning: Misfit function failed with error: {e}. Penalizing.")
                misfit = 1e10

            print(f"Instance: {instance_id} | Misfit: {misfit:.4e} | Params: {params}")
            self.results.append({'misfit': misfit, 'params': params, 'instance_id': instance_id})
            return misfit

        # Start the optimization process
        print(f"\n--- Starting Bayesian Optimization ({n_calls} calls) ---")
        result = gp_minimize(
            func=objective,
            dimensions=dimensions,
            n_calls=n_calls,
            n_initial_points=n_initial_points,
            random_state=42
        )

        # Print summary
        print("\n--- Optimization Finished ---")
        print(f"Best Misfit: {result.fun:.4e}")
        best_params = {dim.name: val for dim, val in zip(dimensions, result.x)}
        print("Best Parameters:")
        for key, value in best_params.items():
            print(f"  {key}: {value:.4e}")
        
        return best_params, result.fun
