# src/fiberis/moose/postprocessor.py
# The module is designed to read MOOSE Exodus II output files and extract data for analysis.
# Shenyao Jin, 5/18/2025, shenyaojin@mines.edu
# This script is part of the fibeRIS project, which is licensed under the WTFPL (https://opensource.org/licenses/WTFPL).
# There are some issues with meshio and MOOSE Exodus II files, especially with the time steps.
# Don't mix this with model builder's postprocessor which is a top module block for ".i" input files.
import meshio
import numpy as np
from scipy.spatial import KDTree  # For finding nearest neighbors
from typing import List, Tuple, Dict, Union, Optional
import pandas as pd  # Optional: for returning data as DataFrame
import os
import re
import datetime
import matplotlib.pyplot as plt
from fiberis.analyzer.Data1D.Data1D_MOOSEps import Data1D_MOOSEps


# There's fatal issue with meshio, so don't use this class `MooseOutputReader` for now. --Shenyao
class MooseOutputReader:
    """
    Reads MOOSE output files (primarily Exodus .e files) and extracts data.
    """

    def __init__(self, exodus_file_path: str):
        """
        Initializes the reader by loading the Exodus file.

        Args:
            exodus_file_path (str): Path to the MOOSE Exodus II output file (.e).
        """
        try:
            self.mesh = meshio.read(exodus_file_path)
        except Exception as e:
            raise IOError(f"Failed to read Exodus file '{exodus_file_path}': {e}")

        self.points = self.mesh.points
        self._kdtree = None

        self.time_steps: Optional[np.ndarray] = None
        if self.mesh.field_data and 'time_whole' in self.mesh.field_data:
            self.time_steps = np.array(self.mesh.field_data['time_whole'])
            if self.time_steps.ndim == 0:  # If time_whole is a scalar, make it a 1-element array
                self.time_steps = np.array([self.time_steps.item()])
        elif self.mesh.point_data and self.mesh.point_data.keys():  # Check if point_data is not empty
            first_var_name = next(iter(self.mesh.point_data))
            point_data_for_var = self.mesh.point_data[first_var_name]

            if isinstance(point_data_for_var, list):  # Transient data: list of np.arrays
                num_steps = len(point_data_for_var)
                print(f"Warning: Actual time values not found in field_data['time_whole']. "
                      f"Assuming {num_steps} abstract steps [0, 1, ..., N-1] based on point_data list length.")
                self.time_steps = np.arange(num_steps)
            elif isinstance(point_data_for_var, np.ndarray):  # Likely steady-state or single time step data
                print(
                    "Info: Data for variables appears to be single-step (e.g., steady state) as point_data values are NumPy arrays "
                    "and 'time_whole' was not found in field_data. Assuming one time step at t=0.0.")
                self.time_steps = np.array([0.0])  # Assume a single, nominal time step
            else:
                print(
                    f"Warning: Unexpected format for point_data['{first_var_name}']. Cannot infer time steps robustly.")

        if self.time_steps is None:
            # Fallback if no point_data and no time_whole, or other unexpected structure
            print("Warning: Could not determine time steps from the Exodus file. "
                  "This might indicate an empty or unusual file. Time-dependent extractions might fail.")
            # Attempt to see if it's a mesh-only file with no time data or point data
            if not self.mesh.point_data and not self.mesh.cell_data:
                print("Info: The Exodus file seems to contain mesh data only, with no field or time data.")

    def _get_kdtree(self) -> KDTree:
        """Initializes and returns the KDTree for node coordinates."""
        if self._kdtree is None:
            # Adjust KDTree dimension based on mesh points dimension
            if self.points.shape[1] >= 2:  # At least 2D
                self._kdtree = KDTree(self.points[:, :2])  # Default to XY for 2D or 3D with Z=0 relevance
                if self.points.shape[1] == 3:
                    print(
                        "Info: KDTree created using XY coordinates of 3D mesh points. For full 3D nearest neighbor, KDTree should use all 3 dims.")
                    # For full 3D: self._kdtree = KDTree(self.points)
            else:  # 1D mesh
                self._kdtree = KDTree(self.points)
        return self._kdtree

    def get_nodal_variable_names(self) -> List[str]:
        """Returns a list of available nodal variable names."""
        return list(self.mesh.point_data.keys()) if self.mesh.point_data else []

    def get_cell_variable_names(self) -> List[str]:
        """Returns a list of available cell (element) variable names."""
        return list(self.mesh.cell_data.keys()) if self.mesh.cell_data else []

    def get_time_steps(self) -> Optional[np.ndarray]:
        """Returns the array of time step values, if found."""
        return self.time_steps

    def find_nearest_node_indices(self, points_coords: List[Tuple[float, ...]]) -> List[int]:
        """
        Finds the indices of the mesh nodes nearest to the given coordinates.
        """
        if not points_coords:
            return []

        kdtree = self._get_kdtree()
        query_points_np = np.array(points_coords)

        # Prepare query points based on KDTree dimension (currently XY)
        if self.points.shape[1] >= 2:  # KDTree built on XY
            if query_points_np.shape[1] == 1:  # Query is 1D, assume it's X, use Y=0
                query_points_for_kdtree = np.hstack([query_points_np, np.zeros_like(query_points_np)])
            elif query_points_np.shape[1] >= 2:
                query_points_for_kdtree = query_points_np[:, :2]
            else:
                raise ValueError("Query points have insufficient dimensions for 2D KDTree.")
        else:  # KDTree built on 1D (X)
            query_points_for_kdtree = query_points_np
            if query_points_for_kdtree.ndim == 1:  # Ensure it's (N,1) for KDTree query
                query_points_for_kdtree = query_points_for_kdtree.reshape(-1, 1)

        _, indices = kdtree.query(query_points_for_kdtree)
        return indices.tolist()

    def extract_point_data_over_time(
            self, 
            variable_name: str,
            target_points_coords: List[Tuple[float, ...]],
            output_format: str = "dict_numpy"
    ) -> Union[Dict[int, np.ndarray], pd.DataFrame, None]:
        """
        Extracts data for a given variable at specified point coordinates over all time steps.
        """
        if not self.mesh.point_data or variable_name not in self.mesh.point_data:
            print(f"Error: Variable '{variable_name}' not found in nodal data.")
            print(f"Available nodal variables: {self.get_nodal_variable_names()}")
            return None

        if self.time_steps is None:
            print("Error: Time steps are not available. Cannot extract data over time.")
            return None

        node_indices = self.find_nearest_node_indices(target_points_coords)
        if not node_indices:
            print("Warning: No target points provided or no nearest nodes found.")
            return None

        data_for_var = self.mesh.point_data[variable_name]

        # Standardize data_for_var to be a list of arrays (one array per time step)
        if isinstance(data_for_var, np.ndarray):  # Likely steady-state or single time step
            if len(self.time_steps) == 1:
                data_all_steps_list = [data_for_var]  # Wrap it in a list
            else:
                print(
                    f"Error: Variable '{variable_name}' is a single array, but found {len(self.time_steps)} time steps. Data inconsistency.")
                return None
        elif isinstance(data_for_var, list) and all(isinstance(step_data, np.ndarray) for step_data in data_for_var):
            data_all_steps_list = data_for_var
        else:
            print(
                f"Error: Data for '{variable_name}' is not in the expected format (NumPy array or list of NumPy arrays).")
            print(f"Type found: {type(data_for_var)}")
            return None

        if len(data_all_steps_list) != len(self.time_steps):
            print(
                f"Warning: Number of data arrays ({len(data_all_steps_list)}) does not match number of time steps ({len(self.time_steps)}).")
            # This can be problematic. Using the minimum length to avoid index errors.

        min_steps = min(len(data_all_steps_list), len(self.time_steps))

        extracted_data = {}
        for i, original_node_idx in enumerate(node_indices):
            point_label = f"point_{i}_node_{original_node_idx}"
            # Ensure node_idx is within bounds for all step_data arrays
            if any(original_node_idx >= step_data.shape[0] for step_data in data_all_steps_list):
                print(
                    f"Error: Node index {original_node_idx} is out of bounds for variable '{variable_name}'. Skipping this point.")
                continue

            time_series = []
            for step_idx in range(min_steps):
                step_data = data_all_steps_list[step_idx]
                if step_data.ndim == 1:
                    time_series.append(step_data[original_node_idx])
                elif step_data.ndim == 2:
                    print(
                        f"Warning: Variable '{variable_name}' at node {original_node_idx} is multi-component. Extracting first component.")
                    time_series.append(step_data[original_node_idx, 0])
                else:
                    print(
                        f"Warning: Unexpected data dimension for variable '{variable_name}' at node {original_node_idx}.")
                    time_series.append(np.nan)
            extracted_data[point_label] = np.array(time_series)

        if not extracted_data:  # If all points had errors or no points were processed
            return pd.DataFrame() if output_format == "pandas_df" else {}

        if output_format == "pandas_df":
            # For DataFrame, ensure all series have the same length as the relevant time_steps
            df_data = {}
            relevant_time_steps = self.time_steps[:min_steps]
            for label, series in extracted_data.items():
                # Series should already be of length min_steps
                df_data[label] = series
            return pd.DataFrame(df_data, index=relevant_time_steps)

        elif output_format == "dict_numpy":
            return extracted_data
        else:
            print(f"Warning: Unknown output_format '{output_format}'. Returning dict_numpy.")
            return extracted_data

    def extract_line_data_for_waterfall(
            self,
            variable_name: str,
            start_coord: Tuple[float, ...],
            end_coord: Tuple[float, ...],
            num_points_on_line: int,
            output_format: str = "dict_numpy"
    ) -> Union[Dict[float, np.ndarray], List[np.ndarray], np.ndarray, None]:
        """
        Extracts data along a line for all time steps, suitable for a waterfall plot.
        (Uses nearest neighbor for points on the line; interpolation would be more accurate).
        """
        print(
            f"Extracting line data for '{variable_name}' from {start_coord} to {end_coord} ({num_points_on_line} points).")

        if not self.mesh.point_data or variable_name not in self.mesh.point_data:
            print(f"Error: Variable '{variable_name}' not found in nodal data.")
            return None
        if self.time_steps is None:
            print("Error: Time steps are not available.")
            return None

        line_points_coords_np = np.linspace(np.array(start_coord), np.array(end_coord), num_points_on_line)

        # Use find_nearest_node_indices to get nodes for points on the line
        nearest_node_indices_on_line = self.find_nearest_node_indices(line_points_coords_np.tolist())
        if not nearest_node_indices_on_line:
            print("Warning: Could not find nearest nodes for the line points.")
            return None

        data_for_var = self.mesh.point_data[variable_name]
        if isinstance(data_for_var, np.ndarray):  # Handle steady-state / single step
            if len(self.time_steps) == 1:
                data_all_steps_list = [data_for_var]
            else:
                print(
                    f"Error: Variable '{variable_name}' is a single array, but found {len(self.time_steps)} time steps.")
                return None
        elif isinstance(data_for_var, list):
            data_all_steps_list = data_for_var
        else:
            print(f"Error: Data for '{variable_name}' is not a list or NumPy array.")
            return None

        min_steps = min(len(data_all_steps_list), len(self.time_steps))
        waterfall_data_list = []

        for step_idx in range(min_steps):
            step_data_for_all_nodes = data_all_steps_list[step_idx]
            # Ensure indices are within bounds
            valid_indices = [idx for idx in nearest_node_indices_on_line if idx < step_data_for_all_nodes.shape[0]]
            if len(valid_indices) != len(nearest_node_indices_on_line):
                print(f"Warning: Some line point nodes are out of bounds for step {step_idx}. Using NaN for those.")

            data_on_line_this_step = np.full(len(nearest_node_indices_on_line), np.nan)

            if step_data_for_all_nodes.ndim == 1:  # Scalar
                for i, node_idx in enumerate(nearest_node_indices_on_line):
                    if node_idx < step_data_for_all_nodes.shape[0]:
                        data_on_line_this_step[i] = step_data_for_all_nodes[node_idx]
            elif step_data_for_all_nodes.ndim == 2:  # Vector, take first component
                for i, node_idx in enumerate(nearest_node_indices_on_line):
                    if node_idx < step_data_for_all_nodes.shape[0]:
                        data_on_line_this_step[i] = step_data_for_all_nodes[node_idx, 0]
            else:  # Should not happen if data is consistent
                pass  # Already NaNs
            waterfall_data_list.append(data_on_line_this_step)

        relevant_time_steps = self.time_steps[:min_steps]

        if output_format == "dict_numpy":
            return {time_val: data_array for time_val, data_array in zip(relevant_time_steps, waterfall_data_list)}
        elif output_format == "list_of_arrays":
            return waterfall_data_list
        elif output_format == "single_2d_array_time_vs_space":
            if not waterfall_data_list: return np.array([])
            return np.array(waterfall_data_list)
        else:
            print(f"Warning: Unknown output_format '{output_format}'. Returning list_of_arrays.")
            return waterfall_data_list


class MoosePointSamplerSet:
    """
    Reads and manages a collection of MOOSE Point Sampler outputs from a single directory.
    This class scans a specified folder for CSV files generated by MOOSE's PointSampler
    post-processor, loads the data for each variable, and provides tools for inspection
    and visualization.
    """

    def __init__(self, folder: str):
        """
        Initializes the reader by scanning a directory for MOOSE Point Sampler CSV files.

        Args:
            folder (str): Path to the directory containing MOOSE output files.
        """
        self.folder = folder
        self.samplers: Dict[str, Data1D_MOOSEps] = {}
        self._discover_and_load_samplers()

    def _discover_and_load_samplers(self):
        """
        Discovers and loads all point sampler CSV files in the directory.
        It identifies point sampler files by finding CSVs that do not follow
        the `_xxxx.csv` numbering pattern of vector samplers.
        """
        print(f"Scanning for point samplers in '{self.folder}'")
        try:
            all_files = os.listdir(self.folder)
            # Regex to match files ending in _xxxx.csv (like vector samplers)
            numbered_files = {f for f in all_files if re.match(r'.*?_\\d+\.csv$', f)}
            all_csv_files = {f for f in all_files if f.endswith('.csv')}
            sampler_files = list(all_csv_files - numbered_files)
        except FileNotFoundError:
            print(f"Error: Directory not found at '{self.folder}'")
            return

        if not sampler_files:
            print(f"Warning: No point sampler CSV files found in '{self.folder}'.")
            return

        print(f"Found {len(sampler_files)} potential sampler file(s): {sampler_files}")

        for csv_file in sampler_files:
            csv_path = os.path.join(self.folder, csv_file)
            try:
                df = pd.read_csv(csv_path)
                if df.empty:
                    print(f"Warning: Skipping empty file '{csv_path}'")
                    continue

                time_column = df.columns[0]
                for var_name in df.columns[1:]:
                    # Use Data1D_MOOSEps to hold the data for consistency
                    sampler = Data1D_MOOSEps(name=var_name)
                    sampler.taxis = df[time_column].to_numpy()
                    sampler.data = df[var_name].to_numpy()
                    sampler.start_time = datetime.datetime.now()
                    sampler.history.add_record(f"Loaded variable '{var_name}' from {csv_path}", level="INFO")

                    if var_name in self.samplers:
                        print(f"Warning: Duplicate sampler variable name '{var_name}'. Overwriting with data from {csv_file}.")
                    self.samplers[var_name] = sampler
                print(f"Successfully loaded {len(df.columns) - 1} variables from '{csv_file}'")

            except Exception as e:
                print(f"Warning: Failed to read or process '{csv_path}'. Error: {e}")

    def get_sampler_names(self) -> List[str]:
        """Returns a list of available sampler variable names."""
        return list(self.samplers.keys())

    def get_sampler(self, name: str) -> Optional[Data1D_MOOSEps]:
        """
        Retrieves a specific sampler by its variable name.

        Args:
            name (str): The name of the variable to retrieve.

        Returns:
            Optional[Data1D_MOOSEps]: The data object, or None if not found.
        """
        return self.samplers.get(name)

    def plot_all_samplers(self, save_dir: Optional[str] = None, show_plots: bool = False):
        """
        Plots each sampler's data over time, optionally saving the plots to a directory.

        Args:
            save_dir (Optional[str]): Directory to save the plot images. If None, plots are not saved.
            show_plots (bool): If True, displays the plots interactively. Defaults to False.
                               Be cautious with this if you have many samplers.
        """
        if not self.samplers:
            print("No samplers loaded to plot.")
            return

        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print(f"Created directory for plots: {save_dir}")

        for name, sampler in self.samplers.items():
            fig, ax = plt.subplots(figsize=(12, 7))
            ax.plot(sampler.taxis, sampler.data, label=name)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel(name)
            ax.set_title(f"Point Sampler: {name}")
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)
            ax.legend()
            fig.tight_layout()

            if save_dir:
                # Sanitize filename to be safe for all filesystems
                safe_name = re.sub(r'[^a-zA-Z0-9_\\-]', '_', name)
                save_path = os.path.join(save_dir, f"sampler_{safe_name}.png")
                try:
                    fig.savefig(save_path)
                    print(f"Saved plot to '{save_path}'")
                except Exception as e:
                    print(f"Error saving plot for '{name}': {e}")

            if show_plots:
                plt.show()
            
            plt.close(fig)  # Close the figure to free memory


if __name__ == "__main__":
    # This example assumes you have a MOOSE output file named 'tutorial_step1_generated_out.e'
    # in a path relative to this script, or you provide a correct absolute path.

    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Path to project root (assuming postprocessor.py is in fibeRIS/src/fiberis/moose/)
    project_root_guess = os.path.join(current_dir, '..', '..', '..', '..')

    example_exodus_file = os.path.join(
        project_root_guess,
        "test_files", "moose_test_output", "ex01_run_output", "tutorial_step1_generated_out.e"
    )

    if not os.path.exists(example_exodus_file):
        print(f"Test Exodus file not found at: {example_exodus_file}")
        print("Please ensure the file exists or update the path for testing this module.")
        print("Creating a dummy mesh object for basic testing (no real data).")
        points = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0.5, 0.5, 0.0]]  # Added a center point
        cells = [("triangle", [[0, 1, 2], [1, 3, 2]])]
        dummy_point_data_step0 = {'diffused': np.array([0.0, 1.0, 0.0, 1.0, 0.5])}  # Values for 5 points
        # For steady state, field_data might not have time_whole, or it might be a scalar
        dummy_field_data = {}  # No time_whole, let the class infer steady state


        class DummyMesh:
            def __init__(self):
                self.points = np.array(points)
                self.cells = [meshio.CellBlock(ctype, cdata) for ctype, cdata in cells]
                self.point_data = dummy_point_data_step0  # Direct ndarray for steady state
                self.cell_data = {}
                self.field_data = dummy_field_data


        class MockMooseOutputReader(MooseOutputReader):
            def __init__(self, exodus_file_path: str):  # exodus_file_path is unused for mock
                print(f"MockMooseOutputReader: Using dummy mesh data.")
                self.mesh = DummyMesh()
                self.points = self.mesh.points
                self._kdtree = None
                # Manually trigger the time step logic from __init__ for the mock
                self.time_steps: Optional[np.ndarray] = None
                if self.mesh.field_data and 'time_whole' in self.mesh.field_data:
                    self.time_steps = np.array(self.mesh.field_data['time_whole'])
                    if self.time_steps.ndim == 0: self.time_steps = np.array([self.time_steps.item()])
                elif self.mesh.point_data and self.mesh.point_data.keys():
                    first_var_name = next(iter(self.mesh.point_data))
                    point_data_for_var = self.mesh.point_data[first_var_name]
                    if isinstance(point_data_for_var, list):
                        self.time_steps = np.arange(len(point_data_for_var))
                    elif isinstance(point_data_for_var, np.ndarray):
                        self.time_steps = np.array([0.0])  # Assume t=0 for steady state
                if self.time_steps is None: print("Mock Warning: Could not determine time steps.")


        try:
            reader = MockMooseOutputReader("dummy_path.e")
            print("\n--- Using Mock Reader with Dummy Data (Steady State assumption) ---")
        except Exception as e_dummy:
            print(f"Could not create dummy mesh for testing: {e_dummy}")
            reader = None
    else:
        try:
            reader = MooseOutputReader(example_exodus_file)
            print(f"\n--- Successfully read: {example_exodus_file} ---")
        except IOError as e:
            print(e)
            reader = None

    if reader:
        print("\nAvailable Nodal Variables:")
        nodal_vars = reader.get_nodal_variable_names()
        print(nodal_vars)

        print("\nAvailable Time Steps:")
        times = reader.get_time_steps()
        print(times if times is not None else "Not found")

        if nodal_vars:
            target_variable = nodal_vars[0]  # Use the first available variable
            points_to_extract = [(0.25, 0.5, 0.0), (0.75, 0.5, 0.0), (0.5, 0.25, 0.0), (0.5, 0.5, 0.0)]

            print(f"\nExtracting '{target_variable}' at points: {points_to_extract}")

            point_data_df = reader.extract_point_data_over_time(target_variable, points_to_extract,
                                                                output_format="pandas_df")
            if point_data_df is not None and not point_data_df.empty:
                print("\nExtracted Point Data (pandas_df format):")
                print(point_data_df)
            elif point_data_df is not None:  # Could be empty
                print("\nExtracted Point Data (pandas_df format): DataFrame is empty or encountered issues.")
            else:  # Was None
                print("\nFailed to extract point data as DataFrame.")

            start_line = (0.0, 0.5, 0.0)
            end_line = (1.0, 0.5, 0.0)
            num_samples_line = 5  # Fewer points for easier display

            print(f"\nExtracting '{target_variable}' for waterfall plot along line from {start_line} to {end_line}")
            waterfall_data_2d_array = reader.extract_line_data_for_waterfall(
                target_variable, start_line, end_line, num_samples_line,
                output_format="single_2d_array_time_vs_space"
            )
            if waterfall_data_2d_array is not None:
                print("\nWaterfall Data (2D NumPy array - time vs. space):")
                print(f"Shape: {waterfall_data_2d_array.shape}")
                print(waterfall_data_2d_array)

    print("\n\n--- Testing MoosePointSamplerSet ---")
    # This example assumes you have a directory with MOOSE point sampler CSVs.
    # We'll use the same test directory and check for CSVs.
    example_output_dir = os.path.join(
        project_root_guess,
        "test_files", "moose_test_output", "ex01_run_output"
    )

    if os.path.exists(example_output_dir):
        print(f"Scanning for point samplers in: {example_output_dir}")
        sampler_set = MoosePointSamplerSet(example_output_dir)
        
        sampler_names = sampler_set.get_sampler_names()
        if sampler_names:
            print("\nFound sampler variables:")
            print(sampler_names)

            # Get and print info about the first sampler
            first_sampler_name = sampler_names[0]
            first_sampler = sampler_set.get_sampler(first_sampler_name)
            if first_sampler:
                print(f"\nData for '{first_sampler_name}':")
                print(f"  Time points: {len(first_sampler.taxis)}")
                print(f"  Data points: {len(first_sampler.data)}")
                # print(first_sampler.data) # Uncomment for debugging

            # Plot all samplers and save them to a directory
            print("\nPlotting all samplers...")
            plot_save_dir = os.path.join(project_root_guess, "output", "sampler_plots")
            sampler_set.plot_all_samplers(save_dir=plot_save_dir, show_plots=False)

        else:
            print("No point samplers found in the directory.")
    else:
        print(f"Test directory not found: {example_output_dir}")