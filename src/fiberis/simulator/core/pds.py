# This script is to provide the utilities for 1D pressure diffusion problem
# Developed by Shenyao Jin, shenyaojin@mines.edu
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from fiberis.simulator.optimizer import tso as tso
from fiberis.simulator.solver import matbuilder
from src.solver import matbuilder  # Load the matrix builder and PDE solver
from src.solver import PDEsolver_IMP, PDESolver_EXP
from src.pds.DSS_analyzer_Mariner import Data1D_GAUGE


# Define the class for the 1D pressure diffusion problem; this class will only support single source term.
# upgrade mesh that can support heterogeneous mesh.
# upgrade the source term that can support multiple source terms.
# upgrade the algorithm using Kazemi's method.
class PDS1D_SingleSource:
    def __init__(self):
        self.mesh = None  # Mesh
        self.source = None  # Source term
        self.lbc = None  # Boundary conditions: left boundary condition
        self.rbc = None  # Boundary conditions: right boundary condition
        self.initial = None  # Initial condition
        self.diffusivity = None  # Diffusivity
        self.t0 = 0  # Initial time
        self.taxis = None  # Time axis
        self.sourceidx = None  # Source index in the mesh; would be a single index
        self.snapshot = None  # Snapshot of the solution
        self.history = []  # History of the solution

    # Define the parameters for the problem
    def set_mesh(self, mesh):
        self.mesh = mesh  # Set mesh, 1D numpy array
        flag, msg = self._check_mesh()
        self.history.append(msg)

    def set_source(self, source):
        self.source = source  # Set source term, which would be the pressure gauge dataframe.
        flag, msg = self._check_source()
        self.history.append(msg)

    def set_bcs(self, lbc='Neumann', rbc='Neumann'):
        self.lbc = lbc  # Set left boundary condition
        self.rbc = rbc
        flag, msg = self._check_bc()
        self.history.append(msg)

    def set_initial(self, initial):
        # Define the initial condition
        self.initial = initial  # 1D array of initial condition having same length as mesh array.
        flag, msg = self._check_initial()
        self.history.append(msg)

    def set_diffusivity(self, diffusivity):
        """
        Set diffusivity for the problem.
        If a single value is provided, broadcast it to the length of the mesh.
        """
        if isinstance(diffusivity, (int, float)):  # Check if diffusivity is a single number
            self.diffusivity = np.full(len(self.mesh), diffusivity)
            print("Diffusivity is a single scalar value, broadcasted to the mesh length.")
        elif isinstance(diffusivity, (list, np.ndarray)) and len(diffusivity) == len(
                self.mesh):  # Check for proper length
            self.diffusivity = np.array(diffusivity)
        else:
            raise ValueError(
                "Diffusivity must be either a single scalar value or an array of the same length as the mesh.")
        flag, msg = self._check_diffusivity()
        self.history.append(msg)

    def set_t0(self, t0):
        # Set initial time
        self.t0 = t0  # Initial time, float
        self.history.append(f"Initial time set done. \nThe simulation starts at {t0}.")

    def set_sourceidx(self, sourceidx):
        # Set source index
        self.sourceidx = sourceidx
        flag, msg = self._check_sourceidx()
        self.history.append(msg)

    # Print the all the parameters
    def print(self):
        """
        Print out a nicely formatted summary of the PDS1D_SingleSource object's attributes,
        using np.unique for any numpy array values.
        """
        print("=========================================")
        print("         PDS1D_SingleSource Info         ")
        print("=========================================")

        # Loop over all attributes in the instance dictionary
        for key, value in self.__dict__.items():
            if isinstance(value, np.ndarray):
                # Print shape and unique values for numpy arrays
                unique_values = np.unique(value)
                print(f"{key:>15}: shape={value.shape}, unique={unique_values}")
            else:
                # For non-array attributes, just print the value
                print(f"{key:>15}: {value}")

        print("=========================================")

    # QC: self check
    def _check_mesh(self):
        """
        Internal check for 'mesh'.
        Return (True, msg) if okay,
               (False, error_msg) otherwise.
        """
        if self.mesh is None:
            return False, "Mesh is not set."
        if not isinstance(self.mesh, np.ndarray):
            return False, "Mesh must be a numpy array."
        if len(self.mesh) < 2:
            return False, "Mesh must have at least 2 points."
        return True, "Mesh is properly set."

    def _check_source(self):
        """
        Internal check for 'source'.
        """
        if self.source is None:
            return False, "Source term is not set."
        # Example: check that source is a DataFrame
        # if not isinstance(self.source, pd.DataFrame):
        #     return (False, "Source must be a pandas DataFrame.")

        return True, "Source term is properly set."

    def _check_bc(self):
        """
        Internal check for 'bc' (boundary conditions).
        """
        if self.lbc is None or self.rbc is None:
            return False, "Boundary condition(s) (lbc/rbc) not set."
        allowed_bc = ['Dirichlet', 'Neumann']
        if self.lbc not in allowed_bc:
            return False, f"Invalid left BC: {self.lbc}. Must be {allowed_bc}."
        if self.rbc not in allowed_bc:
            return False, f"Invalid right BC: {self.rbc}. Must be {allowed_bc}."

        return True, "Boundary conditions are properly set."

    def _check_initial(self):
        """
        Internal check for 'initial' condition.
        """
        if self.initial is None:
            return False, "Initial condition is not set."
        if self.mesh is None:
            return False, "Mesh is not set. Please set the mesh first."
        if len(self.initial) != len(self.mesh):
            return False, "Length of initial condition array must match mesh length."

        return True, "Initial condition is properly set."

    def _check_diffusivity(self):
        """
        Internal check for 'diffusivity'.
        """
        if self.diffusivity is None:
            return False, "Diffusivity is not set."

        if isinstance(self.diffusivity, np.ndarray):
            if len(self.diffusivity) != len(self.mesh):
                return False, "Diffusivity array length must match mesh length."
        else:
            return False, "Diffusivity must be a numpy array or the pds frame is failed to init the diffusivity."

        return True, "Diffusivity is properly set."

    def _check_sourceidx(self):
        """
        Internal check for 'sourceidx'.
        """
        if self.sourceidx is None:
            return False, "Source index is not set."
        mesh_idx = np.arange(len(self.mesh))
        if self.sourceidx not in mesh_idx:
            return False, "Source index is not in the mesh."

        return True, "Source index is properly set."

    def self_check(self, params=None):
        """
        Check if specified parameters are properly set.
        If params is None, all known parameters are checked.

        :param params: None, a string, or a list of strings.
                       E.g. 'bc', 'mesh', ['mesh', 'source'].
        :return: True if all specified checks pass, otherwise False.
        """

        # Dictionary mapping parameter names to the corresponding check function
        check_funcs = {
            'mesh': self._check_mesh,
            'source': self._check_source,
            'bc': self._check_bc,
            'initial': self._check_initial,
            'diffusivity': self._check_diffusivity,
            'sourceidx': self._check_sourceidx
        }

        # If no parameter is specified, we check them all
        if params is None:
            params = list(check_funcs.keys())
        # If a single string is provided, wrap it in a list
        elif isinstance(params, str):
            params = [params]

        all_good = True
        for param in params:
            if param not in check_funcs:
                print(f"Parameter '{param}' is not recognized. Skipping.")
                all_good = False
                continue

            # Call the check function
            ok, msg = check_funcs[param]()
            if not ok:
                print(msg)
                all_good = False
            # If you want to see success messages, you could uncomment the next line:
            # else:
            #     print(msg)

        return all_good

    # Define the function to solve the problem; core function
    def solve(self, optimizer=False, **kwargs):
        # If optimizer is false, then solve the problem with the given parameters and a given time step "dt", pass this to PDE solver.
        # Implicit solver

        # do the self check before solving the problem
        if not self.self_check():
            print("Self check failed. Please check the parameters.")
            return

        # Future plan: use decorator to enhance the function here.
        if not optimizer:
            # Generate the time array using t_total. If not given, then use the time array from the source term.
            if 't_total' in kwargs:
                t_total = kwargs['t_total']
                dt = kwargs['dt']
                print("Time array generated using t_total.")
                self.history.append("Time array generated using t_total.")
            else:
                dt = kwargs['dt']
                t_total = (self.source.calculate_time())[-1] * 3600  # get the last time from the source term
                print("Time array generated using the source term.")
                self.history.append("Time array generated using the source term.")
            # Initialize the snapshot
            self.snapshot = []

            # Set the initial condition
            self.snapshot.append(self.initial)
        else:
            dt_init = kwargs['dt_init']
            if 't_total' in kwargs:
                t_total = kwargs['t_total']
                print("Time array generated using t_total.")
            else:
                t_total = (self.source.calculate_time())[-1] * 3600
                print("Time array generated using the source term.")

            # Initialize the snapshot
            self.snapshot = []

            # Set the initial condition
            self.snapshot.append(self.initial)

        # get the intermediate time parameter, optimizer = True -> dt; optimizer = False -> dt_init
        time_parameter = -1
        # Initialize the time parameter
        if optimizer:
            time_parameter = dt_init
        else:
            time_parameter = dt

        # initialize the taxis
        self.taxis = [self.t0]

        self.record_log("Start to solve the problem.")

        # start to loop through the time array
        while self.taxis[-1] < t_total:
            # Before the loop, get the value of source term at the current time
            source_val = self.source.get_value_by_time(self.taxis[-1] - self.t0)
            self.record_log("Time:", self.taxis[-1], "Source term:", source_val)

            # Full step solution. For the non-optimizer case, only full step solution is needed.
            # Call the Matrix builder
            A, b = matbuilder.matrix_builder_1d_single_source(self, time_parameter)
            # Call the solver, get an updated snapshot
            if 'mode' in kwargs:
                if kwargs['mode'] == 'implicit':
                    snapshot_upd = PDEsolver_IMP.solver_implicit(A, b, solver='numpy')  # call the implicit solver
                else:
                    snapshot_upd = PDESolver_EXP.solver_explicit(A, b)  # call the explicit solver
            else:
                # Default mode is implicit solver
                snapshot_upd = PDEsolver_IMP.solver_implicit(A, b, solver='numpy')

            # If no optimizer, append the snapshot to the list
            if not optimizer:
                self.snapshot.append(snapshot_upd)
                # Update the time
                self.taxis.append(self.taxis[-1] + time_parameter)
                self.record_log("Full step solution done. taxis[-1]=", self.taxis[-1])
                # Print the progress if print_progress is True
                if 'print_progress' in kwargs:
                    if kwargs['print_progress']:
                        print("Time:", self.taxis[-1], "Source term:", source_val)
            else:
                # Call the half step optimizer, then estimate the error
                # Call the Matrix builder to get the matrix for the half step
                A_half, b_half = matbuilder.matrix_builder_1d_single_source(self, time_parameter / 2)
                snapshot_middle_tmp = PDEsolver_IMP.solver_implicit(A_half, b_half, solver='numpy')
                # Store this tmp snapshot to the list. Only snapshot is needed; t is not needed for half step.
                self.snapshot.append(snapshot_middle_tmp)

                # then call the matrix builder again to get the matrix for the full step
                A_full, b_full = matbuilder.matrix_builder_1d_single_source(self, time_parameter / 2)
                snapshot_full = PDEsolver_IMP.solver_implicit(A_full, b_full, solver='numpy')
                # Delete the tmp snapshot
                del self.snapshot[-1]
                # Call the optimizer to 1. decide whether to accept the full step solution; 2. decide the next time
                # step size; 3. update the snapshot list (if accepted); if not accepted, decrease the time step size
                # and redo the full step solution. 4. Record the log.
                time_parameter = tso.time_sampling_optimizer(self, snapshot_upd, snapshot_full, time_parameter,
                                                             **kwargs)
                # Print the progress if print_progress is True
                if 'print_progress' in kwargs:
                    if kwargs['print_progress']:
                        print("Time:", self.taxis[-1], "Source term:", source_val)

        # Convert the snapshot/taxis to numpy array
        self.snapshot = np.array(self.snapshot)
        self.taxis = np.array(self.taxis)

        self.record_log("Problem solved")
        print("Problem solved.")

    # History recording
    def record_log(self, *args):
        time_now = datetime.now()
        # Concatenate all arguments
        msg = " ".join(map(str, args)) + f" | Time: {time_now}"
        # Append the formatted message to the history
        self.history.append(msg)

    def print_log(self):
        for msg in self.history:
            print(msg)

    # Solution data processing; preliminary
    def get_solution(self):
        return self.snapshot, self.taxis

    def get_val_at_source_idx(self):
        return self.snapshot[:, self.sourceidx]

    def get_val_at_idx(self, idx):
        return self.snapshot[:, idx]

    def get_val_at_time(self, time):
        idx = np.argmin(np.abs(self.taxis - time))
        print("Closest time = ", self.taxis[idx])
        return self.snapshot[idx]

    # Plot the solution
    def plot_solution(self, **kwargs):

        # Extract cmap from kwargs
        cmap = kwargs.get('cmap', 'bwr')
        # Extract method from kwargs (imshow or pcolormesh)
        method = kwargs.get('method', 'imshow')

        # imshow for regular grid, pcolormesh for irregular grid
        if method == 'imshow':
            plt.figure()
            plt.imshow(self.snapshot.T, aspect='auto', cmap=cmap,
                       extent=[self.taxis[0], self.taxis[-1], self.mesh[0], self.mesh[-1]])
            # Invert the y-axis
            plt.gca().invert_yaxis()
            plt.xlabel("Time/s")
            plt.ylabel("Distance/ft")
            plt.show()
        elif method == 'pcolormesh':
            plt.figure()
            plt.pcolormesh(self.taxis, self.mesh, self.snapshot.T, cmap=cmap)
            plt.colorbar()
            plt.xlabel("Time/s")
            plt.ylabel("Distance/ft")
            plt.show()
        else:
            raise ValueError("Method must be 'imshow' or 'pcolormesh'.")

    # Pack the result to npz that can be loaded by DSS_analyzer_Mariner
    def pack_result(self, **kwargs):
        # Extract the filename from kwargs
        filename = kwargs.get('filename', 'result.npz')
        mode = kwargs.get('mode', 'dss_analyzer_mariner')

        if mode == 'dss_analyzer_mariner':
            # Pack the result to npz, refer to my notes
            # (distance, time)
            np.savez(filename, daxis=self.mesh, taxis=self.taxis, data=self.snapshot)
        else:
            raise ValueError("Mode must be 'dss_analyzer_mariner'.")

    def reset(self):
        """
        Reset all parameters to the status before solve()
        Returns:
            None
        """

        self.taxis = None
        self.snapshot = None
        self.history = []


# Multiple source term framework inherits from the single source term framework
class PDS1D_MultiSource(PDS1D_SingleSource):
    # Define the class for the 1D pressure diffusion problem; this class will only support multiple source terms.

    def set_sourceidx(self, sourceidx):
        # Set source index
        super().set_sourceidx(sourceidx)
        # Convert to numpy array
        self.sourceidx = np.array(self.sourceidx) # Convert to numpy array for future use.

    def _check_source(self):
        """
        Internal check for 'source'.
        """
        if self.source is None:
            return False, "Source term is not set."

        # check the type of source, should be a list
        if not isinstance(self.source, list):
            return False, "Source term must be a list."

        # check the type of each source term, should be a DataFrame
        for source_iter in self.source:
            if not isinstance(source_iter, Data1D_GAUGE.Data1D_GAUGE):
                datatype = type(source_iter)
                return False, f"Each source term must be a mariner DataFrame. But got {datatype}."


        # Check the max time of the source term list, are they the same?

        return True, "Source term is properly set."

    def _check_sourceidx(self):
        """
        Args:
            self: source idx

        Returns:
            bool + msg
        """

        if self.sourceidx is None:
            return False, "Source index is not set."
        mesh_idx = np.arange(len(self.mesh))

        # Check if the source index is a single value
        if isinstance(self.sourceidx, int):
            return False, ("Source index must be a list of integers. If you want to set a single source index, please "
                           "use PDS1D_SingleSource.")

        # Check if the source index in the numpy array are integers
        for idx_iter in self.sourceidx:
            if isinstance(idx_iter, int):
                return False, "Source index must be a list of integers."

        for idx_iter in self.sourceidx:
            if idx_iter not in mesh_idx:
                return False, "Source index is not in the mesh."

        # if source is set:
        if self.source is not None:
            if len(self.sourceidx) != len(self.source):
                return False, "Source index and source term length do not match."

        return True, "Source index is properly set."

    # Solve the problem
    def solve(self, optimizer=False, **kwargs):
        # If optimizer is false, then solve the problem with the given parameters and a given time step "dt", pass this to PDE
        # solver.

        if not self.self_check():
            print("Self check failed. Please check the parameters.")
            return

        if not optimizer:
            # Generate the time array using t_total. If not given, then use the time array from the source term.
            if 't_total' in kwargs:
                t_total = kwargs['t_total']
                dt = kwargs['dt']
                print("Time array generated using t_total.")
            else:
                dt = kwargs['dt']
                t_total = (self.source[0].calculate_time())[-1] * 3600
                print("Time array generated using the source term.")
            # Initialize the snapshot
            self.snapshot = []

            # Set the initial condition
            self.snapshot.append(self.initial)
        else:
            dt_init = kwargs['dt_init']
            if 't_total' in kwargs:
                t_total = kwargs['t_total']
                print("Time array generated using t_total.")
            else:
                t_total = (self.source[0].calculate_time())[-1] * 3600
                print("Time array generated using the source term.")

            # Initialize the snapshot
            self.snapshot = []

            # Set the initial condition
            self.snapshot.append(self.initial)

        # get the intermediate time parameter, optimizer = True -> dt; optimizer = False -> dt_init
        time_parameter = -1
        # Initialize the time parameter
        if optimizer:
            time_parameter = dt_init
        else:
            time_parameter = dt

        # initialize the taxis
        self.taxis = [self.t0]

        self.record_log("Start to solve the problem.")

        # start to loop through the time array
        while self.taxis[-1] < t_total:
            # Before the loop, get the values of multiple source terms at the current time
            source_val = []
            for source_iter in self.source:
                source_val.append(source_iter.get_value_by_time(self.taxis[-1] - self.t0))
            self.record_log("Time:", self.taxis[-1], "Source term:", source_val)

            # Full step solution. For the non-optimizer case, only full step solution is needed.
            # Call the Matrix builder
            A, b = matbuilder.matrix_builder_1d_multi_source(self, time_parameter)

            # Call the solver, get an updated snapshot
            if 'mode' in kwargs:
                if kwargs['mode'] == 'implicit':
                    snapshot_upd = PDEsolver_IMP.solver_implicit(A, b, solver='numpy')
                else:
                    snapshot_upd = PDESolver_EXP.solver_explicit(A, b)
            else:
                # Default mode is implicit solver
                snapshot_upd = PDEsolver_IMP.solver_implicit(A, b, solver='numpy')

            # If no optimizer, append the snapshot to the list
            if not optimizer:
                self.snapshot.append(snapshot_upd)
                # Update the time
                self.taxis.append(self.taxis[-1] + time_parameter)
                self.record_log("Full step solution done. taxis[-1]=", self.taxis[-1])
                # Print the progress if print_progress is True
                if 'print_progress' in kwargs:
                    if kwargs['print_progress']:
                        print("Time:", self.taxis[-1], "Source term:", source_val)
            else:
                # Call the half step optimizer, then estimate the error
                # Call the Matrix builder to get the matrix for the half step
                A_half, b_half = matbuilder.matrix_builder_1d_multi_source(self, time_parameter / 2)
                snapshot_middle_tmp = PDEsolver_IMP.solver_implicit(A_half, b_half, solver='numpy')
                # Store this tmp snapshot to the list. Only snapshot is needed; t is not needed for half step.
                self.snapshot.append(snapshot_middle_tmp)

                # then call the matrix builder again to get the matrix for the full step
                A_full, b_full = matbuilder.matrix_builder_1d_multi_source(self, time_parameter / 2)
                snapshot_full = PDEsolver_IMP.solver_implicit(A_full, b_full, solver='numpy')
                # Delete the tmp snapshot
                del self.snapshot[-1]

                # Call the optimizer to 1. decide whether to accept the full step solution; 2. decide the next time
                # step size; 3. update the snapshot list (if accepted); if not accepted, decrease the time step size
                # and redo the full step solution. 4. Record the log.
                time_parameter = tso.time_sampling_optimizer(self, snapshot_upd, snapshot_full, time_parameter,
                                                             **kwargs)
                # Print the progress if print_progress is True
                if 'print_progress' in kwargs:
                    if kwargs['print_progress']:
                        print("Time:", self.taxis[-1], "Source term:", source_val)

        # Convert the snapshot/taxis to numpy array
        self.snapshot = np.array(self.snapshot)
        self.taxis = np.array(self.taxis)

        self.record_log("Problem solved")
        print("Problem solved.")