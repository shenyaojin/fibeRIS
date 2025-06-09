# src/fiberis/moose/runner.py
import subprocess
import os
import shlex
import shutil
from typing import Optional, Tuple, Dict, List


class MooseRunner:
    """
    A class to manage and run MOOSE simulations.
    """

    def __init__(self, moose_executable_path: str):
        """
        Initializes the MooseRunner.

        Args:
            moose_executable_path (str): The absolute path to the MOOSE executable
                                         (e.g., '/path/to/moose/your_app-opt' or 'your_app-opt' if in PATH).
        """
        self._original_moose_executable_path = moose_executable_path

        resolved_path = shutil.which(moose_executable_path)
        if resolved_path:
            self.moose_executable_path = resolved_path
        elif os.path.exists(moose_executable_path) and os.access(moose_executable_path, os.X_OK):
            self.moose_executable_path = os.path.abspath(moose_executable_path)
        else:
            raise FileNotFoundError(
                f"MOOSE executable not found or not executable at '{self._original_moose_executable_path}'. "
                "Please provide a valid path or ensure it's in the system PATH and executable."
            )

        self.last_run_stdout: Optional[str] = None
        self.last_run_stderr: Optional[str] = None
        self.last_run_returncode: Optional[int] = None

    def run(self,
            input_file_path: str,
            output_directory: Optional[str] = None,
            num_processors: int = 1,
            additional_args: Optional[List[str]] = None,
            moose_env_vars: Optional[Dict[str, str]] = None,
            log_file_name: Optional[str] = "log.txt") -> Tuple[bool, str, str]:  # Added log_file_name
        """
        Runs a MOOSE simulation and optionally logs STDOUT.

        Args:
            input_file_path (str): Path to the original MOOSE input file (.i).
            output_directory (Optional[str]): Directory where MOOSE simulation will be run
                                             and output files will be saved. If None,
                                             MOOSE runs in the input file's directory.
            num_processors (int): Number of processors to use for the simulation (for MPI).
            additional_args (Optional[List[str]]): A list of additional command-line arguments
                                                   to pass to the MOOSE executable.
            moose_env_vars (Optional[Dict[str, str]]): Environment variables to set for the MOOSE process.
            log_file_name (Optional[str]): Name of the file to save the STDOUT log.
                                           If None, logging is skipped. Defaults to "log.txt".
                                           The log file is saved in the execution directory (cwd).

        Returns:
            Tuple[bool, str, str]: A tuple containing:
                                   - bool: True if the simulation completed successfully (return code 0), False otherwise.
                                   - str: The standard output from the MOOSE process.
                                   - str: The standard error from the MOOSE process.
        """
        if not os.path.exists(input_file_path):
            raise FileNotFoundError(f"Original input file not found: {input_file_path}")

        original_input_file_abspath = os.path.abspath(input_file_path)
        input_file_basename = os.path.basename(original_input_file_abspath)

        if output_directory:
            os.makedirs(output_directory, exist_ok=True)
            cwd = os.path.abspath(output_directory)
            staged_input_file_path = os.path.join(cwd, input_file_basename)

            # --- BUG FIX: Check if source and destination are the same file ---
            if original_input_file_abspath != staged_input_file_path:
                try:
                    shutil.copy(original_input_file_abspath, staged_input_file_path)
                    print(f"Staged input file '{input_file_basename}' in working directory '{cwd}'")
                except shutil.SameFileError:
                    # This case is handled by the check above, but we keep it for safety.
                    print(
                        f"Input file '{input_file_basename}' is already in the working directory '{cwd}'. No copy needed.")
                except Exception as e:
                    error_message = f"Failed to copy input file to output directory: {e}"
                    print(error_message)
                    return False, "", error_message
            else:
                print(
                    f"Input file '{input_file_basename}' is already in the working directory '{cwd}'. No copy needed.")

            input_file_path_for_cmd = input_file_basename
        else:
            cwd = os.path.abspath(os.path.dirname(original_input_file_abspath) or '.')
            input_file_path_for_cmd = input_file_basename
            staged_input_file_path = original_input_file_abspath

        command = []
        if num_processors > 1:
            if not shutil.which("mpiexec"):
                raise EnvironmentError("mpiexec not found in PATH. Cannot run in parallel.")
            command.extend(["mpiexec", "-n", str(num_processors)])

        command.append(self.moose_executable_path)
        command.extend(["-i", input_file_path_for_cmd])

        if additional_args:
            command.extend(additional_args)

        safe_command_str = ' '.join(shlex.quote(str(c)) for c in command)
        print(f"Executing MOOSE command: {safe_command_str}")
        print(f"Working directory: {cwd}")

        current_env = os.environ.copy()
        if moose_env_vars:
            current_env.update(moose_env_vars)

        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=cwd,
                env=current_env
            )
            stdout, stderr = process.communicate()
            returncode = process.returncode

            self.last_run_stdout = stdout
            self.last_run_stderr = stderr
            self.last_run_returncode = returncode

            # --- Logging STDOUT ---
            if log_file_name and stdout:  # Only write if log_file_name is provided and stdout is not empty
                log_file_path = os.path.join(cwd, log_file_name)
                try:
                    with open(log_file_path, 'w') as lf:
                        lf.write(stdout)
                    print(f"STDOUT successfully written to log file: {log_file_path}")
                except IOError as e:
                    print(f"Warning: Could not write STDOUT to log file {log_file_path}: {e}")
            # --- End Logging STDOUT ---

            if output_directory and os.path.exists(
                    staged_input_file_path) and staged_input_file_path != original_input_file_abspath:
                try:
                    # os.remove(staged_input_file_path) # Optional: remove the copied .i file
                    # print(f"Cleaned up staged input file: {staged_input_file_path}")
                    pass
                except OSError as e:
                    print(f"Warning: Could not remove staged input file {staged_input_file_path}: {e}")

            if returncode == 0:
                print(
                    f"MOOSE simulation completed successfully for original input {original_input_file_abspath} (ran as {input_file_path_for_cmd} in {cwd}).")
                return True, stdout, stderr
            else:
                print(
                    f"MOOSE simulation failed for original input {original_input_file_abspath} (ran as {input_file_path_for_cmd} in {cwd}) with return code {returncode}.")
                print("--- STDOUT ---")
                print(stdout)
                print("--- STDERR ---")
                print(stderr)
                return False, stdout, stderr

        except FileNotFoundError as e:
            error_message = (f"Error running MOOSE (FileNotFoundError): {e}. "
                             f"Attempted command: {safe_command_str}. "
                             f"MOOSE executable used: {self.moose_executable_path}")
            print(error_message)
            self.last_run_stdout = ""
            self.last_run_stderr = error_message
            self.last_run_returncode = -1
            return False, "", error_message
        except Exception as e:
            error_message = (f"An unexpected error occurred while running MOOSE: {e}. "
                             f"Command: {safe_command_str}")
            print(error_message)
            self.last_run_stdout = ""
            self.last_run_stderr = error_message
            self.last_run_returncode = -1
            return False, "", error_message


if __name__ == "__main__":
    print("This is a module, intended to be imported.")
    print("To test MooseRunner, please use a separate test script.")
