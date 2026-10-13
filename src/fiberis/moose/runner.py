# src/fiberis/moose/runner.py
import subprocess
import os
import shlex
import re
import shutil
from typing import Optional, Tuple, Dict, List


class MooseRunner:
    """
    A class to manage and run MOOSE simulations.
    """

    # Written to the run directory by detached runs (see run(detach=True) and check_status).
    PID_FILE_NAME = "moose_run.pid"
    EXIT_CODE_FILE_NAME = "moose_run.exitcode"

    def __init__(self, moose_executable_path: str, mpiexec_path: Optional[str] = None):
        """
        Initializes the MooseRunner.

        Args:
            moose_executable_path (str): The absolute path to the MOOSE executable
                                         (e.g., '/path/to/moose/your_app-opt' or 'your_app-opt' if in PATH).
            mpiexec_path (Optional[str]): The absolute path to the mpiexec executable.
                                          If None, it's assumed to be in the system PATH.
        """
        self._original_moose_executable_path = moose_executable_path
        self.mpiexec_path = mpiexec_path

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
        self.last_run_directory: Optional[str] = None
        self.last_log_file_name: Optional[str] = None

    def run(self,
            input_file_path: str,
            output_directory: Optional[str] = None,
            num_processors: int = 1,
            additional_args: Optional[List[str]] = None,
            moose_env_vars: Optional[Dict[str, str]] = None,
            log_file_name: Optional[str] = "log.txt",
            stream_output: bool = True,
            clean_output_dir: bool = True,
            cli_args: Optional[List[str]] = None,
            detach: bool = False) -> Tuple[bool, str, str]:
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
            stream_output (bool): If True (default), streams MOOSE output to the console in real-time.
                                  When streaming, stderr is merged into stdout.
                                  If False, output is captured and returned after the process completes.
            clean_output_dir (bool): If True (default), the output directory will be removed
                                     before the simulation to ensure a clean run. If False,
                                     existing files will be preserved.
            cli_args (Optional[List[str]]): Alias for additional_args.
            detach (bool): If True, start MOOSE as an independent background process and return
                           immediately. The run keeps going if the notebook kernel stops or VS Code
                           is closed. Output is written to the log file as the run progresses
                           (stream_output is ignored). Use check_status() to see if it is finished.

        Returns:
            Tuple[bool, str, str]: A tuple containing:
                                   - bool: True if the simulation completed successfully (return code 0), False otherwise.
                                   - str: The standard output from the MOOSE process.
                                   - str: The standard error from the MOOSE process.
                                   With detach=True: True if the run was started, a short message, and "".
        """
        # Support both naming conventions
        if cli_args and not additional_args:
            additional_args = cli_args
        if not os.path.exists(input_file_path):
            raise FileNotFoundError(f"Original input file not found: {input_file_path}")

        # Read the input file content into memory before any directory operations.
        try:
            with open(input_file_path, 'r') as f:
                input_file_content = f.read()
        except IOError as e:
            error_message = f"Fatal: Failed to read input file '{input_file_path}': {e}"
            print(error_message)
            return False, "", error_message

        input_file_basename = os.path.basename(input_file_path)
        original_input_file_abspath = os.path.abspath(input_file_path)

        if output_directory:
            # If an output directory is specified, prepare it.
            cwd = os.path.abspath(output_directory)
            if os.path.exists(cwd) and clean_output_dir:
                print(f"Output directory '{cwd}' exists. Removing it to ensure a clean run.")
                try:
                    shutil.rmtree(cwd)
                except OSError as e:
                    error_message = f"Fatal: Error removing existing output directory '{cwd}': {e}"
                    print(error_message)
                    return False, "", error_message
            
            # Recreate the directory and write the staged input file.
            os.makedirs(cwd, exist_ok=True)
            staged_input_file_path = os.path.join(cwd, input_file_basename)
            try:
                with open(staged_input_file_path, 'w') as f:
                    f.write(input_file_content)
                print(f"Staged input file '{input_file_basename}' in working directory '{cwd}'")
            except IOError as e:
                error_message = f"Fatal: Failed to write staged input file to '{staged_input_file_path}': {e}"
                print(error_message)
                return False, "", error_message
            
            input_file_path_for_cmd = input_file_basename
        else:
            # If no output directory, run in the input file's directory.
            cwd = os.path.abspath(os.path.dirname(original_input_file_abspath) or '.')
            input_file_path_for_cmd = input_file_basename
            staged_input_file_path = original_input_file_abspath

        command = []
        if num_processors > 1:
            mpiexec_executable = self.mpiexec_path or shutil.which("mpiexec")
            if not mpiexec_executable:
                raise EnvironmentError("mpiexec not found. Please provide the path to mpiexec via the 'mpiexec_path' argument or ensure it's in the system PATH.")
            command.extend([mpiexec_executable, "-n", str(num_processors)])

        command.append(self.moose_executable_path)
        command.extend(["-i", input_file_path_for_cmd, "--n-threads=1"])

        if additional_args:
            command.extend(additional_args)

        safe_command_str = ' '.join(shlex.quote(str(c)) for c in command)
        print(f"Executing MOOSE command: {safe_command_str}")
        print(f"Working directory: {cwd}")

        current_env = os.environ.copy()
        # Force single-thread per MPI rank to prevent over-subscription
        current_env.update({
            "OMP_NUM_THREADS": "1",
            "TBB_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
        })
        if moose_env_vars:
            current_env.update(moose_env_vars)

        self.last_run_directory = cwd
        self.last_log_file_name = log_file_name

        if detach:
            return self._run_detached(command, cwd, current_env, log_file_name or "log.txt")

        try:
            if stream_output:
                print("Streaming MOOSE output in real-time...")
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    cwd=cwd,
                    env=current_env,
                    universal_newlines=True
                )

                stdout_lines = []
                for line in process.stdout:
                    print(line, end='')
                    stdout_lines.append(line)

                process.stdout.close()
                returncode = process.wait()
                stdout = "".join(stdout_lines)
                stderr = ""  # Merged into stdout
            else:
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
                if stderr:
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

    def _run_detached(self, command: List[str], cwd: str, env: Dict[str, str],
                      log_file_name: str) -> Tuple[bool, str, str]:
        """
        Start MOOSE in its own session so it outlives the calling Python process.

        A small shell wrapper writes all output to the log file and, when MOOSE exits,
        writes its exit code to EXIT_CODE_FILE_NAME. The wrapper's PID goes to PID_FILE_NAME.
        """
        self.last_log_file_name = log_file_name
        log_path = os.path.join(cwd, log_file_name)
        exit_code_path = os.path.join(cwd, self.EXIT_CODE_FILE_NAME)
        pid_path = os.path.join(cwd, self.PID_FILE_NAME)
        # Stale status files from a previous run (when clean_output_dir=False) would give a wrong status.
        for path in (exit_code_path, pid_path):
            if os.path.exists(path):
                os.remove(path)

        moose_cmd = ' '.join(shlex.quote(str(c)) for c in command)
        wrapper = (f"{moose_cmd} > {shlex.quote(log_path)} 2>&1; "
                   f"echo $? > {shlex.quote(exit_code_path)}")
        try:
            process = subprocess.Popen(
                ["/bin/sh", "-c", wrapper],
                cwd=cwd,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,  # own session/process group: not killed with the kernel or terminal
            )
        except OSError as e:
            error_message = f"Failed to start detached MOOSE run: {e}"
            print(error_message)
            return False, "", error_message

        with open(pid_path, 'w') as f:
            f.write(str(process.pid))
        log_arg = "" if log_file_name == "log.txt" else f", log_file_name='{log_file_name}'"
        message = (f"MOOSE started in the background (PID {process.pid}). It will keep running if you close "
                   f"VS Code.\nLog: {log_path}\nCheck progress with runner.check_status('{cwd}'{log_arg}).")
        print(message)
        return True, message, ""

    def check_status(self, output_directory: Optional[str] = None, log_file_name: Optional[str] = None,
                     verbose: bool = True) -> str:
        """
        Report the status of a MOOSE run started with run(detach=True).

        Works from a new Python session too: pass the run's output directory.

        Args:
            output_directory (Optional[str]): The run directory. Defaults to the last run from this runner.
            log_file_name (Optional[str]): Log file name. Defaults to the last run's, or "log.txt".
            verbose (bool): If True (default), print a one-line summary (plus the log tail on failure).

        Returns:
            str: One of "running", "finished", "failed", "stopped" (process gone without an exit code,
                 e.g. killed or the server rebooted), or "not started".
        """
        cwd = os.path.abspath(output_directory) if output_directory else self.last_run_directory
        if cwd is None:
            raise ValueError("No output_directory given and this runner has not started a run.")
        log_file_name = log_file_name or (self.last_log_file_name if cwd == self.last_run_directory else None) \
            or "log.txt"
        log_path = os.path.join(cwd, log_file_name)
        exit_code_path = os.path.join(cwd, self.EXIT_CODE_FILE_NAME)
        pid_path = os.path.join(cwd, self.PID_FILE_NAME)

        log_text = ""
        if os.path.exists(log_path):
            with open(log_path, 'r', errors='replace') as f:
                log_text = f.read()
        progress = self._progress_from_log(log_text, cwd)

        if os.path.exists(exit_code_path):
            with open(exit_code_path, 'r') as f:
                exit_code = f.read().strip()
            status = "finished" if exit_code == "0" else "failed"
            summary = f"Finished successfully. {progress}" if status == "finished" else \
                f"FAILED (exit code {exit_code}). {progress}"
        elif os.path.exists(pid_path):
            with open(pid_path, 'r') as f:
                pid = int(f.read().strip())
            if self._pid_alive(pid):
                status, summary = "running", f"Running. {progress}"
            else:
                status = "stopped"
                summary = f"STOPPED: the process (PID {pid}) is gone but never reported an exit code. {progress}"
        else:
            status, summary = "not started", f"No detached run found in {cwd}."

        if verbose:
            print(summary)
            if status in ("failed", "stopped") and log_text:
                print("--- last lines of the log ---")
                print("\n".join(log_text.splitlines()[-30:]))
        return status

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        """True if a process with this PID is running on this machine (zombies count as not running)."""
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        # A killed run launched from a still-open kernel stays a zombie until the kernel reaps it.
        try:
            with open(f"/proc/{pid}/stat", 'r') as f:
                state = f.read().rsplit(')', 1)[1].split()[0]
            return state != 'Z'
        except (OSError, IndexError):
            return True

    @staticmethod
    def _progress_from_log(log_text: str, cwd: str) -> str:
        """Describe progress using the last 'Time Step N, time = T' line and the [Executioner] end_time."""
        steps = re.findall(r"Time Step\s+(\d+), time = ([0-9.eE+-]+)", log_text)
        if not steps:
            return "No time steps completed yet."
        step, t = int(steps[-1][0]), float(steps[-1][1])
        end_time = None
        for name in sorted(os.listdir(cwd)):
            if name.endswith(".i"):
                with open(os.path.join(cwd, name), 'r') as f:
                    # Only look inside [Executioner] (other blocks, e.g. [Controls], also have end_time).
                    match = re.search(r"^\[Executioner\](.*?)^\[\]", f.read(), re.MULTILINE | re.DOTALL)
                if match:
                    end_match = re.search(r"^\s*end_time\s*=\s*'?([0-9.eE+-]+)", match.group(1), re.MULTILINE)
                    if end_match:
                        end_time = float(end_match.group(1))
                        break
        if end_time:
            return f"Time step {step}, t = {t:g} s of {end_time:g} s ({100 * t / end_time:.0f}%)."
        return f"Time step {step}, t = {t:g} s."


if __name__ == "__main__":
    print("This is a module, intended to be imported.")
    print("To test MooseRunner, please use a separate test script.")
