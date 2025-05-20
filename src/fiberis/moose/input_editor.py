# src/fiberis/moose/input_editor.py
import os
from typing import Any, Optional, Callable

# Attempt to import pyhit and moosetree.
# These are typically part of the MOOSE framework's Python environment.
# Ensure MOOSE environment is active or PYTHONPATH is set correctly.
try:
    import pyhit
    import moosetree
except ImportError as e:
    print(
        "Error: Failed to import 'pyhit' or 'moosetree'. "
        "Please ensure your MOOSE environment is activated, "
        "or the MOOSE Python utilities are in your PYTHONPATH. "
        f"Details: {e}"
    )


    # Define dummy placeholders if imports fail, to allow parsing of this file.
    # Actual usage of the editor will fail if imports are not successful.
    class NodePlaceholder:
        pass


    pyhit = NodePlaceholder()  # type: ignore
    moosetree = NodePlaceholder()  # type: ignore
    pyhit.Node = type("Node", (object,), {})  # type: ignore


class MooseInputEditor:
    """
    A class to read, modify, and write MOOSE input files using pyhit and moosetree.
    """

    def __init__(self, input_file_path: Optional[str] = None):
        """
        Initializes the MooseInputEditor.

        Args:
            input_file_path (Optional[str]): Path to a MOOSE input file to load immediately.
        """
        self.root: Optional[pyhit.Node] = None  # type: ignore[name-defined] # Root node of the parsed input file
        self.input_file_path: Optional[str] = None

        if input_file_path:
            self.load_file(input_file_path)

    def load_file(self, input_file_path: str) -> None:
        """
        Loads a MOOSE input file.

        Args:
            input_file_path (str): Path to the MOOSE input file (.i).

        Raises:
            FileNotFoundError: If the input file does not exist.
            RuntimeError: If pyhit fails to load the file or is not available.
        """
        if not hasattr(pyhit, 'load'):
            raise RuntimeError("pyhit module is not correctly imported or available.")

        if not os.path.exists(input_file_path):
            raise FileNotFoundError(f"Input file not found: {input_file_path}")

        try:
            self.root = pyhit.load(input_file_path)  # type: ignore[attr-defined]
            self.input_file_path = input_file_path
            print(f"Successfully loaded MOOSE input file: {input_file_path}")
        except Exception as e:
            self.root = None
            self.input_file_path = None
            raise RuntimeError(f"Failed to load MOOSE input file '{input_file_path}' using pyhit: {e}")

    def find_node(self, path: str) -> Optional[pyhit.Node]:  # type: ignore[name-defined]
        """
        Finds a node in the loaded MOOSE input file structure by its path.
        This version uses manual traversal of the tree.

        Args:
            path (str): The full path to the node (e.g., '/Mesh/gen' or '/Variables/diffused').
                        The leading '/' indicates the root node (which is typically unnamed).

        Returns:
            Optional[pyhit.Node]: The found node, or None if not found.

        Raises:
            ValueError: If the file is not loaded or the path format is incorrect.
        """
        if not self.root:
            raise ValueError("No MOOSE input file loaded. Call load_file() first.")

        if not path.startswith('/'):
            raise ValueError("Path must be absolute, starting with '/' (e.g., '/Mesh/gen').")

        # Remove leading '/' and filter out empty parts (e.g., from '//')
        path_parts = [part for part in path.split('/') if part]

        current_node = self.root
        for part_name in path_parts:
            found_child = None
            # pyhit.Node objects (which inherit from moosetree.Node)
            # are iterable and yield their children (sub-blocks).
            # Parameters are accessed via dict-like interface (node['param_name']).
            if hasattr(current_node, '__iter__'):  # Check if the node is iterable for children
                try:
                    for child_node in current_node:  # type: ignore[attr-defined]
                        # We are looking for child *blocks*, which are also pyhit.Node instances
                        # and have a 'name' attribute.
                        if isinstance(child_node, pyhit.Node) and hasattr(child_node,
                                                                          'name') and child_node.name == part_name:  # type: ignore[name-defined]
                            found_child = child_node
                            break
                except Exception as e_iter:
                    # This might catch issues if iteration itself is problematic
                    print(
                        f"Warning: Error iterating children of node '{current_node.name if hasattr(current_node, 'name') else 'Unnamed'}': {e_iter}")
                    return None

            if found_child:
                current_node = found_child
            else:
                # print(f"Path part '{part_name}' not found under the current scope.")
                return None  # Path part not found

        return current_node

    def get_param_value(self, node_path: str, param_name: str) -> Any:
        """
        Gets the value of a parameter from a specified node.

        Args:
            node_path (str): Full path to the node (e.g., '/Mesh/gen').
            param_name (str): Name of the parameter.

        Returns:
            Any: The value of the parameter.

        Raises:
            ValueError: If the node or parameter is not found.
        """
        node = self.find_node(path=node_path)
        if node is None:
            raise ValueError(f"Node not found at path: {node_path}")

        if param_name not in node:  # pyhit.Node allows dict-like access for parameters
            available_params = []
            if hasattr(node, 'keys'):  # Check if it has dict-like keys method
                available_params = list(node.keys())  # type: ignore[attr-defined]
            raise ValueError(
                f"Parameter '{param_name}' not found in node '{node_path}'. Available params: {available_params}")
        return node[param_name]  # type: ignore[misc]

    def set_param_value(self, node_path: str, param_name: str, new_value: Any) -> None:
        """
        Sets the value of a parameter on a specified node.

        Args:
            node_path (str): Full path to the node (e.g., '/Mesh/gen').
            param_name (str): Name of the parameter.
            new_value (Any): The new value for the parameter. pyhit handles type conversion.

        Raises:
            ValueError: If the node is not found.
        """
        node = self.find_node(path=node_path)
        if node is None:
            raise ValueError(f"Node not found at path: {node_path}")

        node[param_name] = new_value  # type: ignore[misc]
        print(f"Set parameter '{param_name}' in node '{node_path}' to: {new_value}")

    def set_param_comment(self, node_path: str, param_name: str, comment: str) -> None:
        """
        Sets the comment for a parameter on a specified node.

        Args:
            node_path (str): Full path to the node (e.g., '/Mesh/gen').
            param_name (str): Name of the parameter to comment.
            comment (str): The comment string.

        Raises:
            ValueError: If the node is not found or does not support comments for the parameter.
            AttributeError: If the node object does not have 'setComment'.
        """
        node = self.find_node(path=node_path)
        if node is None:
            raise ValueError(f"Node not found at path: {node_path}")

        if not hasattr(node, 'setComment'):
            raise AttributeError(
                f"Node at '{node_path}' (type: {type(node)}) does not support setComment method. Ensure pyhit.Node is correctly loaded.")

        try:
            # First, ensure the parameter exists if setComment requires it
            if param_name not in node:  # type: ignore[operator]
                print(
                    f"Warning: Parameter '{param_name}' does not exist in '{node_path}'. Comment might not be settable or might apply to the block if param_name is empty.")
                # Some implementations of setComment might allow block-level comments if param_name is None/empty

            node.setComment(param_name, comment)  # type: ignore[attr-defined]
            print(f"Set comment for parameter '{param_name}' in node '{node_path}' to: '{comment}'")
        except Exception as e:
            raise ValueError(f"Failed to set comment for '{param_name}' in '{node_path}': {e}")

    def add_param(self, node_path: str, param_name: str, value: Any, comment: Optional[str] = None) -> None:
        """
        Adds a new parameter to a specified node. If the parameter exists, it's updated.

        Args:
            node_path (str): Full path to the node (e.g., '/Mesh/gen').
            param_name (str): Name of the new parameter.
            value (Any): Value of the new parameter.
            comment (Optional[str]): Optional comment for the new parameter.

        Raises:
            ValueError: If the node is not found.
        """
        node = self.find_node(path=node_path)
        if node is None:
            raise ValueError(f"Node not found at path: {node_path}")

        node[param_name] = value  # type: ignore[misc]
        print(f"Added/Updated parameter '{param_name}' in node '{node_path}' to: {value}")
        if comment:
            try:
                self.set_param_comment(node_path, param_name, comment)
            except (ValueError, AttributeError) as e:
                print(f"Warning: Could not set comment for new param '{param_name}' in '{node_path}': {e}")

    def write_file(self, output_file_path: str) -> None:
        """
        Writes the (potentially modified) MOOSE input structure to a new file.

        Args:
            output_file_path (str): Path to save the output .i file.

        Raises:
            ValueError: If no file has been loaded (self.root is None).
            RuntimeError: If pyhit fails to write the file or is not available.
        """
        if not self.root:
            raise ValueError("No MOOSE input file loaded or modified. Cannot write.")
        if not hasattr(pyhit, 'write'):
            raise RuntimeError("pyhit module is not correctly imported or available for writing.")

        try:
            # Ensure output directory exists
            output_dir = os.path.dirname(output_file_path)
            if output_dir:  # Check if dirname is not empty (e.g. for files in current dir)
                os.makedirs(output_dir, exist_ok=True)

            pyhit.write(output_file_path, self.root)  # type: ignore[attr-defined]
            print(f"Successfully wrote modified MOOSE input to: {output_file_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to write MOOSE input file '{output_file_path}' using pyhit: {e}")


# --- Example Usage (for testing this module directly) ---
if __name__ == "__main__":
    # Create a dummy input.i file for the example, similar to pyhit documentation
    dummy_input_content = """
[Mesh]
  [gen]
    type = GeneratedMeshGenerator # This is a sub-block of Mesh
    dim = 1
    xmax = 3 # Original xmax
  [] # Closes [gen]
[] # Closes [Mesh]

[Variables]
  [diffused] # This is a sub-block of Variables
    order = FIRST
  [] # Closes [diffused]
[] # Closes [Variables]

[Kernels]
  [diff]
    type = ADDiffusion
    variable = u
  []
[]

[Executioner]
  type = Steady
[]
"""
    test_dir = "pyhit_editor_test_files_v2"  # Use a different test dir name
    os.makedirs(test_dir, exist_ok=True)
    original_file = os.path.join(test_dir, "original_input_v2.i")
    modified_file = os.path.join(test_dir, "modified_input_v2.i")

    with open(original_file, "w") as f:
        f.write(dummy_input_content)
    print(f"Created dummy input file: {original_file}")

    if not (hasattr(pyhit, 'load') and hasattr(moosetree, 'Node')):  # Check if pyhit/moosetree seem available
        print("\nSkipping MooseInputEditor test as pyhit/moosetree are not properly available.")
    else:
        try:
            print("\n--- Running MooseInputEditor Test ---")
            editor = MooseInputEditor()

            editor.load_file(original_file)

            # Test paths based on the structure of dummy_input_content
            mesh_gen_node_path = '/Mesh/gen'
            var_diffused_node_path = '/Variables/diffused'
            executioner_node_path = '/Executioner'

            print(f"\nAttempting to get 'xmax' from '{mesh_gen_node_path}'...")
            current_xmax = editor.get_param_value(mesh_gen_node_path, "xmax")
            print(f"Current xmax in '{mesh_gen_node_path}': {current_xmax}")

            editor.set_param_value(mesh_gen_node_path, "xmax", 4)
            editor.set_param_comment(mesh_gen_node_path, "xmax", "Changed from 3 to 4 by script")

            print(f"\nAttempting to add 'initial_condition' to '{var_diffused_node_path}'...")
            editor.add_param(var_diffused_node_path, 'initial_condition', 0.5, comment="Set initial condition")

            print(f"\nAttempting to set comment for 'type' in '{executioner_node_path}'...")
            editor.set_param_comment(executioner_node_path, 'type', "Steady-state execution")

            editor.add_param(executioner_node_path, "solve_type", "'NEWTON'")  # Adding solve_type

            editor.write_file(modified_file)

            print(f"\n--- Content of {modified_file} ---")
            with open(modified_file, "r") as f_mod:
                print(f_mod.read())
            print("--- End of content ---")

        except RuntimeError as e:
            print(f"RuntimeError during editor usage: {e}")
        except FileNotFoundError as e:
            print(f"FileNotFoundError during editor usage: {e}")
        except ValueError as e:
            print(f"ValueError during editor usage: {e}")
        except AttributeError as e:
            print(f"AttributeError during editor usage (likely pyhit/moosetree issue or incorrect node access): {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            import traceback

            traceback.print_exc()
