# src/fiberis/moose/editor.py
# This script provides a MooseModelEditor class to programmatically modify
# a configured ModelBuilder instance from the fiberis library. It allows users to
# tweak parameters of a MOOSE model without rebuilding it from scratch, facilitating
# parameter studies and optimization tasks.

# Shenyao Jin, 10/27/2025

from __future__ import annotations
from typing import List, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from fiberis.moose.model_builder import ModelBuilder
    from fiberis.moose.input_generator import MooseBlock

class MooseModelEditor:
    """
    An editor to programmatically modify a configured ModelBuilder instance.
    This allows for parameter studies and optimization by tweaking a base model
    without rebuilding it from scratch every time.
    """

    def __init__(self, builder: ModelBuilder):
        """
        Initializes the editor with a ModelBuilder instance.
        The builder should be fully configured with a base model before editing.
        """
        self.builder = builder

    def _find_block_recursive(self, block_path: List[str], current_blocks: List[MooseBlock]) -> Optional[MooseBlock]:
        """
        Recursively searches for a nested block based on a path of names.
        """
        if not block_path:
            return None
        
        target_name = block_path[0]
        remaining_path = block_path[1:]

        for block in current_blocks:
            # Note: MOOSE block names can be paths themselves, e.g., within [BCs].
            # This simple check assumes direct name matching.
            if block.block_name == target_name:
                if not remaining_path:
                    return block  # Found the target block
                else:
                    # Continue searching in the sub-blocks of the matched block
                    return self._find_block_recursive(remaining_path, block.sub_blocks)
        return None

    def set_parameter(self, block_path: List[str], parameter_name: str, new_value: Any):
        """
        Finds a block by its path and sets or updates a parameter within it.

        Args:
            block_path (List[str]): A list of block names representing the path to the target block.
                                    e.g., ['Materials', 'permeability_Frac1']
            parameter_name (str): The name of the parameter to change. e.g., 'permeability'
            new_value (Any): The new value for the parameter.

        Raises:
            ValueError: If the block path is not found.
        """
        # The top-level blocks are stored in the builder's internal list.
        target_block = self._find_block_recursive(block_path, self.builder._top_level_blocks)

        if target_block:
            target_block.add_param(parameter_name, new_value)
            # print(f"Successfully set parameter '{parameter_name}' in block '{'/'.join(block_path)}'.")
        else:
            raise ValueError(f"Could not find a block at the specified path: {block_path}")

    def get_parameter(self, block_path: List[str], parameter_name: str) -> Any:
        """
        Finds a block by its path and retrieves the value of a parameter.

        Args:
            block_path (List[str]): The path to the target block.
            parameter_name (str): The name of the parameter to retrieve.

        Returns:
            The value of the parameter.

        Raises:
            ValueError: If the block path is not found.
            KeyError: If the parameter does not exist in the block.
        """
        target_block = self._find_block_recursive(block_path, self.builder._top_level_blocks)

        if target_block:
            if parameter_name in target_block.params:
                return target_block.params[parameter_name]
            else:
                raise KeyError(f"Parameter '{parameter_name}' not found in block '{'/'.join(block_path)}'.")
        else:
            raise ValueError(f"Could not find a block at the specified path: {block_path}")

    def print_model_structure(self):
        """
        Prints a tree-like structure of the entire model by calling the
        ModelBuilder's __str__ method. Useful for discovering the correct
        `block_path` for editing.
        """
        print(self.builder)

    def _find_param_recursive(self, parameter_name: str, current_blocks: List[MooseBlock], current_path: List[str], found_paths: List[List[str]]):
        """
        Recursively scans blocks to find all occurrences of a parameter.
        """
        for block in current_blocks:
            new_path = current_path + [block.block_name]
            if parameter_name in block.params:
                found_paths.append(new_path)
            
            # Recurse into sub-blocks
            if block.sub_blocks:
                self._find_param_recursive(parameter_name, block.sub_blocks, new_path, found_paths)

    def find_parameter(self, parameter_name: str) -> List[List[str]]:
        """
        Scans the entire model and returns a list of all block paths that
        contain the specified parameter.

        Args:
            parameter_name (str): The name of the parameter to find (e.g., 'permeability').

        Returns:
            A list of block paths, where each path is a list of strings.
            Returns an empty list if the parameter is not found anywhere.
        """
        found_paths: List[List[str]] = []
        self._find_param_recursive(parameter_name, self.builder._top_level_blocks, [], found_paths)
        return found_paths
