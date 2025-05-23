# Input file generator for MOOSE (Multiphysics Object Oriented Simulation Environment) by INL
# Shenyao Jin, 05/17/2025, shenyaojin@mines.edu
from typing import Dict, Any, List, Union

# --- 1. Define the base Block class ---
class MooseBlock:
    """
    Base class for a generic block [...] in a MOOSE input file.
    This class can be used to create blocks with parameters and sub-blocks.
    """

    def __init__(self, block_name: str, block_type: Union[str, None] = None):
        self.block_name: str = block_name
        self.block_type: Union[str, None] = block_type
        self.params: Dict[str, Any] = {}
        self.sub_blocks: List[MooseBlock] = []

    def add_param(self, key: str, value: Any):
        """Adds a parameter (key = value) to the block."""
        self.params[key] = value

    def add_sub_block(self, sub_block: 'MooseBlock'):
        """Adds a sub-block to the current block."""
        self.sub_blocks.append(sub_block)

    def _format_value(self, value: Any) -> str:
        """Formats a Python value into a MOOSE-compatible string representation."""
        if isinstance(value, bool):
            return str(value).lower()
        if isinstance(value, (int, float)):  # Numbers are not quoted
            return str(value)
        if isinstance(value, (list, tuple)):  # Lists/tuples become space-separated strings
            return ' '.join(map(str, value))
        if isinstance(value, str):
            # If the string is already explicitly quoted (single or double), pass it through.
            if (value.startswith("'") and value.endswith("'")) or \
                    (value.startswith('"') and value.endswith('"')):
                return value
            # Otherwise, quote it with single quotes (common MOOSE style for simple strings).
            return f"'{value}'"
        return str(value)  # Fallback for other types

    def render(self, indent_level: int = 0) -> str:
        """Renders the block and its sub-blocks to a string."""
        indent = '  ' * indent_level
        lines = []

        lines.append(f"{indent}[{self.block_name}]")

        current_block_param_indent = indent + "  "

        if self.block_type:
            lines.append(f"{current_block_param_indent}type = {self.block_type}")

        for key, value in self.params.items():
            if key == 'type' and self.block_type:  # type already handled
                continue
            # Special handling for 'expression' which should not be quoted by _format_value
            if key == 'expression' and isinstance(self, FunctionBlock):
                lines.append(f"{current_block_param_indent}{key} = {value}")
            else:
                lines.append(f"{current_block_param_indent}{key} = {self._format_value(value)}")

        for sub_block_item in self.sub_blocks:
            lines.append(sub_block_item.render(indent_level + 1))
            lines.append(f"{current_block_param_indent}[]")  # Closing tag for the sub_block_item

        return "\n".join(lines)


# --- Specific block classes (can be useful for more complex files) ---
class GeneratedMeshGeneratorBlock(MooseBlock):
    """Specific class for GeneratedMeshGenerator blocks for convenience."""

    def __init__(self, block_name: str, dim: int, nx: int, ny: int, xmax: float, ymax: float, **kwargs):
        super().__init__(block_name, block_type="GeneratedMeshGenerator")
        self.add_param("dim", dim)
        self.add_param("nx", nx)
        self.add_param("ny", ny)
        self.add_param("xmax", xmax)
        self.add_param("ymax", ymax)
        for key, value in kwargs.items():
            self.add_param(key, value)


class FunctionBlock(MooseBlock):
    """Specific block for MOOSE Functions, especially to handle 'expression' correctly."""

    def __init__(self, block_name: str, function_type: str, expression: str = None, **kwargs):
        super().__init__(block_name, block_type=function_type)
        if expression is not None:
            # Assign expression directly to params, it will be handled in render
            self.params["expression"] = expression
        for key, value in kwargs.items():
            if key == "expression" and expression is not None:  # Avoid double assignment
                continue
            self.add_param(key, value)


# --- 3. Main generator function (Generic Version) ---
def generate_moose_input(config: Dict[str, Any], output_filepath: str):
    """
    Generates a MOOSE input file based on the provided configuration dictionary.
    This version is generic and iterates through the config structure.
    """
    all_top_level_blocks_rendered: List[str] = []

    for block_name, block_data_config in config.items():
        # Create the top-level block object
        # Check if the block_data_config itself specifies a 'type' for the top-level block
        # This is unusual for MOOSE, top-level blocks are usually just names like [Mesh], [Variables]
        # But if needed, it could be handled here. For now, assume no 'type' for top-level.
        top_level_block = MooseBlock(block_name)

        if isinstance(block_data_config, dict):
            # This top-level block has direct parameters
            for param_key, param_value in block_data_config.items():
                top_level_block.add_param(param_key, param_value)
        elif isinstance(block_data_config, list):
            # This top-level block contains a list of sub-blocks
            for sub_block_def in block_data_config:
                sub_name = sub_block_def.get("name")
                if not sub_name:
                    raise ValueError(
                        f"Sub-block definition in '{block_name}' is missing a 'name'. Config: {sub_block_def}")

                sub_type = sub_block_def.get("type")
                sub_params_dict = sub_block_def.get("params", {})

                # Instantiate the sub-block
                # Could use FunctionBlock or other specific classes here if sub_type matches
                if sub_type == "ParsedFunction" and "expression" in sub_params_dict:  # Example for FunctionBlock
                    expression_val = sub_params_dict.pop("expression", None)  # Remove expression to pass to constructor
                    sub_block_instance = FunctionBlock(sub_name, function_type=sub_type, expression=expression_val)
                else:
                    sub_block_instance = MooseBlock(sub_name, block_type=sub_type)

                for p_key, p_val in sub_params_dict.items():
                    sub_block_instance.add_param(p_key, p_val)
                top_level_block.add_sub_block(sub_block_instance)
        else:
            raise TypeError(
                f"Configuration for block '{block_name}' must be a dict (for parameters) "
                f"or a list (for sub-blocks). Got: {type(block_data_config)}")

        all_top_level_blocks_rendered.append(top_level_block.render(0) + "\n[]")

    with open(output_filepath, 'w') as f:
        f.write("# MOOSE input file generated by input_generator.py\n\n")
        f.write("\n\n".join(all_top_level_blocks_rendered))

    print(f"MOOSE input file generated: {output_filepath}")