# User layer of input generator. Allow user to specify model parameters without
# having to interact with the input file directly.
# Higher-level API for building MOOSE input files, compared to input_generator.py and input_editor.py.
# Shenyao Jin, shenyaojin@mines.edu, 05/24/2025
# fiberis/moose/model_builder.py
from typing import List, Dict, Any, Union, Tuple, Optional

# Import config classes
from fiberis.moose.config import HydraulicFractureConfig, SRVConfig

# Import the core generation function from your input_generator.py
from fiberis.moose.input_generator import generate_moose_input


class ModelBuilder:
    """
    This class provides a high-level API to construct MOOSE input files.
    It uses configuration objects (HydraulicFractureConfig, SRVConfig)
    to define physical features and translates them into MOOSE mesh operations
    and other input file blocks.
    """

    def __init__(self, project_name: str):
        """
        Initializes the ModelBuilder.

        Args:
            project_name (str): The name of the project.
        """
        self.project_name: str = project_name
        self._mesh_sub_blocks: List[Dict[str, Any]] = []
        self._simple_mesh_params: Optional[Dict[str, Any]] = None
        self._other_top_level_blocks_data: Dict[str, Any] = {}
        self._last_mesh_op_name: Optional[str] = None
        self._block_id_to_name_map: Dict[int, str] = {}  # To store numeric ID to final name mapping
        self._next_available_block_id: int = 1  # Start assigning from 1 (0 is often default matrix)
        self._main_domain_block_id: int = 0  # Default block ID for the main domain/matrix

    # --- Internal Helper Methods for Low-Level Mesh Operations ---
    def _add_mesh_sub_block_internal(self, name: str, block_type: str, params: Dict[str, Any],
                                     is_start_node: bool = False,
                                     explicit_input: Optional[str] = None) -> str:
        """
        Internal helper to add a mesh sub-block definition.
        Manages the self._last_mesh_op_name chain.
        Returns the name of the added operation.
        """
        sub_block_def = {"name": name, "type": block_type, "params": params.copy()}

        if explicit_input is not None:
            sub_block_def['params']['input'] = explicit_input
        elif not is_start_node and 'input' not in sub_block_def['params']:
            if self._last_mesh_op_name is None:
                if block_type not in ["StitchedMeshGenerator"]:  # StitchedMeshGenerator uses 'inputs'
                    raise ValueError(
                        f"Cannot automatically determine 'input' for mesh operation '{name}' ({block_type}). "
                        "Ensure a previous mesh operation exists or provide 'input_mesh_name' explicitly."
                    )
            elif block_type not in ["StitchedMeshGenerator"]:
                sub_block_def['params']['input'] = self._last_mesh_op_name

        self._mesh_sub_blocks.append(sub_block_def)
        self._last_mesh_op_name = name
        self._simple_mesh_params = None  # Complex mesh ops override simple mesh
        return name

    def _generate_unique_op_name(self, base_name: str) -> str:
        """Generates a unique name for a mesh operation if needed."""
        # Simple counter for now, can be made more robust
        count = 1
        op_name = base_name
        existing_names = {op['name'] for op in self._mesh_sub_blocks}
        while op_name in existing_names:
            op_name = f"{base_name}_{count}"
            count += 1
        return op_name

    # --- High-Level API Methods ---

    def set_main_domain_parameters_2d(self,
                                      domain_name: str,  # Will be used for renaming block 0
                                      length: float,  # X-dimension
                                      height: float,  # Y-dimension
                                      num_elements_x: int,
                                      num_elements_y: int,
                                      xmin: float = 0.0,
                                      ymin: float = 0.0,
                                      moose_block_id: int = 0,  # The ID for the main domain in MOOSE
                                      **additional_generator_params) -> 'ModelBuilder':
        """
        Defines the main 2D reservoir domain (matrix) using a GeneratedMeshGenerator.

        Args:
            domain_name (str): Name for the main reservoir domain (e.g., "matrix").
                               This name will be used to rename the 'moose_block_id' later.
            length (float): Total length of the matrix domain (X-dimension).
            height (float): Total height of the matrix domain (Y-dimension).
            num_elements_x (int): Number of mesh elements (nx) along the length.
            num_elements_y (int): Number of mesh elements (ny) along the height.
            xmin (float, optional): Minimum x-coordinate. Defaults to 0.0.
            ymin (float, optional): Minimum y-coordinate. Defaults to 0.0.
            moose_block_id (int, optional): The integer ID for this main domain in MOOSE.
                                           Defaults to 0. This ID will be associated with 'domain_name'.
            **additional_generator_params: Other parameters for GeneratedMeshGenerator.

        Returns:
            ModelBuilder: Returns self for chaining.
        """
        self._main_domain_block_id = moose_block_id
        self._block_id_to_name_map[self._main_domain_block_id] = domain_name

        mesh_op_name = self._generate_unique_op_name(f"{domain_name}_base_mesh")
        params = {
            "dim": 2,
            "nx": num_elements_x,
            "ny": num_elements_y,
            "xmin": xmin,
            "xmax": xmin + length,
            "ymin": ymin,
            "ymax": ymin + height,
            **additional_generator_params
        }
        self._add_mesh_sub_block_internal(mesh_op_name, "GeneratedMeshGenerator", params, is_start_node=True)
        return self

    def add_hydraulic_fracture_2d(self,
                                  fracture_config: HydraulicFractureConfig,
                                  target_moose_block_id: Optional[int] = None,
                                  input_mesh_name: Optional[str] = None,
                                  refinement_passes: Optional[int] = None) -> 'ModelBuilder':
        """
        Adds a hydraulic fracture to the 2D model based on HydraulicFractureConfig.
        Currently assumes axis-aligned fracture (orientation_angle = 0).

        Args:
            fracture_config (HydraulicFractureConfig): Configuration object for the fracture.
            target_moose_block_id (Optional[int], optional): The numeric block ID to assign to this fracture in MOOSE.
                                                            If None, an ID will be automatically assigned.
            input_mesh_name (Optional[str], optional): Name of the mesh operation that serves as input.
                                                       If None, uses the output of the last mesh operation.
            refinement_passes (Optional[int], optional): Number of refinement passes (e.g., 1 for one level of 1x1 refinement).
                                                          Defaults to None (no refinement).

        Returns:
            ModelBuilder: Returns self for chaining.
        """
        if fracture_config.orientation_angle != 0.0:
            print(f"Warning: HydraulicFractureConfig '{fracture_config.name}' has orientation_angle = "
                  f"{fracture_config.orientation_angle}°. This version of add_hydraulic_fracture_2d "
                  "currently only supports axis-aligned fractures (0°). The fracture will be generated as axis-aligned.")

        # Assign block ID
        block_id_to_use = target_moose_block_id
        if block_id_to_use is None:
            block_id_to_use = self._next_available_block_id
            self._next_available_block_id += 1
        elif block_id_to_use == self._main_domain_block_id or block_id_to_use in self._block_id_to_name_map:
            # If ID is already in use (and not for re-assignment with a new name, which is complex here)
            # or conflicts with main domain ID being re-purposed without explicit intent.
            print(f"Warning: target_moose_block_id {block_id_to_use} for fracture '{fracture_config.name}' "
                  "is already in use or is the main domain ID. Assigning a new ID: {self._next_available_block_id}")
            block_id_to_use = self._next_available_block_id
            self._next_available_block_id += 1

        self._block_id_to_name_map[block_id_to_use] = fracture_config.name

        # Calculate bounding box for axis-aligned fracture
        # For a 2D fracture, 'height' is its aperture/thickness in the y-direction of its local coord system
        half_length = fracture_config.length / 2.0
        half_height = fracture_config.height / 2.0  # This is the aperture

        # Assuming axis-aligned (orientation_angle = 0)
        xmin = fracture_config.center_x - half_length
        xmax = fracture_config.center_x + half_length
        ymin = fracture_config.center_y - half_height
        ymax = fracture_config.center_y + half_height

        # Using a very small z-range for 2D problems in MOOSE's 3D BBox generator
        # This might need adjustment based on how MOOSE handles 2D with this generator.
        # Or, ensure the main domain is truly 2D (dim=2 in GeneratedMeshGenerator)
        # and that SubdomainBoundingBoxGenerator correctly interprets 2D coords.
        # For now, let's assume we are working in XY plane, z is negligible.
        # MOOSE often uses a small z thickness for 2D problems if using 3D meshers.
        # If the base mesh is strictly 2D, then z_min/z_max might be ignored or cause issues.
        # Let's assume for now that the BBox generator handles 2D coordinates gracefully.
        # If the base mesh is 2D, 'z' coordinates in BBox are often ignored or should be omitted.
        # For safety with SubdomainBoundingBoxGenerator, we provide a tiny z-extent.
        # This is a common MOOSE pattern if the underlying mesh might have some z-component or if the generator expects 3D coords.
        z_epsilon_bottom = "0.00000001"  # Placeholder, actual value might not matter if truly 2D
        z_epsilon_top = "0"  # Placeholder

        bbox_op_name = self._generate_unique_op_name(f"{fracture_config.name}_bbox")
        bbox_params = {
            "bottom_left": f"{xmin} {ymin} {z_epsilon_bottom}",
            "top_right": f"{xmax} {ymax} {z_epsilon_top}",
            "block_id": block_id_to_use
        }
        current_input = input_mesh_name if input_mesh_name is not None else self._last_mesh_op_name
        self._add_mesh_sub_block_internal(bbox_op_name, "SubdomainBoundingBoxGenerator", bbox_params,
                                          explicit_input=current_input)

        if refinement_passes is not None and refinement_passes > 0:
            refine_op_name = self._generate_unique_op_name(f"{fracture_config.name}_refine")
            # Assuming refinement_passes means (N, N) refinement for 2D.
            # MOOSE's RefineBlockGenerator refinement param is often like '1 1' for one level of 2x2 subdivision.
            # We'll interpret 'refinement_passes' as the number for both directions.
            refinement_str = f"{refinement_passes} {refinement_passes}"
            refine_params = {
                "block": str(block_id_to_use),  # Target the newly created block
                "refinement": refinement_str
            }
            self._add_mesh_sub_block_internal(refine_op_name, "RefineBlockGenerator", refine_params,
                                              explicit_input=self._last_mesh_op_name)

        return self

    def add_srv_zone_2d(self,
                        srv_config: SRVConfig,
                        target_moose_block_id: Optional[int] = None,
                        input_mesh_name: Optional[str] = None,
                        refinement_passes: Optional[int] = None) -> 'ModelBuilder':
        """
        Adds an SRV (Stimulated Reservoir Volume) zone to the 2D model.

        Args:
            srv_config (SRVConfig): Configuration object for the SRV zone.
            target_moose_block_id (Optional[int], optional): The numeric block ID to assign. Auto-assigned if None.
            input_mesh_name (Optional[str], optional): Name of the input mesh operation. Uses last if None.
            refinement_passes (Optional[int], optional): Number of refinement passes. Defaults to None.

        Returns:
            ModelBuilder: Returns self for chaining.
        """
        block_id_to_use = target_moose_block_id
        if block_id_to_use is None:
            block_id_to_use = self._next_available_block_id
            self._next_available_block_id += 1
        elif block_id_to_use == self._main_domain_block_id or block_id_to_use in self._block_id_to_name_map:
            print(f"Warning: target_moose_block_id {block_id_to_use} for SRV '{srv_config.name}' "
                  "is already in use or is the main domain ID. Assigning a new ID: {self._next_available_block_id}")
            block_id_to_use = self._next_available_block_id
            self._next_available_block_id += 1

        self._block_id_to_name_map[block_id_to_use] = srv_config.name

        half_length = srv_config.length / 2.0
        half_height = srv_config.height / 2.0  # SRV 'height' corresponds to y-dimension here

        xmin = srv_config.center_x - half_length
        xmax = srv_config.center_x + half_length
        ymin = srv_config.center_y - half_height
        ymax = srv_config.center_y + half_height

        z_epsilon_bottom = "0.00000001"
        z_epsilon_top = "0"

        bbox_op_name = self._generate_unique_op_name(f"{srv_config.name}_bbox")
        bbox_params = {
            "bottom_left": f"{xmin} {ymin} {z_epsilon_bottom}",
            "top_right": f"{xmax} {ymax} {z_epsilon_top}",
            "block_id": block_id_to_use
        }
        current_input = input_mesh_name if input_mesh_name is not None else self._last_mesh_op_name
        self._add_mesh_sub_block_internal(bbox_op_name, "SubdomainBoundingBoxGenerator", bbox_params,
                                          explicit_input=current_input)

        if refinement_passes is not None and refinement_passes > 0:
            refine_op_name = self._generate_unique_op_name(f"{srv_config.name}_refine")
            refinement_str = f"{refinement_passes} {refinement_passes}"
            refine_params = {
                "block": str(block_id_to_use),
                "refinement": refinement_str
            }
            self._add_mesh_sub_block_internal(refine_op_name, "RefineBlockGenerator", refine_params,
                                              explicit_input=self._last_mesh_op_name)
        return self

    # --- Methods for adding other MOOSE blocks (Variables, Kernels, etc.) ---
    # These methods will be similar to the previous version, taking structured lists/dicts.

    def add_variables_block(self, variable_definitions: List[Dict[str, Any]]) -> 'ModelBuilder':
        self._other_top_level_blocks_data["Variables"] = variable_definitions
        return self

    def add_kernels_block(self, kernel_definitions: List[Dict[str, Any]]) -> 'ModelBuilder':
        self._other_top_level_blocks_data["Kernels"] = kernel_definitions
        return self

    def add_bcs_block(self, bc_definitions: List[Dict[str, Any]]) -> 'ModelBuilder':
        self._other_top_level_blocks_data["BCs"] = bc_definitions
        return self

    def add_materials_block(self, material_definitions: List[Dict[str, Any]]) -> 'ModelBuilder':
        """
        Adds a [Materials] block.
        Each definition should be like:
        {"name": "matrix_props", "type": "GenericConstantMaterial", "params": {"prop_names": "permeability_x porosity", "prop_values": "1e-15 0.2"}}
        Or for multiple materials under [Materials]:
        [
            {"name": "matrix_props", ...},
            {"name": "fracture_props", ...}
        ]
        The 'params' for each material should include 'block' to specify which mesh block(s) it applies to.
        Example: "params": {"block": "matrix_block_name fracture_block_name", ...}
        The block names here should be the final string names (e.g., "matrix", "MainFracture").
        The ModelBuilder will use the _block_id_to_name_map to resolve these if needed,
        or the user can provide the final names directly if they know them.
        For simplicity, we assume the user provides the material properties with the correct
        final block names in their 'params'.
        """
        self._other_top_level_blocks_data["Materials"] = material_definitions
        return self

    def add_functions_block(self, function_definitions: List[Dict[str, Any]]) -> 'ModelBuilder':
        self._other_top_level_blocks_data["Functions"] = function_definitions
        return self

    def add_executioner_block(self, exec_type: Optional[str] = None, **params) -> 'ModelBuilder':
        exec_params = {}
        if exec_type:
            exec_params["type"] = exec_type
        exec_params.update(params)
        self._other_top_level_blocks_data["Executioner"] = exec_params
        return self

    def add_outputs_block(self, output_definitions: Union[List[Dict[str, Any]], Dict[str, Any]]) -> 'ModelBuilder':
        self._other_top_level_blocks_data["Outputs"] = output_definitions
        return self

    def add_nodeset_by_coord(self, nodeset_op_name: str,
                             new_boundary_name: str,
                             coordinates: Union[Tuple[float, ...], str],
                             input_mesh_name: Optional[str] = None,
                             **additional_params) -> 'ModelBuilder':
        """
        Adds an ExtraNodesetGenerator block by coordinates.
        This is a lower-level mesh operation but useful for defining points for BCs.
        """
        params = {
            "new_boundary": new_boundary_name,
            "coord": ' '.join(map(str, coordinates)) if isinstance(coordinates, tuple) else coordinates,
            **additional_params
        }
        current_input = input_mesh_name if input_mesh_name is not None else self._last_mesh_op_name
        self._add_mesh_sub_block_internal(nodeset_op_name, "ExtraNodesetGenerator", params,
                                          explicit_input=current_input)
        return self

    # --- File Generation ---
    def _finalize_mesh_block(self):
        """
        Adds a RenameBlockGenerator at the end of the mesh operations
        if there are mappings in _block_id_to_name_map.
        """
        if self._block_id_to_name_map and self._mesh_sub_blocks:  # Only add if there are mesh ops and names to map
            # Check if a RenameBlockGenerator for these exact mappings already exists to prevent duplicates.
            # This is a simple check; more complex logic might be needed if multiple renames are allowed.
            # For now, we assume one final rename operation.
            existing_rename_op = None
            for op in self._mesh_sub_blocks:
                if op["type"] == "RenameBlockGenerator":
                    # A more robust check would compare old_block and new_block lists.
                    # For simplicity, if one exists, we assume it's the one we want or needs updating.
                    # However, it's safer to just add one at the very end based on the final map.
                    # Let's remove any pre-existing one and add a fresh one.
                    # This part is tricky if user adds RenameBlockGenerator manually.
                    # Safest: only add our own if no user-defined one seems to cover these.
                    # For now, let's just add it if _block_id_to_name_map is populated.
                    pass  # We will add a new one based on the complete map.

            old_block_ids = []
            new_block_names_str = []

            # Ensure consistent ordering for reproducible output
            sorted_block_ids = sorted(self._block_id_to_name_map.keys())

            for block_id in sorted_block_ids:
                old_block_ids.append(str(block_id))
                new_block_names_str.append(self._block_id_to_name_map[block_id])

            if old_block_ids:  # If there's anything to rename
                rename_op_name = self._generate_unique_op_name("final_rename")
                rename_params = {
                    "old_block": ' '.join(old_block_ids),
                    "new_block": ' '.join(new_block_names_str)
                }
                # This rename should operate on the very last mesh state
                self._add_mesh_sub_block_internal(rename_op_name, "RenameBlockGenerator", rename_params,
                                                  explicit_input=self._last_mesh_op_name)

    def get_config_dict(self) -> Dict[str, Any]:
        """
        Builds and returns the total configuration dictionary.
        """
        self._finalize_mesh_block()  # Ensure RenameBlockGenerator is added if needed

        final_config: Dict[str, Any] = {}
        if self._mesh_sub_blocks:
            final_config["Mesh"] = self._mesh_sub_blocks
        elif self._simple_mesh_params is not None:
            final_config["Mesh"] = self._simple_mesh_params

        final_config.update(self._other_top_level_blocks_data)
        return final_config

    def generate_input_file(self, output_filepath: str):
        """
        Generates the MOOSE input file.
        """
        config_dict = self.get_config_dict()
        if not config_dict:
            raise ValueError("No blocks defined in the model. Cannot generate an empty input file.")

        generate_moose_input(config_dict, output_filepath)
        print(f"MOOSE input file generated by ModelBuilder: {output_filepath}")

    # --- Example Usage Method ---
    @staticmethod
    def build_example_with_configs(output_filepath: str = "example_with_configs.i"):
        builder = ModelBuilder(project_name="FracSRVExample")

        # 1. Define main domain
        builder.set_main_domain_parameters_2d(
            domain_name="matrix",
            length=1000, height=500,
            num_elements_x=100, num_elements_y=50,
            moose_block_id=0  # This will be renamed to "matrix"
        )

        # 2. Define SRV
        srv1_conf = SRVConfig(
            name="SRV1",  # This will be the final block name
            length=300, height=80,
            center_x=500, center_y=250
        )
        # Assign a specific numeric ID for SRV1, e.g., 1
        # This ID will be renamed to "SRV1"
        builder.add_srv_zone_2d(srv_config=srv1_conf, target_moose_block_id=1, refinement_passes=1)

        # 3. Define Fracture
        frac1_conf = HydraulicFractureConfig(
            name="Frac1",  # This will be the final block name
            length=200, height=0.2,  # height is aperture
            center_x=500, center_y=250,
            orientation_angle=0  # Axis-aligned
        )
        # Assign a specific numeric ID for Frac1, e.g., 2
        # This ID will be renamed to "Frac1"
        builder.add_hydraulic_fracture_2d(fracture_config=frac1_conf, target_moose_block_id=2, refinement_passes=2)

        # 4. Add an injection point (Nodeset)
        # The input for this nodeset will be the last mesh operation (likely the refinement of Frac1)
        builder.add_nodeset_by_coord(
            nodeset_op_name="injection_well_nodes",
            new_boundary_name="injection_well",  # This is the MOOSE boundary name
            coordinates=(500, 250)  # Coordinates of the injection point
        )

        # 5. Add Variables
        builder.add_variables_block([
            {"name": "pp", "params": {"order": "FIRST", "family": "LAGRANGE"}}  # Pore pressure
        ])

        # 6. Add Kernels (Example: Diffusion for pore pressure)
        builder.add_kernels_block([
            {"name": "pp_diffusion", "type": "Diffusion", "params": {"variable": "pp"}}
        ])

        # 7. Add Materials (Example: Permeability for matrix, SRV, Frac)
        # User needs to ensure block names here match final names ('matrix', 'SRV1', 'Frac1')
        builder.add_materials_block([
            {
                "name": "matrix_material",
                "type": "GenericConstantMaterial",
                "params": {
                    "block": "matrix",  # Final name
                    "prop_names": "permeability",
                    "prop_values": "1e-15"  # Matrix permeability
                }
            },
            {
                "name": "srv1_material",
                "type": "GenericConstantMaterial",
                "params": {
                    "block": "SRV1",  # Final name
                    "prop_names": "permeability",
                    "prop_values": "1e-13"  # SRV1 permeability
                }
            },
            {
                "name": "frac1_material",
                "type": "GenericConstantMaterial",
                "params": {
                    "block": "Frac1",  # Final name
                    "prop_names": "permeability",
                    "prop_values": "1e-10"  # Frac1 permeability
                }
            }
        ])

        # 8. Add Boundary Conditions (Example: Fixed pressure at injection well)
        # First, define a function for the BC value if it's not constant
        builder.add_functions_block([
            {"name": "injection_pressure_value", "type": "ParsedFunction", "params": {"expression": "10e6"}}  # 10 MPa
        ])
        builder.add_bcs_block([
            {
                "name": "injection_bc",
                "type": "FunctionDirichletBC",
                "params": {
                    "variable": "pp",
                    "boundary": "injection_well",  # Matches new_boundary_name from add_nodeset_by_coord
                    "function": "injection_pressure_value"
                }
            }
        ])

        # 9. Add Executioner
        builder.add_executioner_block(exec_type="Steady", solve_type="PJFNK")

        # 10. Add Outputs
        builder.add_outputs_block({"exodus": True, "file_base": "hydraulic_fracturing_sim"})

        # Generate the file
        builder.generate_input_file(output_filepath)
        print(f"Example file with configs generated: {output_filepath}")


if __name__ == '__main__':
    import os

    # Ensure the directory for the output file exists
    output_dir = "test_files/moose_input_file_test"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    example_output_file = os.path.join(output_dir, "example_frac_srv_model.i")
    ModelBuilder.build_example_with_configs(example_output_file)

    # tutorial_output_file = os.path.join(output_dir, "tutorial_step1_from_builder_v2.i")
    # ModelBuilder.build_tutorial_step1_example(tutorial_output_file) # Assuming this method is adapted or kept




