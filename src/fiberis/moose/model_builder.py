# User layer of input generator. Allow user to specify model parameters without
# having to interact with the input file directly.
# Higher-level API for building MOOSE input files, compared to input_generator.py and input_editor.py.
# Shenyao Jin, shenyaojin@mines.edu, 05/24/2025
# fiberis/moose/model_builder.py
from typing import List, Dict, Any, Union, Tuple, Optional

# Import config classes
from fiberis.moose.config import HydraulicFractureConfig, SRVConfig

# Import the core generation function from your input_generator.py
from fiberis.moose.input_generator import MooseBlock


class ModelBuilder:
    """
    This class provides a high-level API to construct MOOSE input files.
    This version focuses on building the [Mesh] block directly using MooseBlock objects.
    It uses configuration objects (HydraulicFractureConfig, SRVConfig)
    to define physical features and translates them into MOOSE mesh operations.
    """

    def __init__(self, project_name: str):
        """
        Initializes the ModelBuilder.

        Args:
            project_name (str): The name of the project.
        """
        self.project_name: str = project_name
        self._top_level_blocks: List[MooseBlock] = []  # Stores top-level MooseBlock objects (e.g., for Mesh, Variables)

        # Mesh specific tracking
        self._block_id_to_name_map: Dict[int, str] = {}  # Numeric ID to final name for blocks
        self._next_available_block_id: int = 1  # Start assigning from 1 (0 is often default matrix)
        self._main_domain_block_id: int = 0
        self._last_mesh_op_name_within_mesh_block: Optional[
            str] = None  # Tracks the name of the last sub-operation *within* the [Mesh] block

    def _generate_unique_op_name(self, base_name: str, existing_names_list: List[str]) -> str:
        """Generates a unique name for an operation within a list of existing names."""
        count = 1
        op_name = base_name
        while op_name in existing_names_list:
            op_name = f"{base_name}_{count}"
            count += 1
        return op_name

    def _get_or_create_mesh_moose_block(self) -> MooseBlock:
        """
        Retrieves the main 'Mesh' MooseBlock from self._top_level_blocks,
        or creates and adds it if it doesn't exist.
        """
        for block in self._top_level_blocks:
            if block.block_name == "Mesh":
                return block
        # If not found, create it
        mesh_block = MooseBlock("Mesh")
        self._top_level_blocks.append(mesh_block)
        return mesh_block

    # --- High-Level API Methods for Mesh Construction ---

    def set_main_domain_parameters_2d(self,
                                      domain_name: str,
                                      length: float,  # X-dimension
                                      height: float,  # Y-dimension
                                      num_elements_x: int,
                                      num_elements_y: int,
                                      xmin: float = 0.0,
                                      ymin: float = 0.0,
                                      moose_block_id: int = 0,
                                      **additional_generator_params) -> 'ModelBuilder':
        """
        Defines the main 2D reservoir domain (matrix) using a GeneratedMeshGenerator.

        Args:
            domain_name (str): Name for the main reservoir domain (e.g., "matrix").
            length (float): Total length of the matrix domain (X-dimension).
            height (float): Total height of the matrix domain (Y-dimension).
            num_elements_x (int): Number of mesh elements (nx) along the length.
            num_elements_y (int): Number of mesh elements (ny) along the height.
            xmin (float, optional): Minimum x-coordinate. Defaults to 0.0.
            ymin (float, optional): Minimum y-coordinate. Defaults to 0.0.
            moose_block_id (int, optional): The integer ID for this main domain. Defaults to 0.
            **additional_generator_params: Other parameters for GeneratedMeshGenerator.

        Returns:
            ModelBuilder: Returns self for chaining.
        """
        mesh_moose_block = self._get_or_create_mesh_moose_block()
        existing_sub_block_names = [sb.block_name for sb in mesh_moose_block.sub_blocks]

        self._main_domain_block_id = moose_block_id
        self._block_id_to_name_map[self._main_domain_block_id] = domain_name

        # Name for the GeneratedMeshGenerator operation itself
        gmg_op_name = self._generate_unique_op_name(f"{domain_name}_base_mesh", existing_sub_block_names)

        gmg_params = {
            "dim": 2,
            "nx": num_elements_x,
            "ny": num_elements_y,
            "xmin": xmin,
            "xmax": xmin + length,
            "ymin": ymin,
            "ymax": ymin + height,
            **additional_generator_params
        }

        gmg_sub_block = MooseBlock(gmg_op_name, block_type="GeneratedMeshGenerator")
        for p_name, p_val in gmg_params.items():
            gmg_sub_block.add_param(p_name, p_val)

        mesh_moose_block.add_sub_block(gmg_sub_block)
        self._last_mesh_op_name_within_mesh_block = gmg_op_name
        return self

    def add_hydraulic_fracture_2d(self,
                                  fracture_config: HydraulicFractureConfig,
                                  target_moose_block_id: Optional[int] = None,
                                  refinement_passes: Optional[int] = None) -> 'ModelBuilder':
        """
        Adds a hydraulic fracture to the 2D model based on HydraulicFractureConfig.
        Currently assumes axis-aligned fracture.

        Args:
            fracture_config (HydraulicFractureConfig): Configuration object for the fracture.
            target_moose_block_id (Optional[int], optional): Numeric block ID. Auto-assigned if None.
            refinement_passes (Optional[int], optional): Number of refinement passes. Defaults to None.

        Returns:
            ModelBuilder: Returns self for chaining.
        """
        mesh_moose_block = self._get_or_create_mesh_moose_block()
        existing_sub_block_names = [sb.block_name for sb in mesh_moose_block.sub_blocks]

        if fracture_config.orientation_angle != 0.0:
            print(f"Warning: HydraulicFractureConfig '{fracture_config.name}' has orientation_angle = "
                  f"{fracture_config.orientation_angle}°. This API currently only supports axis-aligned "
                  "fractures. The fracture will be generated as axis-aligned.")

        block_id_to_use = target_moose_block_id
        if block_id_to_use is None:
            block_id_to_use = self._next_available_block_id
            self._next_available_block_id += 1
        elif block_id_to_use == self._main_domain_block_id or block_id_to_use in self._block_id_to_name_map:
            new_id = self._next_available_block_id
            print(f"Warning: target_moose_block_id {block_id_to_use} for fracture '{fracture_config.name}' "
                  f"is already in use or is the main domain ID. Assigning a new ID: {new_id}")
            block_id_to_use = new_id
            self._next_available_block_id += 1
        self._block_id_to_name_map[block_id_to_use] = fracture_config.name

        half_length = fracture_config.length / 2.0
        half_height = fracture_config.height / 2.0  # Aperture

        xmin = fracture_config.center_x - half_length
        xmax = fracture_config.center_x + half_length
        ymin = fracture_config.center_y - half_height
        ymax = fracture_config.center_y + half_height

        z_epsilon_bottom = "0.00000001"
        z_epsilon_top = "0"

        bbox_op_name = self._generate_unique_op_name(f"{fracture_config.name}_bbox", existing_sub_block_names)
        bbox_params = {
            "bottom_left": f"{xmin} {ymin} {z_epsilon_bottom}",
            "top_right": f"{xmax} {ymax} {z_epsilon_top}",
            "block_id": block_id_to_use,
            "input": self._last_mesh_op_name_within_mesh_block  # Link to previous mesh op
        }
        bbox_sub_block = MooseBlock(bbox_op_name, block_type="SubdomainBoundingBoxGenerator")
        for p_name, p_val in bbox_params.items():
            bbox_sub_block.add_param(p_name, p_val)
        mesh_moose_block.add_sub_block(bbox_sub_block)
        current_op_chain_head = bbox_op_name

        if refinement_passes is not None and refinement_passes > 0:
            existing_sub_block_names.append(current_op_chain_head)  # Update list for unique name gen
            refine_op_name = self._generate_unique_op_name(f"{fracture_config.name}_refine", existing_sub_block_names)
            refinement_str = f"{refinement_passes} {refinement_passes}"
            refine_params = {
                "block": str(block_id_to_use),
                "refinement": refinement_str,
                "input": current_op_chain_head  # Link to the BBox operation
            }
            refine_sub_block = MooseBlock(refine_op_name, block_type="RefineBlockGenerator")
            for p_name, p_val in refine_params.items():
                refine_sub_block.add_param(p_name, p_val)
            mesh_moose_block.add_sub_block(refine_sub_block)
            current_op_chain_head = refine_op_name

        self._last_mesh_op_name_within_mesh_block = current_op_chain_head
        return self

    def add_srv_zone_2d(self,
                        srv_config: SRVConfig,
                        target_moose_block_id: Optional[int] = None,
                        refinement_passes: Optional[int] = None) -> 'ModelBuilder':
        """
        Adds an SRV (Stimulated Reservoir Volume) zone to the 2D model.

        Args:
            srv_config (SRVConfig): Configuration object for the SRV zone.
            target_moose_block_id (Optional[int], optional): Numeric block ID. Auto-assigned if None.
            refinement_passes (Optional[int], optional): Number of refinement passes. Defaults to None.

        Returns:
            ModelBuilder: Returns self for chaining.
        """
        mesh_moose_block = self._get_or_create_mesh_moose_block()
        existing_sub_block_names = [sb.block_name for sb in mesh_moose_block.sub_blocks]

        block_id_to_use = target_moose_block_id
        if block_id_to_use is None:
            block_id_to_use = self._next_available_block_id
            self._next_available_block_id += 1
        elif block_id_to_use == self._main_domain_block_id or block_id_to_use in self._block_id_to_name_map:
            new_id = self._next_available_block_id
            print(f"Warning: target_moose_block_id {block_id_to_use} for SRV '{srv_config.name}' "
                  f"is already in use or is the main domain ID. Assigning a new ID: {new_id}")
            block_id_to_use = new_id
            self._next_available_block_id += 1
        self._block_id_to_name_map[block_id_to_use] = srv_config.name

        half_length = srv_config.length / 2.0
        half_width = srv_config.width / 2.0

        xmin = srv_config.center_x - half_length
        xmax = srv_config.center_x + half_length
        ymin = srv_config.center_y - half_width
        ymax = srv_config.center_y + half_width

        z_epsilon_bottom = "0.00000001"
        z_epsilon_top = "0"

        bbox_op_name = self._generate_unique_op_name(f"{srv_config.name}_bbox", existing_sub_block_names)
        bbox_params = {
            "bottom_left": f"{xmin} {ymin} {z_epsilon_bottom}",
            "top_right": f"{xmax} {ymax} {z_epsilon_top}",
            "block_id": block_id_to_use,
            "input": self._last_mesh_op_name_within_mesh_block
        }
        bbox_sub_block = MooseBlock(bbox_op_name, block_type="SubdomainBoundingBoxGenerator")
        for p_name, p_val in bbox_params.items():
            bbox_sub_block.add_param(p_name, p_val)
        mesh_moose_block.add_sub_block(bbox_sub_block)
        current_op_chain_head = bbox_op_name

        if refinement_passes is not None and refinement_passes > 0:
            existing_sub_block_names.append(current_op_chain_head)
            refine_op_name = self._generate_unique_op_name(f"{srv_config.name}_refine", existing_sub_block_names)
            refinement_str = f"{refinement_passes} {refinement_passes}"
            refine_params = {
                "block": str(block_id_to_use),
                "refinement": refinement_str,
                "input": current_op_chain_head
            }
            refine_sub_block = MooseBlock(refine_op_name, block_type="RefineBlockGenerator")
            for p_name, p_val in refine_params.items():
                refine_sub_block.add_param(p_name, p_val)
            mesh_moose_block.add_sub_block(refine_sub_block)
            current_op_chain_head = refine_op_name

        self._last_mesh_op_name_within_mesh_block = current_op_chain_head
        return self

    def add_nodeset_by_coord(self,
                             nodeset_op_name: str,
                             new_boundary_name: str,
                             coordinates: Union[Tuple[float, ...], str],
                             **additional_params) -> 'ModelBuilder':
        """
        Adds an ExtraNodesetGenerator block by coordinates.
        The 'input' for this operation will be the last known mesh operation.
        """
        mesh_moose_block = self._get_or_create_mesh_moose_block()
        existing_sub_block_names = [sb.block_name for sb in mesh_moose_block.sub_blocks]

        op_name = self._generate_unique_op_name(nodeset_op_name, existing_sub_block_names)

        params = {
            "new_boundary": new_boundary_name,
            "coord": ' '.join(map(str, coordinates)) if isinstance(coordinates, tuple) else coordinates,
            "input": self._last_mesh_op_name_within_mesh_block,
            **additional_params
        }
        nodeset_sub_block = MooseBlock(op_name, block_type="ExtraNodesetGenerator")
        for p_name, p_val in params.items():
            nodeset_sub_block.add_param(p_name, p_val)

        mesh_moose_block.add_sub_block(nodeset_sub_block)
        self._last_mesh_op_name_within_mesh_block = op_name
        return self

    def _finalize_mesh_block_renaming(self):
        """
        Adds a RenameBlockGenerator at the end of the mesh operations
        if there are mappings in _block_id_to_name_map.
        This method should be called before rendering the Mesh block.
        """
        mesh_moose_block = self._get_or_create_mesh_moose_block()  # Ensures mesh_moose_block exists

        if self._block_id_to_name_map and self._last_mesh_op_name_within_mesh_block:
            old_block_ids = []
            new_block_names_str = []

            sorted_block_ids = sorted(self._block_id_to_name_map.keys())

            for block_id in sorted_block_ids:
                old_block_ids.append(str(block_id))
                new_block_names_str.append(self._block_id_to_name_map[block_id])

            if old_block_ids:
                existing_sub_block_names = [sb.block_name for sb in mesh_moose_block.sub_blocks]
                rename_op_name = self._generate_unique_op_name("final_block_rename", existing_sub_block_names)

                rename_params = {
                    "old_block": ' '.join(old_block_ids),
                    "new_block": ' '.join(new_block_names_str),
                    "input": self._last_mesh_op_name_within_mesh_block  # Link to the very last mesh state
                }
                rename_sub_block = MooseBlock(rename_op_name, block_type="RenameBlockGenerator")
                for p_name, p_val in rename_params.items():
                    rename_sub_block.add_param(p_name, p_val)

                mesh_moose_block.add_sub_block(rename_sub_block)
                self._last_mesh_op_name_within_mesh_block = rename_op_name

    # --- File Generation ---
    def generate_input_file(self, output_filepath: str):
        """
        Generates the MOOSE input file by rendering all top-level MooseBlock objects.
        Currently, this will primarily render the [Mesh] block.
        """
        self._finalize_mesh_block_renaming()  # Ensure renaming is the last mesh operation

        if not self._top_level_blocks:
            # If only mesh operations were called, _top_level_blocks might still contain the Mesh block.
            # Check specifically if the Mesh block has content.
            mesh_block_found = any(
                block.block_name == "Mesh" and (block.params or block.sub_blocks) for block in self._top_level_blocks)
            if not mesh_block_found:
                raise ValueError("No mesh operations defined. Cannot generate an empty input file.")

        all_rendered_blocks = []
        for block in self._top_level_blocks:
            # Only render if the block actually has parameters or sub-blocks,
            # or if it's a block type that can be empty (like a simple [Mesh] with no file).
            # For now, we assume if it's in _top_level_blocks, it's meant to be rendered.
            # The MooseBlock.render() handles its internal structure.
            # The extra '[]' is for closing the top-level block itself.
            all_rendered_blocks.append(block.render(indent_level=0) + "\n[]")

        with open(output_filepath, 'w') as f:
            f.write("# MOOSE input file generated by ModelBuilder (Mesh Part Only)\n\n")
            f.write("\n\n".join(all_rendered_blocks))

        print(f"MOOSE input file (Mesh part) generated by ModelBuilder: {output_filepath}")

    # --- Example Usage Method (Mesh Only) ---
    @staticmethod
    def build_mesh_example(output_filepath: str = "example_mesh_only.i"):
        builder = ModelBuilder(project_name="MeshOnlyExample")

        # 1. Define main domain
        builder.set_main_domain_parameters_2d(
            domain_name="matrix",  # This will be the final name for block 0
            length=1000, height=500,
            num_elements_x=50, num_elements_y=25,  # Coarser for quicker example
            moose_block_id=0
        )

        # 2. Define SRV
        srv1_conf = SRVConfig(
            name="MySRV",
            length=300, width=80,
            center_x=500, center_y=250
        )
        builder.add_srv_zone_2d(srv_config=srv1_conf, target_moose_block_id=1, refinement_passes=1)

        # 3. Define Fracture
        frac1_conf = HydraulicFractureConfig(
            name="MyFrac",
            length=200, height=0.2,
            center_x=500, center_y=250,
            orientation_angle=0  # Axis-aligned
        )
        builder.add_hydraulic_fracture_2d(fracture_config=frac1_conf, target_moose_block_id=2, refinement_passes=2)

        # 4. Add an injection point (Nodeset)
        builder.add_nodeset_by_coord(
            nodeset_op_name="injection_pt_nodes",
            new_boundary_name="injection_well_nodeset",
            coordinates=(500, 250)
        )

        # Generate the file
        builder.generate_input_file(output_filepath)
        print(f"Example mesh-only file generated: {output_filepath}")


if __name__ == '__main__':
    import os

    # Ensure the directory for the output file exists
    output_dir = "test_files/moose_input_file_test"  # Using the same test directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    example_output_file = os.path.join(output_dir, "model_builder_mesh_output.i")
    ModelBuilder.build_mesh_example(example_output_file)

    # You can inspect "model_builder_mesh_output.i" to see the generated [Mesh] block