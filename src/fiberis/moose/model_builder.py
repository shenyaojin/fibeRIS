# User layer of input generator. Allow user to specify model parameters without
# having to interact with the input file directly.
# Higher-level API for building MOOSE input files, compared to input_generator.py and input_editor.py.
# Shenyao Jin, shenyaojin@mines.edu, 05/24/2025
# fiberis/moose/model_builder.py
from typing import List, Dict, Any, Union, Tuple, Optional

# Import config classes and lower-level MooseBlock class.
from fiberis.moose.config import HydraulicFractureConfig, SRVConfig
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

    # --- API Methods for Mesh Construction ---

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
        half_height = srv_config.height / 2.0

        xmin = srv_config.center_x - half_length
        xmax = srv_config.center_x + half_length
        ymin = srv_config.center_y - half_height
        ymax = srv_config.center_y + half_height

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

    def _get_or_create_kernels_moose_block(self) -> 'MooseBlock':
        """
        Retrieves the main 'Kernels' MooseBlock from self._top_level_blocks,
        or creates and adds it if it doesn't exist.
        """
        # This assumes self._top_level_blocks is a List[MooseBlock]
        for block in self._top_level_blocks:
            if block.block_name == "Kernels":
                return block
        # If not found, create it
        kernels_block = MooseBlock("Kernels")
        self._top_level_blocks.append(kernels_block)  # Add to the list of top-level blocks
        return kernels_block

    def add_time_derivative_kernel(self,
                                   variable: str,
                                   kernel_name: Optional[str] = None) -> 'ModelBuilder':
        """
        Adds a TimeDerivative kernel.

        Args:
            variable (str): The variable this kernel acts upon (e.g., "pp").
            kernel_name (Optional[str], optional): The name for this kernel sub-block.
                                                   Defaults to "dot_{variable}" if None.

        Returns:
            ModelBuilder: Returns self for chaining.
        """
        kernels_main_block = self._get_or_create_kernels_moose_block()
        name_to_use = kernel_name if kernel_name is not None else f"dot_{variable}"

        kernel_obj = MooseBlock(name_to_use, block_type="TimeDerivative")
        kernel_obj.add_param("variable", variable)

        kernels_main_block.add_sub_block(kernel_obj)
        return self

    def add_function_diffusion_kernel(self,
                                      kernel_name: str,
                                      variable: str,
                                      function_name: str,
                                      block_names: Union[str, List[str]]) -> 'ModelBuilder':
        """
        Adds a FunctionDiffusion kernel.

        Args:
            kernel_name (str): The name for this kernel sub-block (e.g., "srv_diffusion").
            variable (str): The variable this kernel acts upon (e.g., "pp").
            function_name (str): The name of the MOOSE Function that defines the diffusion coefficient.
            block_names (Union[str, List[str]]): The block(s) this kernel applies to.
                                                 Can be a single string or a list of strings.

        Returns:
            ModelBuilder: Returns self for chaining.
        """
        kernels_main_block = self._get_or_create_kernels_moose_block()

        kernel_obj = MooseBlock(kernel_name, block_type="FunctionDiffusion")
        kernel_obj.add_param("variable", variable)
        kernel_obj.add_param("function", function_name)
        if isinstance(block_names, list):
            kernel_obj.add_param("block", ' '.join(block_names))
        else:
            kernel_obj.add_param("block", block_names)

        kernels_main_block.add_sub_block(kernel_obj)
        return self

    def add_anisotropic_diffusion_kernel(self,
                                         kernel_name: str,
                                         variable: str,
                                         block_names: Union[str, List[str]],
                                         tensor_coefficient: str) -> 'ModelBuilder':
        """
        Adds an AnisotropicDiffusion kernel.

        Args:
            kernel_name (str): The name for this kernel sub-block (e.g., "matrix_diffusion").
            variable (str): The variable this kernel acts upon (e.g., "pp").
            block_names (Union[str, List[str]]): The block(s) this kernel applies to.
            tensor_coefficient (str): The string representation of the anisotropic diffusion tensor
                                      (e.g., "'Kxx Kxy Kxz  Kxy Kyy Kyz  Kxz Kyz Kzz'").

        Returns:
            ModelBuilder: Returns self for chaining.
        """
        kernels_main_block = self._get_or_create_kernels_moose_block()

        kernel_obj = MooseBlock(kernel_name, block_type="AnisotropicDiffusion")
        kernel_obj.add_param("variable", variable)
        if isinstance(block_names, list):
            kernel_obj.add_param("block", ' '.join(block_names))
        else:
            kernel_obj.add_param("block", block_names)
        kernel_obj.add_param("tensor_coeff", tensor_coefficient)  # Value is already a string

        kernels_main_block.add_sub_block(kernel_obj)
        return self

    def add_porous_flow_darcy_base_kernel(self,
                                          kernel_name: str,
                                          variable: str,
                                          gravity_vector: str = '0 0 0') -> 'ModelBuilder':
        """
        Adds a PorousFlowFullySaturatedDarcyBase kernel (or similar Darcy flux term).

        Args:
            kernel_name (str): The name for this kernel sub-block (e.g., "flux").
            variable (str): The variable this kernel acts upon (e.g., "pp").
            gravity_vector (str, optional): String representation of the gravity vector.
                                            Defaults to '0 0 0'.

        Returns:
            ModelBuilder: Returns self for chaining.
        """
        kernels_main_block = self._get_or_create_kernels_moose_block()

        kernel_obj = MooseBlock(kernel_name, block_type="PorousFlowFullySaturatedDarcyBase")
        kernel_obj.add_param("variable", variable)
        kernel_obj.add_param("gravity", gravity_vector)  # Value is already a string

        kernels_main_block.add_sub_block(kernel_obj)
        return self

    def add_stress_divergence_tensor_kernel(self,
                                            kernel_name: str,
                                            variable: str,
                                            component: int) -> 'ModelBuilder':
        """
        Adds a StressDivergenceTensors kernel.

        Args:
            kernel_name (str): The name for this kernel sub-block (e.g., "grad_stress_x").
            variable (str): The displacement variable this kernel acts upon (e.g., "disp_x").
            component (int): The component (0 for x, 1 for y, 2 for z).

        Returns:
            ModelBuilder: Returns self for chaining.
        """
        kernels_main_block = self._get_or_create_kernels_moose_block()

        kernel_obj = MooseBlock(kernel_name, block_type="StressDivergenceTensors")
        kernel_obj.add_param("variable", variable)
        kernel_obj.add_param("component", component)

        kernels_main_block.add_sub_block(kernel_obj)
        return self

    def add_porous_flow_effective_stress_coupling_kernel(self,
                                                         kernel_name: str,
                                                         variable: str,
                                                         # This is typically a displacement variable component
                                                         component: int,
                                                         biot_coefficient: float) -> 'ModelBuilder':
        """
        Adds a PorousFlowEffectiveStressCoupling kernel.
        This kernel adds a term related to the divergence of displacement (strain) to the flow equation.

        Args:
            kernel_name (str): The name for this kernel sub-block (e.g., "poro_x").
            variable (str): The displacement variable component this term is derived from (e.g., "disp_x").
            component (int): The component of displacement gradient (0 for d/dx, 1 for d/dy).
            biot_coefficient (float): The Biot coefficient.

        Returns:
            ModelBuilder: Returns self for chaining.
        """
        kernels_main_block = self._get_or_create_kernels_moose_block()

        kernel_obj = MooseBlock(kernel_name, block_type="PorousFlowEffectiveStressCoupling")
        kernel_obj.add_param("variable", variable)
        kernel_obj.add_param("component", component)
        kernel_obj.add_param("biot_coefficient", biot_coefficient)

        kernels_main_block.add_sub_block(kernel_obj)
        return self

    def add_porous_flow_mass_volumetric_expansion_kernel(self,
                                                         kernel_name: str,
                                                         variable: str,  # This is typically the pore pressure variable
                                                         fluid_component: int = 0) -> 'ModelBuilder':
        """
        Adds a PorousFlowMassVolumetricExpansion kernel.
        This kernel accounts for fluid compressibility and other volumetric expansion effects.

        Args:
            kernel_name (str): The name for this kernel sub-block (e.g., "vol_strain_rate_water").
            variable (str): The pore pressure variable this term is associated with (e.g., "pp").
            fluid_component (int, optional): The fluid component index if using multi-component flow.
                                             Defaults to 0.

        Returns:
            ModelBuilder: Returns self for chaining.
        """
        kernels_main_block = self._get_or_create_kernels_moose_block()

        kernel_obj = MooseBlock(kernel_name, block_type="PorousFlowMassVolumetricExpansion")
        kernel_obj.add_param("variable", variable)
        kernel_obj.add_param("fluid_component", fluid_component)

        kernels_main_block.add_sub_block(kernel_obj)
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

    # --- This would be the updated static method and __main__ in ModelBuilder ---
    @staticmethod
    def build_example_with_configs(output_filepath: str = "example_with_configs.i"):
        builder = ModelBuilder(project_name="FracSRVExampleWithKernels")

        # 1. Define main domain
        builder.set_main_domain_parameters_2d(
            domain_name="matrix",
            length=1000, height=500,
            num_elements_x=50, num_elements_y=25,
            moose_block_id=0
        )

        # 2. Define SRV
        srv1_conf = SRVConfig(
            name="SRV1",
            length=300, height=80,
            center_x=500, center_y=250,
            permeability=1e-13  # Example permeability for SRV
        )
        builder.add_srv_zone_2d(srv_config=srv1_conf, target_moose_block_id=1, refinement_passes=1)

        # Define another SRV for demonstration if needed, e.g., SRV2 for block 'srv srv2 srv3'
        srv2_conf = SRVConfig(name="SRV2", length=50, height=50, center_x=400, center_y=200, permeability=1e-13)
        builder.add_srv_zone_2d(srv_config=srv2_conf, target_moose_block_id=3)  # Assuming ID 3 for SRV2

        srv3_conf = SRVConfig(name="SRV3", length=50, height=50, center_x=600, center_y=300, permeability=1e-13)
        builder.add_srv_zone_2d(srv_config=srv3_conf, target_moose_block_id=4)  # Assuming ID 4 for SRV3

        # 3. Define Fracture
        frac1_conf = HydraulicFractureConfig(
            name="Frac1",
            length=200, height=0.2,
            center_x=500, center_y=250,
            orientation_angle=0,
            permeability=1e-10  # Example permeability for Fracture
        )
        builder.add_hydraulic_fracture_2d(fracture_config=frac1_conf, target_moose_block_id=2, refinement_passes=2)

        # 4. Add an injection point (Nodeset)
        builder.add_nodeset_by_coord(
            nodeset_op_name="injection_well_nodes",
            new_boundary_name="injection_well",
            coordinates=(500, 250)
        )

        # 5. Add GlobalParams (Example)
        # Assuming a method add_global_params exists or will be added to ModelBuilder
        # For now, let's imagine it creates a MooseBlock("GlobalParams") and adds params
        # For this example, we'll skip direct GlobalParams addition to focus on Kernels

        # 6. Add Variables
        # Assuming a method add_variables_block similar to previous discussions
        # builder.add_variables_block([
        #     {"name": "pp", "params": {"initial_condition": 26.4E6}},
        #     {"name": "disp_x", "params": {"scaling": 1E-10}},
        #     {"name": "disp_y", "params": {"scaling": 1E-10}}
        # ])
        # For now, let's assume these variables are expected by the kernels

        # --- ADDING KERNELS, EXAMPLE ---
        builder.add_time_derivative_kernel(variable="pp", kernel_name="dot")

        # Assuming functions 'srv_diff' and 'frac_diff' will be defined in [Functions] block later
        builder.add_function_diffusion_kernel(kernel_name="srv_diffusion",
                                              variable="pp",
                                              function_name="srv_diff",
                                              block_names=["SRV1", "SRV2", "SRV3"])  # Using final block names

        builder.add_function_diffusion_kernel(kernel_name="fracture_diffusion",
                                              variable="pp",
                                              function_name="frac_diff",
                                              block_names="Frac1")  # Using final block name

        builder.add_anisotropic_diffusion_kernel(kernel_name="matrix_diffusion",
                                                 variable="pp",
                                                 block_names="matrix",  # Using final block name
                                                 tensor_coefficient="'0.00005894 0 0  0 0.00005894 0  0 0 0.00005894'")

        builder.add_porous_flow_darcy_base_kernel(kernel_name="flux",
                                                  variable="pp",
                                                  gravity_vector="'0 0 0'")  # MOOSE expects strings for vectors

        builder.add_stress_divergence_tensor_kernel(kernel_name="grad_stress_x",
                                                    variable="disp_x",
                                                    component=0)
        builder.add_stress_divergence_tensor_kernel(kernel_name="grad_stress_y",
                                                    variable="disp_y",
                                                    component=1)

        builder.add_porous_flow_effective_stress_coupling_kernel(kernel_name="poro_x",
                                                                 variable="disp_x",
                                                                 component=0,
                                                                 biot_coefficient=0.7)
        builder.add_porous_flow_effective_stress_coupling_kernel(kernel_name="poro_y",
                                                                 variable="disp_y",
                                                                 component=1,
                                                                 biot_coefficient=0.7)

        builder.add_porous_flow_mass_volumetric_expansion_kernel(kernel_name="vol_strain_rate_water",
                                                                 variable="pp",
                                                                 fluid_component=0)

        # Note: if needed, you can add more kernels here following the same pattern. --Shenyao

        # --- END OF KERNELS ---


        # (Skipping Materials, BCs, Executioner, Outputs for this focused update)

        # Generate the file
        builder.generate_input_file(output_filepath)
        print(f"Example file with Kernels generated: {output_filepath}")

    # if __name__ == '__main__': (within ModelBuilder class, this static method would be called from outside or by other static methods)
    # This part should be outside the class definition if it's the main execution script for testing

if __name__ == '__main__':  # This should be at the bottom of your model_builder.py file
    import os
    # Ensure ModelBuilder and MooseBlock are correctly importable
    # from .input_generator import MooseBlock # If ModelBuilder itself is not in the same file

    output_dir = "test_files/moose_input_file_test"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    example_kernels_output_file = os.path.join(output_dir, "example_model_with_kernels.i")
    ModelBuilder.build_example_with_configs(example_kernels_output_file)