# User layer of input generator. Allow user to specify model parameters without
# having to interact with the input file directly.
# Higher-level API for building MOOSE input files, compared to input_generator.py and input_editor.py.
# Shenyao Jin, shenyaojin@mines.edu, 05/24/2025
# fiberis/moose/model_builder.py
from typing import List, Dict, Any, Union, Tuple, Optional

# Import config classes and lower-level MooseBlock class.
from fiberis.moose.config import HydraulicFractureConfig, SRVConfig, AdaptivityConfig, \
    PointValueSamplerConfig, LineValueSamplerConfig, PostprocessorConfigBase
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

    # --- API Methods for Kernels ---
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

    def _get_or_create_adaptivity_moose_block(self, adaptivity_block_name: str = "Adaptivity") -> 'MooseBlock':
        """
        Retrieves the main 'Adaptivity' MooseBlock from self._top_level_blocks,
        or creates and adds it if it doesn't exist.
        Internal helper method.
        """
        for block in self._top_level_blocks:
            if block.block_name == adaptivity_block_name:
                # Clear existing params and sub_blocks if we are re-configuring it
                block.params.clear()
                block.sub_blocks.clear()
                return block
        # If not found, create it
        adapt_block = MooseBlock(adaptivity_block_name)
        self._top_level_blocks.append(adapt_block)
        return adapt_block

    def set_adaptivity_options(self,
                               enable: bool = True,
                               config: Optional['AdaptivityConfig'] = None,
                               # Use forward reference if AdaptivityConfig is in the same file and defined later, or import
                               default_template_settings: Optional[Dict[str, Any]] = None,
                               adaptivity_block_name: str = "Adaptivity") -> 'ModelBuilder':
        """
        Sets the options for the [Adaptivity] block.

        This method allows enabling/disabling adaptivity, using a detailed AdaptivityConfig object,
        or applying a default template based on simple settings.

        Args:
            enable (bool, optional): If True, enables and configures adaptivity.
                                     If False, removes the adaptivity block. Defaults to True.
            config (Optional[AdaptivityConfig], optional): A pre-configured AdaptivityConfig object
                                                           for detailed AMA setup. Defaults to None.
            default_template_settings (Optional[Dict[str, Any]], optional):
                If 'enable' is True and 'config' is None, these settings are used
                to apply a default AMA template. Expected keys might include:
                - 'monitored_variable' (str): Variable to base default indicator on (e.g., 'pp').
                - 'refine_fraction' (float): Fraction of elements to refine (e.g., 0.3).
                - 'coarsen_fraction' (float): Fraction of elements to coarsen (e.g., 0.05).
                - 'steps' (int): Number of adaptivity steps (e.g., 2).
                Defaults to None.
            adaptivity_block_name (str, optional): The name for the [Adaptivity] top-level block.
                                                   Defaults to "Adaptivity".

        Returns:
            ModelBuilder: Returns self for chaining.
        """
        if not enable:
            # Remove the adaptivity block if it exists
            self._top_level_blocks = [
                block for block in self._top_level_blocks if block.block_name != adaptivity_block_name
            ]
            print(f"Info: Adaptivity block '{adaptivity_block_name}' removed (disabled).")
            return self

        adapt_moose_block = self._get_or_create_adaptivity_moose_block(adaptivity_block_name)

        if config is not None:
            # --- Configure using a detailed AdaptivityConfig object ---
            if not isinstance(config, AdaptivityConfig):  # Make sure AdaptivityConfig is imported or defined
                raise TypeError("The 'config' argument must be an instance of AdaptivityConfig.")

            adapt_moose_block.add_param("marker", config.marker_to_use)
            adapt_moose_block.add_param("steps", config.steps)

            if config.indicators:
                indicators_main_sub = MooseBlock("Indicators")
                for ind_conf in config.indicators:
                    indicator_obj = MooseBlock(ind_conf.name, block_type=ind_conf.type)
                    for p_name, p_val in ind_conf.params.items():
                        indicator_obj.add_param(p_name, p_val)
                    indicators_main_sub.add_sub_block(indicator_obj)
                adapt_moose_block.add_sub_block(indicators_main_sub)

            if config.markers:
                markers_main_sub = MooseBlock("Markers")
                for marker_conf in config.markers:
                    marker_obj = MooseBlock(marker_conf.name, block_type=marker_conf.type)
                    for p_name, p_val in marker_conf.params.items():
                        marker_obj.add_param(p_name, p_val)
                    markers_main_sub.add_sub_block(marker_obj)
                adapt_moose_block.add_sub_block(markers_main_sub)

            print(f"Info: Adaptivity block '{adaptivity_block_name}' configured using provided AdaptivityConfig.")

        elif default_template_settings is not None:
            # --- Configure using a default template ---
            print(f"Info: Configuring adaptivity block '{adaptivity_block_name}' using default template settings.")

            mon_var = default_template_settings.get("monitored_variable", "pp")  # Default to 'pp'
            ref_frac = default_template_settings.get("refine_fraction", 0.3)
            coarse_frac = default_template_settings.get("coarsen_fraction", 0.05)
            adapt_steps = default_template_settings.get("steps", 2)

            default_indicator_name = f"indicator_on_{mon_var}"
            default_marker_name = f"marker_for_{mon_var}"

            adapt_moose_block.add_param("marker", default_marker_name)
            adapt_moose_block.add_param("steps", adapt_steps)

            # Default Indicator: GradientJumpIndicator on the monitored variable
            indicators_main_sub = MooseBlock("Indicators")
            default_indicator = MooseBlock(default_indicator_name, block_type="GradientJumpIndicator")
            default_indicator.add_param("variable", mon_var)
            default_indicator.add_param("outputs", "none")  # Common default
            indicators_main_sub.add_sub_block(default_indicator)
            adapt_moose_block.add_sub_block(indicators_main_sub)

            # Default Marker: ErrorFractionMarker using the default indicator
            markers_main_sub = MooseBlock("Markers")
            default_marker = MooseBlock(default_marker_name, block_type="ErrorFractionMarker")
            default_marker.add_param("indicator", default_indicator_name)
            default_marker.add_param("refine", ref_frac)
            default_marker.add_param("coarsen", coarse_frac)
            default_marker.add_param("outputs", "none")  # Common default
            markers_main_sub.add_sub_block(default_marker)
            adapt_moose_block.add_sub_block(markers_main_sub)

            print(f"Info: Applied default AMA template for variable '{mon_var}'.")

        else:
            # enable=True, but no config and no default_template_settings provided
            # Option: apply a very basic built-in template or raise an error/warning
            print(f"Warning: Adaptivity enabled for '{adaptivity_block_name}', but no specific 'config' or "
                  "'default_template_settings' provided. A minimal/default AMA setup might be applied if available, "
                  "or it might be an empty [Adaptivity] block which could be an error in MOOSE.")
            # For a truly minimal setup, you might just set steps, but MOOSE needs a marker.
            # Let's apply a very basic template if nothing else is given.
            adapt_moose_block.add_param("marker", "default_marker")  # Requires a marker named 'default_marker'
            adapt_moose_block.add_param("steps", 1)

            indicators_main_sub = MooseBlock("Indicators")
            default_indicator = MooseBlock("default_indicator", block_type="GradientJumpIndicator")
            default_indicator.add_param("variable", "pp")  # Fallback variable
            indicators_main_sub.add_sub_block(default_indicator)
            adapt_moose_block.add_sub_block(indicators_main_sub)

            markers_main_sub = MooseBlock("Markers")
            default_marker = MooseBlock("default_marker", block_type="ErrorFractionMarker")
            default_marker.add_param("indicator", "default_indicator")
            default_marker.add_param("refine", 0.5)
            markers_main_sub.add_sub_block(default_marker)
            adapt_moose_block.add_sub_block(markers_main_sub)
            print("Info: Applied a very basic fallback AMA template for variable 'pp'.")

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

    def _get_or_create_bcs_moose_block(self) -> 'MooseBlock':
        """
        Retrieves the main 'BCs' MooseBlock from self._top_level_blocks,
        or creates and adds it if it doesn't exist.
        Internal helper method.
        """
        for block in self._top_level_blocks:
            if block.block_name == "BCs":
                # For BCs, we might want to append rather than clear,
                # unless a method is designed to set ALL BCs at once.
                # For now, let's assume we can add multiple BCs.
                return block
        # If not found, create it
        bcs_block = MooseBlock("BCs")
        self._top_level_blocks.append(bcs_block)
        return bcs_block

    # Custom method to generate BCs

    def add_boundary_condition(self,
                               name: str,
                               bc_type: str,
                               variable: str,
                               boundary_name: Union[str, List[str]],
                               params: Optional[Dict[str, Any]] = None) -> 'ModelBuilder':
        """
        Adds a single, generic boundary condition to the [BCs] block.

        Args:
            name (str): The user-chosen name for this boundary condition sub-block (e.g., "pressure_left_wall").
            bc_type (str): The MOOSE type for the boundary condition (e.g., "DirichletBC", "FunctionNeumannBC").
            variable (str): The variable this boundary condition applies to (e.g., "pp", "disp_x").
            boundary_name (Union[str, List[str]]): The name(s) of the mesh boundary (sideset or nodeset)
                                                   this BC applies to. If a list, names will be space-separated.
            params (Optional[Dict[str, Any]], optional): A dictionary of additional parameters specific
                                                         to this bc_type (e.g., {"value": 0.0},
                                                         {"function": "my_func_name"}). Defaults to None.

        Returns:
            ModelBuilder: Returns self for chaining.
        """
        bcs_main_block = self._get_or_create_bcs_moose_block()

        bc_sub_block = MooseBlock(name, block_type=bc_type)
        bc_sub_block.add_param("variable", variable)

        if isinstance(boundary_name, list):
            bc_sub_block.add_param("boundary", ' '.join(boundary_name))
        else:
            bc_sub_block.add_param("boundary", boundary_name)

        if params:
            for p_name, p_val in params.items():
                bc_sub_block.add_param(p_name, p_val)

        bcs_main_block.add_sub_block(bc_sub_block)
        print(f"Info: Added Boundary Condition '{name}'.")
        return self

    # pre defined BC generators

    def add_boundary_condition(self,
                               name: str,
                               bc_type: str,
                               variable: str,
                               boundary_name: Union[str, List[str]],
                               params: Optional[Dict[str, Any]] = None) -> 'ModelBuilder':
        """
        Adds a single, generic boundary condition to the [BCs] block.

        Args:
            name (str): The user-chosen name for this boundary condition sub-block (e.g., "pressure_left_wall").
            bc_type (str): The MOOSE type for the boundary condition (e.g., "DirichletBC", "FunctionNeumannBC").
            variable (str): The variable this boundary condition applies to (e.g., "pp", "disp_x").
            boundary_name (Union[str, List[str]]): The name(s) of the mesh boundary (sideset or nodeset)
                                                   this BC applies to. If a list, names will be space-separated.
            params (Optional[Dict[str, Any]], optional): A dictionary of additional parameters specific
                                                         to this bc_type (e.g., {"value": 0.0},
                                                         {"function": "my_func_name"}). Defaults to None.

        Returns:
            ModelBuilder: Returns self for chaining.
        """
        bcs_main_block = self._get_or_create_bcs_moose_block()

        bc_sub_block = MooseBlock(name, block_type=bc_type)
        bc_sub_block.add_param("variable", variable)

        if isinstance(boundary_name, list):
            bc_sub_block.add_param("boundary", ' '.join(boundary_name))
        else:
            bc_sub_block.add_param("boundary", boundary_name)

        if params:
            for p_name, p_val in params.items():
                bc_sub_block.add_param(p_name, p_val)

        bcs_main_block.add_sub_block(bc_sub_block)
        print(f"Info: Added Boundary Condition '{name}'.")
        return self

    def set_hydraulic_fracturing_bcs(self,
                                     injection_well_boundary_name: str,
                                     injection_pressure_function_name: str,
                                     confine_disp_x_boundaries: Union[str, List[str]],
                                     confine_disp_y_boundaries: Union[str, List[str]],
                                     pressure_variable: str = "pp",
                                     disp_x_variable: str = "disp_x",
                                     disp_y_variable: str = "disp_y") -> 'ModelBuilder':
        """
        Sets up a predefined set of typical boundary conditions for a hydraulic fracturing simulation.
        This method clears any existing BCs and adds a specific common set.
        Consider using add_boundary_condition for more granular control.

        Args:
            injection_well_boundary_name (str): Name of the nodeset for injection well.
            injection_pressure_function_name (str): Name of the MOOSE Function defining injection pressure.
            confine_disp_x_boundaries (Union[str, List[str]]): Boundary name(s) for confining x-displacement.
            confine_disp_y_boundaries (Union[str, List[str]]): Boundary name(s) for confining y-displacement.
            pressure_variable (str, optional): Name of the pore pressure variable. Defaults to "pp".
            disp_x_variable (str, optional): Name of the x-displacement variable. Defaults to "disp_x".
            disp_y_variable (str, optional): Name of the y-displacement variable. Defaults to "disp_y".

        Returns:
            ModelBuilder: Returns self for chaining.
        """
        bcs_main_block = self._get_or_create_bcs_moose_block()
        bcs_main_block.sub_blocks.clear()  # Start fresh for this specific set of BCs

        # 1. Injection Pressure BC
        self.add_boundary_condition(
            name="injection_pressure",
            bc_type="FunctionDirichletBC",
            variable=pressure_variable,
            boundary_name=injection_well_boundary_name,
            params={"function": injection_pressure_function_name}
        )

        # 2. Confine X-Displacement BC
        self.add_boundary_condition(
            name="confinex",
            bc_type="DirichletBC",
            variable=disp_x_variable,
            boundary_name=confine_disp_x_boundaries,
            params={"value": 0}
        )

        # 3. Confine Y-Displacement BC
        self.add_boundary_condition(
            name="confiney",
            bc_type="DirichletBC",
            variable=disp_y_variable,
            boundary_name=confine_disp_y_boundaries,
            params={"value": 0}
        )

        print(f"Info: Set standard hydraulic fracturing BCs using predefined set.")
        return self

    def _get_or_create_user_objects_moose_block(self) -> 'MooseBlock':
        """
        Retrieves the main 'UserObjects' MooseBlock from self._top_level_blocks,
        or creates and adds it if it doesn't exist.
        Internal helper method.
        """
        for block in self._top_level_blocks:
            if block.block_name == "UserObjects":
                return block
        # If not found, create it
        uo_block = MooseBlock("UserObjects")
        self._top_level_blocks.append(uo_block)
        return uo_block

    def set_porous_flow_dictator(self,
                                 dictator_name: str = "dictator",
                                 porous_flow_variables: Union[str, List[str]] = "pp",
                                 num_fluid_phases: int = 1,
                                 num_fluid_components: int = 1,
                                 **other_params) -> 'ModelBuilder':
        """
        Sets up the PorousFlowDictator UserObject with common parameters.
        If a UserObject with 'dictator_name' already exists, it will be replaced.

        Args:
            dictator_name (str, optional): Name for the PorousFlowDictator. Defaults to "dictator".
            porous_flow_variables (Union[str, List[str]], optional): Variable(s) governed by PorousFlow.
                                                                    Defaults to "pp".
            num_fluid_phases (int, optional): Number of fluid phases. Defaults to 1.
            num_fluid_components (int, optional): Number of fluid components. Defaults to 1.
            **other_params: Additional parameters for the PorousFlowDictator.

        Returns:
            ModelBuilder: Returns self for chaining.
        """
        uo_main_block = self._get_or_create_user_objects_moose_block()

        # Remove existing dictator with the same name to replace it
        uo_main_block.sub_blocks = [
            sb for sb in uo_main_block.sub_blocks if sb.block_name != dictator_name
        ]

        vars_str = ' '.join(porous_flow_variables) if isinstance(porous_flow_variables, list) else porous_flow_variables

        params = {
            "porous_flow_vars": vars_str,
            "number_fluid_phases": num_fluid_phases,
            "number_fluid_components": num_fluid_components,
            **other_params
        }

        dictator_obj = MooseBlock(dictator_name, block_type="PorousFlowDictator")
        for p_name, p_val in params.items():
            dictator_obj.add_param(p_name, p_val)

        uo_main_block.add_sub_block(dictator_obj)
        print(f"Info: Set PorousFlowDictator '{dictator_name}'.")
        return self

    def add_user_object(self,
                        name: str,
                        uo_type: str,
                        params: Optional[Dict[str, Any]] = None) -> 'ModelBuilder':
        """
        Adds a single, generic UserObject to the [UserObjects] block.
        If a UserObject with the same 'name' already exists, it will be replaced.

        Args:
            name (str): The user-chosen name for this UserObject sub-block.
            uo_type (str): The MOOSE type for the UserObject.
            params (Optional[Dict[str, Any]], optional): A dictionary of parameters specific
                                                         to this uo_type. Defaults to None.

        Returns:
            ModelBuilder: Returns self for chaining.
        """
        uo_main_block = self._get_or_create_user_objects_moose_block()

        # Remove existing UO with the same name to replace it
        uo_main_block.sub_blocks = [
            sb for sb in uo_main_block.sub_blocks if sb.block_name != name
        ]

        uo_sub_block = MooseBlock(name, block_type=uo_type)
        if params:
            for p_name, p_val in params.items():
                uo_sub_block.add_param(p_name, p_val)

        uo_main_block.add_sub_block(uo_sub_block)
        print(f"Info: Added/Updated UserObject '{name}'.")
        return self

    def _get_or_create_postprocessors_moose_block(self) -> 'MooseBlock':
        """
        Retrieves the main 'Postprocessors' MooseBlock from self._top_level_blocks,
        or creates and adds it if it doesn't exist.
        Internal helper method.
        """
        for block in self._top_level_blocks:
            if block.block_name == "Postprocessors":
                return block
        # If not found, create it
        pp_block = MooseBlock("Postprocessors")
        self._top_level_blocks.append(pp_block)
        return pp_block

    def add_postprocessor(self,
                          config: 'PostprocessorConfigBase') -> 'ModelBuilder':  # Use forward reference for PostprocessorConfigBase
        """
        Adds a single postprocessor to the [Postprocessors] block based on the provided config object.

        Args:
            config (PostprocessorConfigBase): A configuration object derived from PostprocessorConfigBase
                                              (e.g., PointValueSamplerConfig, LineValueSamplerConfig).

        Returns:
            ModelBuilder: Returns self for chaining.
        """
        if not isinstance(config, PostprocessorConfigBase):  # Ensure PostprocessorConfigBase is imported
            raise TypeError("config must be an instance of a class derived from PostprocessorConfigBase.")

        pp_main_block = self._get_or_create_postprocessors_moose_block()

        # Create the sub-block for this specific postprocessor
        pp_sub_block = MooseBlock(config.name, block_type=config.pp_type)

        # Add common parameters from PostprocessorConfigBase
        if config.execute_on:
            exec_on_str = ' '.join(config.execute_on) if isinstance(config.execute_on, list) else config.execute_on
            pp_sub_block.add_param("execute_on", exec_on_str)

        if config.variable:
            pp_sub_block.add_param("variable", config.variable)

        if config.variables:  # For postprocessors that take a list of variables
            pp_sub_block.add_param("variables", ' '.join(config.variables))  # MOOSE usually expects space-separated

        # Add type-specific parameters by checking the instance type
        if isinstance(config, PointValueSamplerConfig):  # Ensure PointValueSamplerConfig is imported
            if config.point:  # point is mandatory for PointValueSamplerConfig as per its __init__
                pp_sub_block.add_param("point", config.point)

        # For LineValueSamplerConfig, its specific parameters (start_point, end_point, num_points, etc.)
        # are already placed into its 'other_params' dictionary by its __init__ method.
        # So, they will be handled by the loop below.

        # Add all parameters from the 'other_params' dictionary
        if config.other_params:
            for p_name, p_val in config.other_params.items():
                pp_sub_block.add_param(p_name, p_val)

        pp_main_block.add_sub_block(pp_sub_block)
        print(f"Info: Added Postprocessor '{config.name}' of type '{config.pp_type}'.")
        return self

    # --- File Generation ---
    def generate_input_file(self, output_filepath: str):
        """
        Generates the MOOSE input file by rendering all top-level MooseBlock objects.
        This will render all the blocks in self._top_level_blocks.
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

    @staticmethod
    def build_example_with_ama(output_filepath: str = "example_with_ama.i"):
        # First, ensure AdaptivityConfig, IndicatorConfig, MarkerConfig are importable
        # from .configs import AdaptivityConfig, IndicatorConfig, MarkerConfig

        builder = ModelBuilder(project_name="ExampleWithAMA")

        # 1. Define main domain (copied from a previous example for context)
        builder.set_main_domain_parameters_2d(
            domain_name="matrix",
            length=1000, height=500,
            num_elements_x=20, num_elements_y=10,  # Coarser for AMA demo
            moose_block_id=0
        )

        # 2. Define SRV (example)
        srv_conf = SRVConfig(name="SRVZone", length=300, height=80, center_x=500, center_y=250)
        builder.add_srv_zone_2d(srv_config=srv_conf, target_moose_block_id=1)

        # 3. Define Fracture (example)
        frac_conf = HydraulicFractureConfig(name="MainFrac", length=200, height=0.2, center_x=500, center_y=250)
        builder.add_hydraulic_fracture_2d(fracture_config=frac_conf, target_moose_block_id=2)

        # 4. Add Variables (minimal for AMA example, assuming 'pp' is used)
        # This part would use your actual add_variables_block method
        # For now, let's assume a MooseBlock for Variables is created and 'pp' is defined
        # Example:
        # var_block = builder._get_or_create_toplevel_moose_block("Variables") # Assuming such helper
        # pp_var = MooseBlock("pp")
        # pp_var.add_param("initial_condition", 0)
        # var_block.add_sub_block(pp_var)

        # --- Option 1: Enable AMA using default template settings ---
        builder.set_adaptivity_options(
            enable=True,
            default_template_settings={
                "monitored_variable": "pp",  # Assuming 'pp' is the primary variable
                "refine_fraction": 0.4,
                "coarsen_fraction": 0.1,
                "steps": 3
            }
        )

        # --- Option 2: Enable AMA using a detailed AdaptivityConfig object ---
        # from fiberis.moose.config import IndicatorConfig, MarkerConfig
        # indicator_conf = IndicatorConfig(name="pp_gradient", type="GradientJumpIndicator", params={"variable": "pp"})
        # marker_conf = MarkerConfig(name="pp_frac_marker", type="ErrorFractionMarker", params={"indicator": "pp_gradient", "refine": 0.5, "coarsen":0.05})
        # custom_ama_config = AdaptivityConfig(marker_to_use="pp_frac_marker", steps=2, indicators=[indicator_conf], markers=[marker_conf])
        # builder.set_adaptivity_options(enable=True, config=custom_ama_config)

        # --- Option 3: Disable AMA ---
        # builder.set_adaptivity_options(enable=False)

        # (Add other necessary blocks like Kernels, Executioner, Outputs for a runnable sim)
        # For this example, we focus on generating the [Adaptivity] block structure.

        builder.generate_input_file(output_filepath)
        print(f"Example file with AMA settings generated: {output_filepath}")

    @staticmethod
    def build_example_with_bcs_uos(output_filepath: str = "example_with_bcs_uos.i"):
        # This static method demonstrates the set_hydraulic_fracturing_bcs (scenario-specific)
        builder = ModelBuilder(project_name="ExampleWithScenarioBCsAndUOs")

        builder.set_main_domain_parameters_2d(
            domain_name="matrix", length=1000, height=500,
            num_elements_x=20, num_elements_y=10, moose_block_id=0
        )
        builder.add_nodeset_by_coord(
            nodeset_op_name="injection_point_nodes", new_boundary_name="injection_well",
            coordinates=(500, 250)
        )
        builder.set_hydraulic_fracturing_bcs(
            injection_well_boundary_name="injection_well",
            injection_pressure_function_name="pres_func",
            confine_disp_x_boundaries=["left", "right"],
            confine_disp_y_boundaries=["top", "bottom"],
        )
        builder.set_porous_flow_dictator(porous_flow_variables="pp")
        builder.generate_input_file(output_filepath)
        print(f"Example file with scenario BCs and UserObjects generated: {output_filepath}")

    @staticmethod
    def build_example_with_individual_bcs(output_filepath: str = "example_with_individual_bcs.i"):
        # This new static method demonstrates the generic add_boundary_condition
        builder = ModelBuilder(project_name="ExampleWithIndividualBCsAndUOs")

        # 1. Define main domain
        builder.set_main_domain_parameters_2d(
            domain_name="matrix", length=1000, height=500,
            num_elements_x=20, num_elements_y=10, moose_block_id=0
        )
        # Assume sidesets 'left', 'right', 'top', 'bottom' are known.

        # 2. Add nodeset for injection
        builder.add_nodeset_by_coord(
            nodeset_op_name="injection_point_nodes", new_boundary_name="injection_well",
            coordinates=(500, 250)
        )
        # Optionally, add a production point nodeset
        builder.add_nodeset_by_coord(
            nodeset_op_name="production_point_nodes", new_boundary_name="production_well",
            coordinates=(700, 250)  # Example coordinates
        )

        # 3. Add Boundary Conditions one by one using the generic method
        # Injection pressure
        builder.add_boundary_condition(
            name="injection_pressure_bc",
            bc_type="FunctionDirichletBC",
            variable="pp",
            boundary_name="injection_well",
            params={"function": "pres_func"}
        )
        # Confine x-displacement on left and right boundaries
        builder.add_boundary_condition(
            name="confine_disp_x_lr",
            bc_type="DirichletBC",
            variable="disp_x",
            boundary_name=["left", "right"],
            params={"value": 0}
        )
        # Confine y-displacement on top boundary
        builder.add_boundary_condition(
            name="confine_disp_y_top",
            bc_type="DirichletBC",
            variable="disp_y",  # Assuming 'disp_y' variable
            boundary_name="top",
            params={"value": 0}
        )
        # Example: No-flow (zero Neumann) on bottom boundary for pressure
        # builder.add_boundary_condition(
        #     name="no_flow_bottom",
        #     bc_type="NeumannBC",
        #     variable="pp",
        #     boundary_name="bottom",
        #     params={"value": 0.0}
        # )

        # 4. Set PorousFlowDictator UserObject (using the specific method or generic add_user_object)
        builder.set_porous_flow_dictator(
            porous_flow_variables="pp",
            num_fluid_phases=1,
            num_fluid_components=1
        )
        # Alternatively, using the generic method:
        # builder.add_user_object(
        #     name="dictator",
        #     uo_type="PorousFlowDictator",
        #     params={
        #         "porous_flow_vars": "pp",
        #         "number_fluid_phases": 1,
        #         "number_fluid_components": 1
        #     }
        # )

        # (Placeholder for adding Variables, Functions, Kernels etc. for a runnable simulation)
        # Example: Add a placeholder for Variables and Functions to make the output more complete
        # This would use actual methods like add_variables_block, add_functions_block

        # Placeholder Variables block
        # vars_block_obj = builder._get_or_create_toplevel_moose_block("Variables") # Assuming such helper
        # pp_var = MooseBlock("pp"); vars_block_obj.add_sub_block(pp_var)
        # dx_var = MooseBlock("disp_x"); vars_block_obj.add_sub_block(dx_var)
        # dy_var = MooseBlock("disp_y"); vars_block_obj.add_sub_block(dy_var)

        # Placeholder Functions block (for pres_func)
        # funcs_block_obj = builder._get_or_create_toplevel_moose_block("Functions")
        # pres_func_obj = MooseBlock("pres_func", block_type="ParsedFunction")
        # pres_func_obj.add_param("expression", "1.0e6 * t") # Dummy expression
        # funcs_block_obj.add_sub_block(pres_func_obj)

        builder.generate_input_file(output_filepath)
        print(f"Example file with individual BCs and UserObjects settings generated: {output_filepath}")

    @staticmethod
    def build_example_with_postprocessors(output_filepath: str = "example_with_pps.i"):
        # Ensure config classes are importable here if not globally
        # from .configs import PointValueSamplerConfig, LineValueSamplerConfig, HydraulicFractureConfig, SRVConfig

        builder = ModelBuilder(project_name="ExampleWithPostprocessors")

        # 1. Define main domain (example setup)
        builder.set_main_domain_parameters_2d(
            domain_name="matrix",
            length=1000, height=500,
            num_elements_x=20, num_elements_y=10,
            moose_block_id=0
        )
        # 2. Add a nodeset for injection (example setup)
        builder.add_nodeset_by_coord(
            nodeset_op_name="injection_point_nodes",
            new_boundary_name="injection_well",
            coordinates=(500.0, 0.0, 0.0)  # Using 3D coord for consistency with PointValue
        )
        builder.add_nodeset_by_coord(
            nodeset_op_name="production_point_nodes",
            new_boundary_name="production_well",
            coordinates=(610.0, 0.0, 0.0)
        )
        # Example monitoring points (assuming 2D, z=0 for PointValue)
        monitor_points_coords = [
            (435.0, 5.0, 0.0), (545.0, 2.0, 0.0), (545.0, 5.0, 0.0),
            (545.0, 10.0, 0.0), (545.0, 50.0, 0.0), (545.0, 200.0, 0.0),
            (435.0, 2.0, 0.0)
        ]

        # (Placeholder: Add Variables block and define 'pp', 'strain_yy', 'diffusivity')
        # builder.add_variables_block(...)
        # This is crucial for postprocessors to work.

        # 3. Add Postprocessors using the new config classes

        # PointValue samplers for 'pp'
        builder.add_postprocessor(PointValueSamplerConfig(name="pp_inj", variable="pp", point=(500.0, 0.0, 0.0)))
        builder.add_postprocessor(PointValueSamplerConfig(name="pp_prod", variable="pp", point=(610.0, 0.0, 0.0)))
        for i, coord in enumerate(monitor_points_coords):
            builder.add_postprocessor(
                PointValueSamplerConfig(name=f"pp_mon{i + 1}", variable="pp", point=coord)
            )

        # PointValue samplers for 'strain_yy'
        builder.add_postprocessor(
            PointValueSamplerConfig(name="strain_yy_inj", variable="strain_yy", point=(500.0, 0.0, 0.0)))
        builder.add_postprocessor(
            PointValueSamplerConfig(name="strain_yy_prod", variable="strain_yy", point=(610.0, 0.0, 0.0)))

        # PointValue sampler for 'diffusivity'
        builder.add_postprocessor(
            PointValueSamplerConfig(name="diff_inj", variable="diffusivity", point=(500.0, 0.0, 0.0)))

        # Example LineValueSampler
        builder.add_postprocessor(LineValueSamplerConfig(
            name="pressure_profile_x_axis",
            variable="pp",
            start_point=(0.0, 0.0, 0.0),
            end_point=(1000.0, 0.0, 0.0),
            num_points=51,
            output_vector=True,  # To get CSV output of all points
            execute_on="timestep_end"
        ))

        # (Add other necessary blocks like Kernels, Functions, BCs, Executioner, Outputs
        #  for a runnable simulation.)

        builder.generate_input_file(output_filepath)
        print(f"Example file with Postprocessors generated: {output_filepath}")


if __name__ == '__main__':  # This should be at the bottom of your model_builder.py file
    import os
    # Ensure ModelBuilder and MooseBlock are correctly importable
    # from .input_generator import MooseBlock # If ModelBuilder itself is not in the same file

    output_dir = "test_files/moose_input_file_test"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Test example with config
    # example_kernels_output_file = os.path.join(output_dir, "example_model_with_kernels.i")
    # ModelBuilder.build_example_with_configs(example_kernels_output_file)

    # 2. Test AMA
    # example_ama_output_file = os.path.join(output_dir, "model_builder_ama_output.i")
    #
    # # Call the static method to build and generate the file
    # ModelBuilder.build_example_with_ama(example_ama_output_file)

    # 3. Test the scenario-specific BCs setup
    # example_scenario_bcs_output_file = os.path.join(output_dir, "model_builder_scenario_bcs_uos_output.i")
    # ModelBuilder.build_example_with_bcs_uos(example_scenario_bcs_output_file)
    #
    # print("-" * 30)
    #
    # # Test the individual BCs setup
    # example_individual_bcs_output_file = os.path.join(output_dir, "model_builder_individual_bcs_uos_output.i")
    # ModelBuilder.build_example_with_individual_bcs(example_individual_bcs_output_file)

    # 4. Test Postprocessors
    example_pps_output_file = os.path.join(output_dir, "model_builder_postprocessors_output.i")

    # Call the static method to build and generate the file
    ModelBuilder.build_example_with_postprocessors(example_pps_output_file)