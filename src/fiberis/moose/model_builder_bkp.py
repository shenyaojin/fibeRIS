# User layer of input generator. Allow user to specify model parameters without
# having to interact with the input file directly.
# Higher-level API for building MOOSE input files, compared to input_generator.py and input_editor.py.
# Shenyao Jin, shenyaojin@mines.edu, 05/24/2025
#
# Refactored by Gemini on 06/06/2025 to use a single generic block getter.
# Corrected by Gemini on 06/06/2025 to fix various errors and add features.
# Final fix by Gemini on 06/07/2025 to correctly separate Postprocessors and VectorPostprocessors.

from typing import List, Dict, Any, Union, Tuple, Optional

# Import config classes and lower-level MooseBlock class.
# These imports assume the file structure is correct within the fiberis project.
from fiberis.moose.config import HydraulicFractureConfig, SRVConfig, AdaptivityConfig, \
    PointValueSamplerConfig, LineValueSamplerConfig, PostprocessorConfigBase, \
    SimpleFluidPropertiesConfig, MatrixConfig
from fiberis.moose.input_generator import MooseBlock
from fiberis.analyzer.Data1D import core1D


class ModelBuilder:
    """
    This class provides a high-level API to construct MOOSE input files.
    It uses configuration objects (e.g., HydraulicFractureConfig, SRVConfig)
    to define physical features and translates them into MOOSE mesh operations and other blocks.
    """

    def __init__(self, project_name: str):
        """
        Initializes the ModelBuilder.

        Args:
            project_name (str): The name of the project, used for identification.
        """
        self.project_name: str = project_name
        self._top_level_blocks: List[MooseBlock] = []

        # Configuration object storage
        self.matrix_config: Optional[MatrixConfig] = None
        self.srv_configs: List[SRVConfig] = []
        self.fracture_configs: List[HydraulicFractureConfig] = []
        self.fluid_properties_configs: List[SimpleFluidPropertiesConfig] = []

        # Mesh specific tracking
        self._block_id_to_name_map: Dict[int, str] = {}
        self._next_available_block_id: int = 1
        self._main_domain_block_id: int = 0
        self._last_mesh_op_name_within_mesh_block: Optional[str] = None

    def _generate_unique_op_name(self, base_name: str, existing_names_list: List[str]) -> str:
        """Generates a unique name for an operation within a list of existing names."""
        count = 1
        op_name = base_name
        while op_name in existing_names_list:
            op_name = f"{base_name}_{count}"
            count += 1
        return op_name

    def _get_or_create_toplevel_moose_block(self, block_name: str) -> MooseBlock:
        """
        Retrieves a top-level MooseBlock from the internal list by its name,
        or creates and adds it if it doesn't exist.
        """
        for block in self._top_level_blocks:
            if block.block_name == block_name:
                return block
        new_block = MooseBlock(block_name)
        self._top_level_blocks.append(new_block)
        return new_block

    # --- Configuration Setup ---
    def set_matrix_config(self, config: 'MatrixConfig') -> 'ModelBuilder':
        """Sets the configuration for the main matrix block."""
        self.matrix_config = config
        return self

    def add_srv_config(self, config: 'SRVConfig') -> 'ModelBuilder':
        """Adds a configuration for a Stimulated Reservoir Volume (SRV) zone."""
        self.srv_configs.append(config)
        return self

    def add_fracture_config(self, config: 'HydraulicFractureConfig') -> 'ModelBuilder':
        """Adds a configuration for a hydraulic fracture."""
        self.fracture_configs.append(config)
        return self

    def add_fluid_properties_config(self, config: 'SimpleFluidPropertiesConfig') -> 'ModelBuilder':
        """Adds or replaces a fluid properties configuration."""
        self.fluid_properties_configs = [c for c in self.fluid_properties_configs if c.name != config.name]
        self.fluid_properties_configs.append(config)
        return self

    # --- Variables ---
    def add_variables(self, variables: List[Union[str, Dict[str, Any]]]) -> 'ModelBuilder':
        """
        Adds variables to the [Variables] block.

        Args:
            variables (List[Union[str, Dict[str, Any]]]): A list where each item
                is either a string (the variable name) or a dictionary.
                If a dictionary, it must have a 'name' key and can have an
                optional 'params' key which is a dict of parameters.
                e.g., ["pp", {"name": "disp_x", "params": {"initial_condition": 0.0}}]
        """
        vars_block = self._get_or_create_toplevel_moose_block("Variables")
        for var_config in variables:
            if isinstance(var_config, str):
                var_block = MooseBlock(var_config)
                vars_block.add_sub_block(var_block)
            elif isinstance(var_config, dict):
                name = var_config.get("name")
                if not name:
                    raise ValueError("Variable configuration dictionary must have a 'name' key.")
                var_block = MooseBlock(name)
                if "params" in var_config:
                    for p_name, p_val in var_config["params"].items():
                        var_block.add_param(p_name, p_val)
                vars_block.add_sub_block(var_block)
            else:
                raise TypeError(f"Invalid variable configuration: {var_config}")

        print(f"Info: Added {len(variables)} variables.")
        return self

    # --- Mesh Construction ---
    def set_main_domain_parameters_2d(self,
                                      domain_name: str,
                                      length: float,
                                      height: float,
                                      num_elements_x: int,
                                      num_elements_y: Optional[int],  # allow num_elements_y to be None
                                      xmin: float = 0.0,
                                      ymin: float = 0.0,
                                      moose_block_id: int = 0,
                                      **additional_generator_params) -> 'ModelBuilder':
        """
        Defines the main 2D reservoir domain (matrix) using a GeneratedMeshGenerator.
        """
        mesh_moose_block = self._get_or_create_toplevel_moose_block("Mesh")
        existing_sub_block_names = [sb.block_name for sb in mesh_moose_block.sub_blocks]

        self._main_domain_block_id = moose_block_id
        self._block_id_to_name_map[self._main_domain_block_id] = domain_name

        gmg_op_name = self._generate_unique_op_name(f"{domain_name}_base_mesh", existing_sub_block_names)

        # --- START: Corrected Parameters Logic ---
        gmg_params = {
            "dim": 2, "nx": num_elements_x,
            "xmin": xmin, "xmax": xmin + length, "ymin": ymin, "ymax": ymin + height,
            **additional_generator_params
        }
        if num_elements_y is not None:
            gmg_params['ny'] = num_elements_y
        # --- END: Corrected Parameters Logic ---

        gmg_sub_block = MooseBlock(gmg_op_name, block_type="GeneratedMeshGenerator")
        for p_name, p_val in gmg_params.items():
            gmg_sub_block.add_param(p_name, p_val)

        mesh_moose_block.add_sub_block(gmg_sub_block)
        self._last_mesh_op_name_within_mesh_block = gmg_op_name
        return self

    # In ModelBuilder class
    def add_hydraulic_fracture_2d(self,
                                  fracture_config: HydraulicFractureConfig,
                                  target_moose_block_id: Optional[int] = None) -> 'ModelBuilder':
        """
        (Refactored) Adds a hydraulic fracture subdomain. Refinement is now handled separately.
        All subdomain generators now operate on the base mesh.
        """
        # Block ID management (this part was fixed before and is correct)
        block_id_to_use = target_moose_block_id
        if block_id_to_use is None:
            block_id_to_use = self._next_available_block_id
        elif block_id_to_use == self._main_domain_block_id or block_id_to_use in self._block_id_to_name_map:
            raise ValueError(f"Block ID {block_id_to_use} for '{fracture_config.name}' is already in use.")
        self._block_id_to_name_map[block_id_to_use] = fracture_config.name
        self._next_available_block_id = max(self._next_available_block_id, block_id_to_use + 1)

        # --- Key Change: Create BBox from the BASE MESH, not the last operation ---
        mesh_moose_block = self._get_or_create_toplevel_moose_block("Mesh")
        base_mesh_name = "matrix_base_mesh"  # Assuming this is the name set in set_main_domain_parameters_2d

        op_name = f"{fracture_config.name}_bbox"
        bbox_sub_block = MooseBlock(op_name, block_type="SubdomainBoundingBoxGenerator")
        bbox_sub_block.add_param("input", base_mesh_name)
        bbox_sub_block.add_param("block_id", block_id_to_use)

        half_length = fracture_config.length / 2.0
        half_height = fracture_config.height / 2.0
        bbox_sub_block.add_param("bottom_left",
                                 f"'{fracture_config.center_x - half_length} {fracture_config.center_y - half_height} 0.00000001'")
        bbox_sub_block.add_param("top_right",
                                 f"'{fracture_config.center_x + half_length} {fracture_config.center_y + half_height} 0'")

        mesh_moose_block.add_sub_block(bbox_sub_block)
        self._last_mesh_op_name_within_mesh_block = op_name  # Still track the last op, for refinement to use later
        return self

    # In ModelBuilder class
    def add_srv_zone_2d(self,
                        srv_config: SRVConfig,
                        target_moose_block_id: Optional[int] = None) -> 'ModelBuilder':
        """
        (Refactored) Adds an SRV subdomain. Refinement is now handled separately.
        All subdomain generators now operate on the base mesh.
        """
        # Block ID management
        block_id_to_use = target_moose_block_id
        if block_id_to_use is None:
            block_id_to_use = self._next_available_block_id
        elif block_id_to_use == self._main_domain_block_id or block_id_to_use in self._block_id_to_name_map:
            raise ValueError(f"Block ID {block_id_to_use} for '{srv_config.name}' is already in use.")
        self._block_id_to_name_map[block_id_to_use] = srv_config.name
        self._next_available_block_id = max(self._next_available_block_id, block_id_to_use + 1)

        # --- Key Change: Create BBox from the BASE MESH ---
        mesh_moose_block = self._get_or_create_toplevel_moose_block("Mesh")
        base_mesh_name = "matrix_base_mesh"

        op_name = f"{srv_config.name}_bbox"
        bbox_sub_block = MooseBlock(op_name, block_type="SubdomainBoundingBoxGenerator")
        bbox_sub_block.add_param("input", base_mesh_name)
        bbox_sub_block.add_param("block_id", block_id_to_use)

        half_length = srv_config.length / 2.0
        half_height = srv_config.height / 2.0
        bbox_sub_block.add_param("bottom_left",
                                 f"'{srv_config.center_x - half_length} {srv_config.center_y - half_height} 0.00000001'")
        bbox_sub_block.add_param("top_right",
                                 f"'{srv_config.center_x + half_length} {srv_config.center_y + half_height} 0'")

        mesh_moose_block.add_sub_block(bbox_sub_block)
        # This architecture does not chain BBox generators, but we need to know the last one for the refinement step
        self._last_mesh_op_name_within_mesh_block = op_name
        return self

    # In ModelBuilder class
    def refine_blocks(self,
                      op_name: str,
                      block_ids: List[int],
                      refinement_levels: Union[int, List[int]]) -> 'ModelBuilder':
        """Adds a dedicated refinement step for one or more blocks."""
        mesh_moose_block = self._get_or_create_toplevel_moose_block("Mesh")

        refine_sub_block = MooseBlock(op_name, block_type="RefineBlockGenerator")
        refine_sub_block.add_param("input", self._last_mesh_op_name_within_mesh_block)

        str_block_ids = ' '.join(map(str, block_ids))
        refine_sub_block.add_param("block", f"'{str_block_ids}'")

        if isinstance(refinement_levels, list):
            if len(refinement_levels) != len(block_ids):
                raise ValueError("Length of refinement_levels must match length of block_ids.")
            str_ref_levels = ' '.join(map(str, refinement_levels))
            refine_sub_block.add_param("refinement", f"'{str_ref_levels}'")
        else:  # is an int
            # If one level is given for multiple blocks, create a list of the same level
            levels = [refinement_levels] * len(block_ids)
            str_ref_levels = ' '.join(map(str, levels))
            refine_sub_block.add_param("refinement", f"'{str_ref_levels}'")

        mesh_moose_block.add_sub_block(refine_sub_block)
        self._last_mesh_op_name_within_mesh_block = op_name
        return self

    def add_nodeset_by_coord(self,
                             nodeset_op_name: str,
                             new_boundary_name: str,
                             coordinates: Union[Tuple[float, ...], str],
                             **additional_params) -> 'ModelBuilder':
        """Adds an ExtraNodesetGenerator block to define a boundary by coordinates."""
        mesh_moose_block = self._get_or_create_toplevel_moose_block("Mesh")
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

    # --- Kernels ---
    def add_time_derivative_kernel(self,
                                   variable: str,
                                   kernel_name: Optional[str] = None) -> 'ModelBuilder':
        """Adds a TimeDerivative kernel."""
        kernels_main_block = self._get_or_create_toplevel_moose_block("Kernels")
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
        """Adds a FunctionDiffusion kernel."""
        kernels_main_block = self._get_or_create_toplevel_moose_block("Kernels")
        kernel_obj = MooseBlock(kernel_name, block_type="FunctionDiffusion")
        kernel_obj.add_param("variable", variable)
        kernel_obj.add_param("function", function_name)
        kernel_obj.add_param("block", ' '.join(block_names) if isinstance(block_names, list) else block_names)
        kernels_main_block.add_sub_block(kernel_obj)
        return self

    def add_anisotropic_diffusion_kernel(self,
                                         kernel_name: str,
                                         variable: str,
                                         block_names: Union[str, List[str]],
                                         tensor_coefficient: str) -> 'ModelBuilder':
        """Adds an AnisotropicDiffusion kernel."""
        kernels_main_block = self._get_or_create_toplevel_moose_block("Kernels")
        kernel_obj = MooseBlock(kernel_name, block_type="AnisotropicDiffusion")
        kernel_obj.add_param("variable", variable)
        kernel_obj.add_param("block", ' '.join(block_names) if isinstance(block_names, list) else block_names)
        kernel_obj.add_param("tensor_coeff", tensor_coefficient)
        kernels_main_block.add_sub_block(kernel_obj)
        return self

    def add_porous_flow_darcy_base_kernel(self,
                                          kernel_name: str,
                                          variable: str,
                                          gravity_vector: str = '0 0 0') -> 'ModelBuilder':
        """Adds a PorousFlowFullySaturatedDarcyBase kernel."""
        kernels_main_block = self._get_or_create_toplevel_moose_block("Kernels")
        kernel_obj = MooseBlock(kernel_name, block_type="PorousFlowFullySaturatedDarcyBase")
        kernel_obj.add_param("variable", variable)
        kernel_obj.add_param("gravity", gravity_vector)
        kernels_main_block.add_sub_block(kernel_obj)
        return self

    def add_stress_divergence_tensor_kernel(self,
                                            kernel_name: str,
                                            variable: str,
                                            component: int) -> 'ModelBuilder':
        """Adds a StressDivergenceTensors kernel."""
        kernels_main_block = self._get_or_create_toplevel_moose_block("Kernels")
        kernel_obj = MooseBlock(kernel_name, block_type="StressDivergenceTensors")
        kernel_obj.add_param("variable", variable)
        kernel_obj.add_param("component", component)
        kernels_main_block.add_sub_block(kernel_obj)
        return self

    def add_porous_flow_effective_stress_coupling_kernel(self,
                                                         kernel_name: str,
                                                         variable: str,
                                                         component: int,
                                                         biot_coefficient: float) -> 'ModelBuilder':
        """Adds a PorousFlowEffectiveStressCoupling kernel."""
        kernels_main_block = self._get_or_create_toplevel_moose_block("Kernels")
        kernel_obj = MooseBlock(kernel_name, block_type="PorousFlowEffectiveStressCoupling")
        kernel_obj.add_param("variable", variable)
        kernel_obj.add_param("component", component)
        kernel_obj.add_param("biot_coefficient", biot_coefficient)
        kernels_main_block.add_sub_block(kernel_obj)
        return self

    def add_porous_flow_mass_volumetric_expansion_kernel(self,
                                                         kernel_name: str,
                                                         variable: str,
                                                         fluid_component: int = 0) -> 'ModelBuilder':
        """Adds a PorousFlowMassVolumetricExpansion kernel."""
        kernels_main_block = self._get_or_create_toplevel_moose_block("Kernels")
        kernel_obj = MooseBlock(kernel_name, block_type="PorousFlowMassVolumetricExpansion")
        kernel_obj.add_param("variable", variable)
        kernel_obj.add_param("fluid_component", fluid_component)
        kernels_main_block.add_sub_block(kernel_obj)
        return self

    # --- Adaptivity ---
    def set_adaptivity_options(self,
                               enable: bool = True,
                               config: Optional[AdaptivityConfig] = None,
                               default_template_settings: Optional[Dict[str, Any]] = None,
                               adaptivity_block_name: str = "Adaptivity") -> 'ModelBuilder':
        """
        Sets the options for the [Adaptivity] block.
        """
        if not enable:
            self._top_level_blocks = [
                block for block in self._top_level_blocks if block.block_name != adaptivity_block_name
            ]
            print(f"Info: Adaptivity block '{adaptivity_block_name}' removed (disabled).")
            return self

        # Get or create the block, then clear it before populating
        adapt_moose_block = self._get_or_create_toplevel_moose_block(adaptivity_block_name)
        adapt_moose_block.params.clear()
        adapt_moose_block.sub_blocks.clear()

        if config is not None:
            if not isinstance(config, AdaptivityConfig):
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
            print(f"Info: Configuring adaptivity block '{adaptivity_block_name}' using default template settings.")
            mon_var = default_template_settings.get("monitored_variable", "pp")
            ref_frac = default_template_settings.get("refine_fraction", 0.3)
            coarse_frac = default_template_settings.get("coarsen_fraction", 0.05)
            adapt_steps = default_template_settings.get("steps", 2)
            default_indicator_name = f"indicator_on_{mon_var}"
            default_marker_name = f"marker_for_{mon_var}"

            adapt_moose_block.add_param("marker", default_marker_name)
            adapt_moose_block.add_param("steps", adapt_steps)

            indicators_main_sub = MooseBlock("Indicators")
            default_indicator = MooseBlock(default_indicator_name, block_type="GradientJumpIndicator")
            default_indicator.add_param("variable", mon_var)
            indicators_main_sub.add_sub_block(default_indicator)
            adapt_moose_block.add_sub_block(indicators_main_sub)

            markers_main_sub = MooseBlock("Markers")
            default_marker = MooseBlock(default_marker_name, block_type="ErrorFractionMarker")
            default_marker.add_param("indicator", default_indicator_name)
            default_marker.add_param("refine", ref_frac)
            default_marker.add_param("coarsen", coarse_frac)
            markers_main_sub.add_sub_block(default_marker)
            adapt_moose_block.add_sub_block(markers_main_sub)
            print(f"Info: Applied default AMA template for variable '{mon_var}'.")
        else:
            print(
                f"Warning: Adaptivity enabled for '{adaptivity_block_name}', but no config provided. Applying basic fallback.")
            adapt_moose_block.add_param("marker", "default_marker")
            adapt_moose_block.add_param("steps", 1)

            indicators_sub = MooseBlock("Indicators")
            default_indicator_block = MooseBlock("default_indicator", "GradientJumpIndicator")
            default_indicator_block.add_param("variable", "pp")
            indicators_sub.add_sub_block(default_indicator_block)
            adapt_moose_block.add_sub_block(indicators_sub)

            markers_sub = MooseBlock("Markers")
            default_marker_block = MooseBlock("default_marker", "ErrorFractionMarker")
            default_marker_block.add_param("indicator", "default_indicator")
            default_marker_block.add_param("refine", 0.5)
            markers_sub.add_sub_block(default_marker_block)
            adapt_moose_block.add_sub_block(markers_sub)

        return self

    def _finalize_mesh_block_renaming(self):
        """Adds a RenameBlockGenerator at the end of the mesh operations if needed."""
        mesh_moose_block = self._get_or_create_toplevel_moose_block("Mesh")
        if self._block_id_to_name_map and self._last_mesh_op_name_within_mesh_block:
            old_block_ids, new_block_names = [], []
            for block_id in sorted(self._block_id_to_name_map.keys()):
                old_block_ids.append(str(block_id))
                new_block_names.append(self._block_id_to_name_map[block_id])
            if old_block_ids:
                rename_op_name = self._generate_unique_op_name("final_block_rename",
                                                               [sb.block_name for sb in mesh_moose_block.sub_blocks])
                rename_sub_block = MooseBlock(rename_op_name, block_type="RenameBlockGenerator")
                rename_sub_block.add_param("old_block", ' '.join(old_block_ids))
                rename_sub_block.add_param("new_block", ' '.join(new_block_names))
                rename_sub_block.add_param("input", self._last_mesh_op_name_within_mesh_block)
                mesh_moose_block.add_sub_block(rename_sub_block)
                self._last_mesh_op_name_within_mesh_block = rename_op_name

    # --- Boundary Conditions ---
    def add_boundary_condition(self,
                               name: str,
                               bc_type: str,
                               variable: str,
                               boundary_name: Union[str, List[str]],
                               params: Optional[Dict[str, Any]] = None) -> 'ModelBuilder':
        """Adds a single, generic boundary condition to the [BCs] block."""
        bcs_main_block = self._get_or_create_toplevel_moose_block("BCs")
        bc_sub_block = MooseBlock(name, block_type=bc_type)
        bc_sub_block.add_param("variable", variable)
        bc_sub_block.add_param("boundary",
                               ' '.join(boundary_name) if isinstance(boundary_name, list) else boundary_name)
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
        """Sets up a predefined set of typical boundary conditions for a hydraulic fracturing simulation."""
        bcs_main_block = self._get_or_create_toplevel_moose_block("BCs")
        bcs_main_block.sub_blocks.clear()
        self.add_boundary_condition(
            name="injection_pressure", bc_type="FunctionDirichletBC", variable=pressure_variable,
            boundary_name=injection_well_boundary_name, params={"function": injection_pressure_function_name}
        )
        self.add_boundary_condition(
            name="confinex", bc_type="DirichletBC", variable=disp_x_variable,
            boundary_name=confine_disp_x_boundaries, params={"value": 0}
        )
        self.add_boundary_condition(
            name="confiney", bc_type="DirichletBC", variable=disp_y_variable,
            boundary_name=confine_disp_y_boundaries, params={"value": 0}
        )
        print("Info: Set standard hydraulic fracturing BCs using predefined set.")
        return self

    # --- User Objects ---
    def add_user_object(self,
                        name: str,
                        uo_type: str,
                        params: Optional[Dict[str, Any]] = None) -> 'ModelBuilder':
        """Adds a single, generic UserObject, replacing any existing UO with the same name."""
        uo_main_block = self._get_or_create_toplevel_moose_block("UserObjects")
        uo_main_block.sub_blocks = [sb for sb in uo_main_block.sub_blocks if sb.block_name != name]
        uo_sub_block = MooseBlock(name, block_type=uo_type)
        if params:
            for p_name, p_val in params.items():
                uo_sub_block.add_param(p_name, p_val)
        uo_main_block.add_sub_block(uo_sub_block)
        print(f"Info: Added/Updated UserObject '{name}'.")
        return self

    def set_porous_flow_dictator(self,
                                 dictator_name: str = "dictator",
                                 porous_flow_variables: Union[str, List[str]] = "pp",
                                 num_fluid_phases: int = 1,
                                 num_fluid_components: int = 1,
                                 **other_params) -> 'ModelBuilder':
        """Convenience method to set up the PorousFlowDictator UserObject."""
        vars_str = ' '.join(porous_flow_variables) if isinstance(porous_flow_variables, list) else porous_flow_variables
        params = {
            "porous_flow_vars": vars_str,
            "number_fluid_phases": num_fluid_phases,
            "number_fluid_components": num_fluid_components,
            **other_params
        }
        return self.add_user_object(name=dictator_name, uo_type="PorousFlowDictator", params=params)

    # --- Postprocessors and VectorPostprocessors ---
    def add_postprocessor(self, config: PostprocessorConfigBase) -> 'ModelBuilder':
        """
        Adds a postprocessor based on the provided config object, correctly
        routing it to [Postprocessors] or [VectorPostprocessors].
        """
        if not isinstance(config, PostprocessorConfigBase):
            raise TypeError("config must be derived from PostprocessorConfigBase.")

        # Determine the correct top-level block based on the config type
        if isinstance(config, LineValueSamplerConfig):
            main_block = self._get_or_create_toplevel_moose_block("VectorPostprocessors")
        elif isinstance(config, PointValueSamplerConfig):
            main_block = self._get_or_create_toplevel_moose_block("Postprocessors")
        else:
            # Default to Postprocessors for any other type, with a warning.
            main_block = self._get_or_create_toplevel_moose_block("Postprocessors")
            print(f"Warning: Postprocessor type for '{config.name}' not explicitly handled. "
                  "Defaulting to [Postprocessors] block. This may be incorrect for vector types.")

        pp_sub_block = MooseBlock(config.name, block_type=config.pp_type)

        if config.execute_on:
            exec_on = ' '.join(config.execute_on) if isinstance(config.execute_on, list) else config.execute_on
            pp_sub_block.add_param("execute_on", exec_on)

        # This logic handles both single 'variable' and plural 'variables' attributes from the config.
        # It ensures that even if a config holds a list, it's converted to a space-separated string for MOOSE.
        if hasattr(config, 'variables') and config.variables:
            pp_sub_block.add_param("variable", ' '.join(config.variables))
        elif hasattr(config, 'variable') and config.variable:
            pp_sub_block.add_param("variable", config.variable)

        if isinstance(config, PointValueSamplerConfig) and config.point:
            pp_sub_block.add_param("point", config.point)

        for p_name, p_val in config.other_params.items():
            pp_sub_block.add_param(p_name, p_val)

        main_block.add_sub_block(pp_sub_block)
        print(f"Info: Added '{config.name}' to [{main_block.block_name}].")
        return self

    # --- Functions ---
    def add_piecewise_function_from_data1d(self,
                                           name: str,
                                           source_data1d: core1D.Data1D,
                                           other_params: Optional[Dict[str, Any]] = None) -> 'ModelBuilder':
        """Adds a PiecewiseConstant function from a fiberis Data1D object."""
        if not isinstance(source_data1d, core1D.Data1D):
            raise TypeError(f"source_data1d for function '{name}' must be a Data1D instance.")
        if source_data1d.taxis is None or source_data1d.data is None:
            raise ValueError(f"Data1D object '{source_data1d.name or 'Unnamed'}' needs 'taxis' and 'data'.")

        functions_main_block = self._get_or_create_toplevel_moose_block("Functions")
        func_sub_block = MooseBlock(name, block_type="PiecewiseConstant")
        func_sub_block.add_param("x", ' '.join(map(str, source_data1d.taxis)))
        func_sub_block.add_param("y", ' '.join(map(str, source_data1d.data)))

        if other_params:
            for p_name, p_val in other_params.items():
                func_sub_block.add_param(p_name, p_val)
        functions_main_block.add_sub_block(func_sub_block)
        print(f"Info: Added PiecewiseConstant Function '{name}' from Data1D source.")
        return self

    # --- Fluid Properties ---
    def add_simple_fluid_properties(self, config: SimpleFluidPropertiesConfig) -> 'ModelBuilder':
        """Adds a SimpleFluidProperties material based on a configuration object."""
        if not isinstance(config, SimpleFluidPropertiesConfig):
            raise TypeError("config must be a SimpleFluidPropertiesConfig instance.")

        fp_main_block = self._get_or_create_toplevel_moose_block("FluidProperties")
        fp_main_block.sub_blocks = [sb for sb in fp_main_block.sub_blocks if sb.block_name != config.name]

        fp_sub_block = MooseBlock(config.name, block_type="SimpleFluidProperties")
        fp_sub_block.add_param("bulk_modulus", config.bulk_modulus)
        fp_sub_block.add_param("viscosity", config.viscosity)
        fp_sub_block.add_param("density0", config.density0)
        if config.thermal_expansion is not None: fp_sub_block.add_param("thermal_expansion", config.thermal_expansion)
        if config.cp is not None: fp_sub_block.add_param("cp", config.cp)
        if config.cv is not None: fp_sub_block.add_param("cv", config.cv)
        if config.porepressure_coefficient is not None: fp_sub_block.add_param("porepressure_coefficient",
                                                                               config.porepressure_coefficient)

        fp_main_block.add_sub_block(fp_sub_block)
        print(f"Info: Added SimpleFluidProperties '{config.name}'.")
        return self

    # --- Materials Block Generation ---
    def add_poromechanics_materials(self,
                                    fluid_properties_name: str,
                                    biot_coefficient: float,
                                    solid_bulk_compliance: float,
                                    displacements: List[str] = ['disp_x', 'disp_y'],
                                    porepressure_variable: str = 'pp') -> 'ModelBuilder':
        """Builds the entire [Materials] block based on stored configs."""
        fluid_config = next((c for c in self.fluid_properties_configs if c.name == fluid_properties_name), None)
        if not fluid_config:
            raise ValueError(f"FluidPropertiesConfig '{fluid_properties_name}' not found.")

        self.add_simple_fluid_properties(config=fluid_config)

        mat_block = self._get_or_create_toplevel_moose_block("Materials")
        all_configs = ([self.matrix_config] if self.matrix_config else []) + self.srv_configs + self.fracture_configs
        all_block_names = [c.name for c in all_configs if c]

        for conf in all_configs:
            if not conf: continue
            poro_mat = MooseBlock(f"porosity_{conf.name}", "PorousFlowPorosityConst")
            poro_mat.add_param("porosity", conf.materials.porosity)
            poro_mat.add_param("block", conf.name)
            mat_block.add_sub_block(poro_mat)

            perm_mat = MooseBlock(f"permeability_{conf.name}", "PorousFlowPermeabilityConst")
            perm_mat.add_param("permeability", conf.materials.permeability)
            perm_mat.add_param("block", conf.name)
            mat_block.add_sub_block(perm_mat)

        mat_block.add_sub_block(MooseBlock("temperature", "PorousFlowTemperature"))

        biot_mod_params = {
            "biot_coefficient": biot_coefficient,
            "solid_bulk_compliance": solid_bulk_compliance,
            "fluid_bulk_modulus": fluid_config.bulk_modulus,
            "block": ' '.join(all_block_names)
        }
        biot_mod_mat = MooseBlock("biot_modulus", "PorousFlowConstantBiotModulus")
        for p_name, p_val in biot_mod_params.items():
            biot_mod_mat.add_param(p_name, p_val)
        mat_block.add_sub_block(biot_mod_mat)

        mat_block.add_sub_block(MooseBlock("massfrac", "PorousFlowMassFraction"))

        fluid_mat = MooseBlock("simple_fluid", "PorousFlowSingleComponentFluid")
        fluid_mat.add_param("fp", fluid_config.name)
        fluid_mat.add_param("phase", 0)
        mat_block.add_sub_block(fluid_mat)

        ps_mat = MooseBlock("PS", "PorousFlow1PhaseFullySaturated")
        ps_mat.add_param("porepressure", porepressure_variable)
        mat_block.add_sub_block(ps_mat)

        relp_mat = MooseBlock("relp", "PorousFlowRelativePermeabilityConst")
        relp_mat.add_param("phase", 0)
        mat_block.add_sub_block(relp_mat)

        mat_block.add_sub_block(MooseBlock("eff_fluid_pressure_qp", "PorousFlowEffectiveFluidPressure"))

        elasticity_mat = MooseBlock("elasticity_tensor_matrix", "ComputeIsotropicElasticityTensor")
        youngs_modulus = self.matrix_config.materials.youngs_modulus if self.matrix_config and self.matrix_config.materials.youngs_modulus is not None else 5.0E10
        poissons_ratio = self.matrix_config.materials.poissons_ratio if self.matrix_config and self.matrix_config.materials.poissons_ratio is not None else 0.2
        elasticity_mat.add_param("youngs_modulus", youngs_modulus)
        elasticity_mat.add_param("poissons_ratio", poissons_ratio)
        mat_block.add_sub_block(elasticity_mat)

        strain_mat = MooseBlock("strain", "ComputeSmallStrain")
        strain_mat.add_param("displacements", ' '.join(displacements))
        mat_block.add_sub_block(strain_mat)

        stress_mat = MooseBlock("stress", "ComputeLinearElasticStress")
        mat_block.add_sub_block(stress_mat)

        vol_strain_mat = MooseBlock("vol_strain", "PorousFlowVolumetricStrain")
        mat_block.add_sub_block(vol_strain_mat)

        print("Info: Added poromechanics materials based on stored configurations.")
        return self

    # --- AuxVariables and AuxKernels ---
    def add_standard_tensor_aux_vars_and_kernels(self, tensor_map: Dict[str, str]):
        """
        Automatically generates AuxVariables and AuxKernels for visualizing 2D tensors.

        Args:
            tensor_map (Dict[str, str]): A map where key is the material property name of the
                                        tensor (e.g., "stress", "strain") and value is the
                                        base name for the aux variables (e.g., "stress", "total_strain").
        """
        aux_vars_block = self._get_or_create_toplevel_moose_block("AuxVariables")
        aux_kernels_block = self._get_or_create_toplevel_moose_block("AuxKernels")

        components = [('xx', 0, 0), ('xy', 0, 1), ('yx', 1, 0), ('yy', 1, 1)]

        for material_tensor_name, aux_base_name in tensor_map.items():
            for suffix, i, j in components:
                var_name = f"{aux_base_name}_{suffix}"

                # Create AuxVariable
                aux_var = MooseBlock(var_name)
                aux_var.add_param("order", "CONSTANT")
                aux_var.add_param("family", "MONOMIAL")
                aux_vars_block.add_sub_block(aux_var)

                # Create AuxKernel
                aux_kernel = MooseBlock(var_name, block_type="RankTwoAux")
                aux_kernel.add_param("rank_two_tensor", material_tensor_name)
                aux_kernel.add_param("variable", var_name)
                aux_kernel.add_param("index_i", i)
                aux_kernel.add_param("index_j", j)
                aux_kernels_block.add_sub_block(aux_kernel)

        print(f"Info: Added standard AuxVariables and AuxKernels for tensors: {list(tensor_map.keys())}")
        return self

    # --- Executioner, Preconditioning, and Outputs ---
    def add_executioner_block(self,
                              end_time: float,
                              dt: float,
                              time_stepper_type: str = "IterationAdaptiveDT",
                              **kwargs) -> 'ModelBuilder':
        """
        Adds a standard [Executioner] block for transient simulations.

        Args:
            end_time (float): The simulation end time.
            dt (float): The initial time step size.
            time_stepper_type (str, optional): The type of time stepper. Defaults to "IterationAdaptiveDT".
            **kwargs: Additional parameters to set on the [Executioner] block, overriding defaults.
        """
        exec_block = self._get_or_create_toplevel_moose_block("Executioner")

        params = {
            "type": "Transient", "solve_type": "Newton", "end_time": end_time, "verbose": True,
            "l_tol": 1e-3, "l_max_its": 2000, "nl_max_its": 200, "nl_abs_tol": 1e-3, "nl_rel_tol": 1e-3,
            **kwargs
        }
        for p_name, p_val in params.items():
            exec_block.add_param(p_name, p_val)

        ts_block = MooseBlock("TimeStepper", block_type=time_stepper_type)
        ts_block.add_param("dt", dt)
        if time_stepper_type == "IterationAdaptiveDT":
            ts_block.add_param("timestep_limiting_function",
                               'constant_step_1 constant_step_2 adaptive_step adaptive_final')
            ts_block.add_param("force_step_every_function_point", True)

        exec_block.add_sub_block(ts_block)
        print("Info: Added [Executioner] block.")
        return self

    def add_preconditioning_block(self, active_preconditioner: str = 'mumps', **kwargs) -> 'ModelBuilder':
        """
        Adds a standard [Preconditioning] block with common options.
        """
        precond_block = self._get_or_create_toplevel_moose_block("Preconditioning")
        precond_block.add_param("active", active_preconditioner)

        mumps_block = MooseBlock("mumps", block_type="SMP")
        mumps_block.add_param("full", True)
        mumps_block.add_param("petsc_options",
                              '-snes_converged_reason -ksp_diagonal_scale -ksp_diagonal_scale_fix -ksp_gmres_modifiedgramschmidt -snes_linesearch_monitor')
        mumps_block.add_param("petsc_options_iname",
                              '-ksp_type -pc_type -pc_factor_mat_solver_package -pc_factor_shift_type')
        mumps_block.add_param("petsc_options_value", 'gmres      lu         mumps                     NONZERO')
        precond_block.add_sub_block(mumps_block)

        basic_block = MooseBlock("basic", block_type="SMP")
        basic_block.add_param("full", True)
        precond_block.add_sub_block(basic_block)

        preferred_block = MooseBlock("preferred_but_might_not_be_installed", block_type="SMP")
        preferred_block.add_param("full", True)
        preferred_block.add_param("petsc_options_iname", '-pc_type -pc_factor_mat_solver_package')
        preferred_block.add_param("petsc_options_value", ' lu         mumps')
        precond_block.add_sub_block(preferred_block)

        for p_name, p_val in kwargs.items():
            precond_block.add_param(p_name, p_val)

        print(f"Info: Added [Preconditioning] block with '{active_preconditioner}' active.")
        return self

    def add_outputs_block(self, exodus: bool = True, csv: bool = True, **kwargs) -> 'ModelBuilder':
        """
        Adds a standard [Outputs] block.
        """
        outputs_block = self._get_or_create_toplevel_moose_block("Outputs")
        if exodus:
            outputs_block.add_param("exodus", True)
        if csv:
            csv_block = MooseBlock("csv", block_type="CSV")
            outputs_block.add_sub_block(csv_block)

        for p_name, p_val in kwargs.items():
            outputs_block.add_param(p_name, p_val)

        print("Info: Added [Outputs] block.")
        return self

    # --- File Generation ---
    def generate_input_file(self, output_filepath: str):
        """Generates the MOOSE input file by rendering all configured blocks."""
        self._finalize_mesh_block_renaming()
        if not self._top_level_blocks:
            raise ValueError("No blocks defined. Cannot generate an empty input file.")

        all_rendered_blocks = [block.render(indent_level=0) + "\n[]" for block in self._top_level_blocks]
        with open(output_filepath, 'w') as f:
            f.write(f"# MOOSE input file generated by ModelBuilder for project: {self.project_name}\n\n")
            f.write("\n\n".join(all_rendered_blocks))
        print(f"MOOSE input file generated: {output_filepath}")

    # --- Static Methods for Examples ---
    @staticmethod
    def build_example_with_all_features(output_filepath: str = "example_full_build.i"):
        """A static method to demonstrate building a more complete input file."""
        import numpy as np
        from fiberis.moose.config import MatrixConfig, SRVConfig, HydraulicFractureConfig, ZoneMaterialProperties, \
            PointValueSamplerConfig, LineValueSamplerConfig, SimpleFluidPropertiesConfig
        from fiberis.analyzer.Data1D import core1D

        builder = ModelBuilder(project_name="FullFracExample")

        # 1. Define materials and fluid properties
        matrix_mats = ZoneMaterialProperties(porosity=0.05, permeability="'1e-15 0 0 0 1e-15 0 0 0 1e-16'",
                                             youngs_modulus=3e10, poissons_ratio=0.25)
        srv_mats = ZoneMaterialProperties(porosity=0.1, permeability="'1e-13 0 0 0 1e-13 0 0 0 1e-14'")
        frac_mats = ZoneMaterialProperties(porosity=0.5, permeability="'1e-10 0 0 0 1e-10 0 0 0 1e-11'")
        water_props = SimpleFluidPropertiesConfig(name="water", bulk_modulus=2.2E9, viscosity=1.0E-3, density0=1000.0)

        # 2. Add configs to builder
        builder.set_matrix_config(MatrixConfig(name="matrix", materials=matrix_mats))
        builder.add_srv_config(
            SRVConfig(name="SRV1", length=300, height=80, center_x=500, center_y=250, materials=srv_mats))
        builder.add_fracture_config(
            HydraulicFractureConfig(name="Frac1", length=200, height=0.2, center_x=500, center_y=250,
                                    materials=frac_mats))
        builder.add_fluid_properties_config(water_props)

        # 3. Add primary variables
        builder.add_variables([
            {"name": "pp", "params": {"initial_condition": 26.4E6}},
            "disp_x",
            "disp_y"
        ])

        # 4. Build Mesh
        builder.set_main_domain_parameters_2d(domain_name="matrix", length=1000, height=500, num_elements_x=50,
                                              num_elements_y=25)
        builder.add_srv_zone_2d(srv_config=builder.srv_configs[0], target_moose_block_id=1, refinement_passes=1)
        builder.add_hydraulic_fracture_2d(fracture_config=builder.fracture_configs[0], target_moose_block_id=2,
                                          refinement_passes=2)
        builder.add_nodeset_by_coord(nodeset_op_name="injection_well_nodes", new_boundary_name="injection_well",
                                     coordinates=(500, 250))

        # 5. Add Kernels
        builder.add_time_derivative_kernel(variable="pp")
        builder.add_porous_flow_darcy_base_kernel(kernel_name="flux", variable="pp")
        builder.add_stress_divergence_tensor_kernel(kernel_name="grad_stress_x", variable="disp_x", component=0)
        builder.add_stress_divergence_tensor_kernel(kernel_name="grad_stress_y", variable="disp_y", component=1)

        # 6. Build Materials Block
        builder.add_poromechanics_materials(fluid_properties_name="water", biot_coefficient=0.7,
                                            solid_bulk_compliance=1e-11)

        # 7. Add Functions
        pressure_data1d = core1D.Data1D(taxis=np.array([0, 10, 20]), data=np.array([27e6, 35e6, 30e6]))
        builder.add_piecewise_function_from_data1d(name="injection_pressure_func", source_data1d=pressure_data1d)

        # 8. Add Boundary Conditions
        builder.set_hydraulic_fracturing_bcs(
            injection_well_boundary_name="injection_well",
            injection_pressure_function_name="injection_pressure_func",
            confine_disp_x_boundaries=["left", "right"],
            confine_disp_y_boundaries=["top", "bottom"],
        )

        # 9. Add AuxVariables and AuxKernels automatically
        tensor_to_output_map = {"stress": "stress", "strain": "strain"}
        builder.add_standard_tensor_aux_vars_and_kernels(tensor_to_output_map)

        # 10. Add Postprocessors and VectorPostprocessors
        builder.add_postprocessor(PointValueSamplerConfig(name="pp_well", variable="pp", point=(500, 250, 0)))
        builder.add_postprocessor(LineValueSamplerConfig(
            name="pressure_x_profile",
            variable="pp stress_yy",
            start_point=(0, 250, 0),
            end_point=(1000, 250, 0),
            num_points=101
        ))

        # 11. Add Executioner, Preconditioning, and Outputs
        builder.add_executioner_block(end_time=3600, dt=100)
        builder.add_preconditioning_block(active_preconditioner='mumps')
        builder.add_outputs_block(exodus=True, csv=True)

        # 12. Generate the file
        builder.generate_input_file(output_filepath)


if __name__ == '__main__':
    import os

    output_dir = "test_files/moose_input_file_test"
    os.makedirs(output_dir, exist_ok=True)

    full_example_file = os.path.join(output_dir, "model_builder_full_example.i")
    ModelBuilder.build_example_with_all_features(full_example_file)
