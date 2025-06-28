# src/fiberis/moose/model_builder.py
# FINAL, COMPLETE REFACTORED VERSION BASED ON USER'S FILE AND NEW ARCHITECTURE
# This version implements the data-driven, stitched-mesh approach for multi-fracture modeling,
# while preserving all other functionalities from the original file.

from typing import List, Dict, Any, Union, Tuple, Optional
import numpy as np

# Import config classes and lower-level MooseBlock class from the user's original file.
from fiberis.moose.config import HydraulicFractureConfig, SRVConfig, AdaptivityConfig, \
    PointValueSamplerConfig, LineValueSamplerConfig, PostprocessorConfigBase, \
    SimpleFluidPropertiesConfig, MatrixConfig, AdaptiveTimeStepperConfig
from fiberis.moose.input_generator import MooseBlock
from fiberis.analyzer.Data1D import core1D


class ModelBuilder:
    """
    (Refactored) This class provides a high-level API to construct MOOSE input files.
    It now includes a robust, data-driven mesh generation system for complex,
    multi-fracture simulations based on a stitched mesh approach.
    """

    def __init__(self, project_name: str):
        """Initializes the ModelBuilder."""
        self.project_name: str = project_name
        self._top_level_blocks: List[MooseBlock] = []
        self.matrix_config: Optional[MatrixConfig] = None
        self.srv_configs: List[SRVConfig] = []
        self.fracture_configs: List[HydraulicFractureConfig] = []
        self.fluid_properties_configs: List[SimpleFluidPropertiesConfig] = []
        self._block_id_to_name_map: Dict[int, str] = {}
        self._next_available_block_id: int = 1
        self._last_mesh_op_name_within_mesh_block: Optional[str] = None

    def _generate_unique_op_name(self, base_name: str, existing_names_list: List[str]) -> str:
        count = 1
        op_name = base_name
        while op_name in existing_names_list:
            op_name = f"{base_name}_{count}"
            count += 1
        return op_name

    def _get_or_create_toplevel_moose_block(self, block_name: str) -> MooseBlock:
        for block in self._top_level_blocks:
            if block.block_name == block_name:
                return block
        new_block = MooseBlock(block_name)
        self._top_level_blocks.append(new_block)
        return new_block

    def _add_generic_mesh_generator(self, op_name: str, op_type: str, params: Dict[str, Any],
                                    input_op: Optional[str] = "USE_LAST") -> str:
        mesh_moose_block = self._get_or_create_toplevel_moose_block("Mesh")
        all_op_names = [sb.block_name for sb in mesh_moose_block.sub_blocks]
        unique_op_name = self._generate_unique_op_name(op_name, all_op_names)
        op_sub_block = MooseBlock(unique_op_name, block_type=op_type)
        final_input_op = self._last_mesh_op_name_within_mesh_block if input_op == "USE_LAST" else input_op
        if final_input_op and 'input' not in params and not (op_type == 'StitchedMeshGenerator' and 'inputs' in params):
            op_sub_block.add_param("input", final_input_op)
        for p_name, p_val in params.items():
            op_sub_block.add_param(p_name, p_val)
        mesh_moose_block.add_sub_block(op_sub_block)
        self._last_mesh_op_name_within_mesh_block = unique_op_name
        return unique_op_name

    def build_stitched_mesh_for_fractures(self,
                                          fracture_y_coords: List[float],
                                          domain_bounds: Tuple[float, float],
                                          domain_length: float = 1000.0,
                                          nx: int = 100,
                                          ny_per_layer_half: int = 20,
                                          bias_y: float = 1.3):
        ymin, ymax = domain_bounds
        all_y_points = sorted(list(set([ymin, ymax] + fracture_y_coords)), reverse=True)

        stitched_layer_names = []
        for i in range(len(all_y_points) - 1):
            y_upper, y_lower = all_y_points[i], all_y_points[i + 1]
            y_mid = (y_upper + y_lower) / 2.0

            panel_a_params = {'dim': 2, 'nx': nx, 'ny': ny_per_layer_half, 'bias_y': 1 / bias_y, 'xmin': 0,
                              'xmax': domain_length, 'ymin': y_mid, 'ymax': y_upper}
            panel_a_name = self._add_generic_mesh_generator(f"layer{i}_panel_a", "GeneratedMeshGenerator",
                                                            panel_a_params, input_op="")

            panel_b_params = {'dim': 2, 'nx': nx, 'ny': ny_per_layer_half, 'bias_y': bias_y, 'xmin': 0,
                              'xmax': domain_length, 'ymin': y_lower, 'ymax': y_mid}
            panel_b_name = self._add_generic_mesh_generator(f"layer{i}_panel_b", "GeneratedMeshGenerator",
                                                            panel_b_params, input_op="")

            layer_stitch_params = {'inputs': f"'{panel_a_name} {panel_b_name}'",
                                   'stitch_boundaries_pairs': "'bottom top'"}
            stitched_layer_name = self._add_generic_mesh_generator(f"stitched_layer_{i}", "StitchedMeshGenerator",
                                                                   layer_stitch_params, input_op="")
            stitched_layer_names.append(stitched_layer_name)

        if len(stitched_layer_names) > 1:
            current_mesh_name = stitched_layer_names[0]
            for i in range(1, len(stitched_layer_names)):
                next_layer_name = stitched_layer_names[i]
                final_stitch_params = {'inputs': f"'{current_mesh_name} {next_layer_name}'",
                                       'stitch_boundaries_pairs': "'bottom top'", 'clear_stitched_boundary_ids': True}
                current_mesh_name = self._add_generic_mesh_generator(f"final_stitch_{i - 1}", "StitchedMeshGenerator",
                                                                     final_stitch_params, input_op="")
            self._last_mesh_op_name_within_mesh_block = current_mesh_name
        elif stitched_layer_names:
            self._last_mesh_op_name_within_mesh_block = stitched_layer_names[0]

        print(f"Info: Successfully built stitched base mesh '{self._last_mesh_op_name_within_mesh_block}'.")
        return self

    def add_global_params(self, params: Dict[str, Any]) -> 'ModelBuilder':
        """Adds a [GlobalParams] block to the input file."""
        gp_block = self._get_or_create_toplevel_moose_block("GlobalParams")
        for p_name, p_val in params.items():
            gp_block.add_param(p_name, p_val)
        print("Info: Added [GlobalParams] block.")
        return self

    def add_hydraulic_fracture_2d(self, config: HydraulicFractureConfig, target_block_id: int):
        self._block_id_to_name_map[target_block_id] = config.name
        self._next_available_block_id = max(self._next_available_block_id, target_block_id + 1)
        half_length, half_height = config.length / 2.0, config.height / 2.0
        params = {'block_id': target_block_id,
                  'bottom_left': f"'{config.center_x - half_length} {config.center_y - half_height} 0'",
                  'top_right': f"'{config.center_x + half_length} {config.center_y + half_height} 0'"}
        self._add_generic_mesh_generator(f"{config.name}_bbox", "SubdomainBoundingBoxGenerator", params)
        return self

    def add_srv_zone_2d(self, config: SRVConfig, target_block_id: int):
        self._block_id_to_name_map[target_block_id] = config.name
        self._next_available_block_id = max(self._next_available_block_id, target_block_id + 1)
        half_length, half_height = config.length / 2.0, config.height / 2.0
        params = {'block_id': target_block_id,
                  'bottom_left': f"'{config.center_x - half_length} {config.center_y - half_height} 0'",
                  'top_right': f"'{config.center_x + half_length} {config.center_y + half_height} 0'"}
        self._add_generic_mesh_generator(f"{config.name}_bbox", "SubdomainBoundingBoxGenerator", params)
        return self

    def refine_blocks(self, op_name: str, block_ids: List[int], refinement_levels: Union[int, List[int]]):
        str_block_ids = ' '.join(map(str, block_ids))
        if isinstance(refinement_levels, list):
            if len(refinement_levels) != len(block_ids): raise ValueError(
                "Length of refinement_levels must match length of block_ids.")
            str_ref_levels = ' '.join(map(str, refinement_levels))
        else:
            str_ref_levels = ' '.join(map(str, [refinement_levels] * len(block_ids)))
        params = {'block': f"'{str_block_ids}'", 'refinement': f"'{str_ref_levels}'"}
        self._add_generic_mesh_generator(op_name, "RefineBlockGenerator", params)
        return self

    def _finalize_mesh_block_renaming(self):
        if self._block_id_to_name_map:
            if 0 not in self._block_id_to_name_map:
                self._block_id_to_name_map[0] = (self.matrix_config.name if self.matrix_config else "matrix")
            old_block_ids = sorted(self._block_id_to_name_map.keys())
            new_block_names = [self._block_id_to_name_map[bid] for bid in old_block_ids]
            params = {'old_block': f"'{' '.join(map(str, old_block_ids))}'",
                      'new_block': f"'{' '.join(new_block_names)}'"}
            self._add_generic_mesh_generator("final_block_rename", "RenameBlockGenerator", params)

    def add_named_boundary(self, new_boundary_name: str, bottom_left: Tuple, top_right: Tuple):
        """NEW: Creates a named boundary (sideset) using a bounding box. Essential for reliable BCs."""
        params = {'new_boundary_name': new_boundary_name,
                  'bottom_left': f"'{bottom_left[0]} {bottom_left[1]} {bottom_left[2]}'",
                  'top_right': f"'{top_right[0]} {top_right[1]} {top_right[2]}'"}
        self._add_generic_mesh_generator(f"create_{new_boundary_name}", "SideSetBoundingBoxGenerator", params)
        return self

    # --- PRESERVED METHODS FROM ORIGINAL FILE ---

    def add_global_params(self, params: Dict[str, Any]) -> 'ModelBuilder':
        """Adds a [GlobalParams] block to the input file."""
        gp_block = self._get_or_create_toplevel_moose_block("GlobalParams")
        for p_name, p_val in params.items():
            gp_block.add_param(p_name, p_val)
        print("Info: Added [GlobalParams] block.")
        return self

    def set_matrix_config(self, config: 'MatrixConfig') -> 'ModelBuilder':
        self.matrix_config = config
        return self

    def add_srv_config(self, config: 'SRVConfig') -> 'ModelBuilder':
        self.srv_configs.append(config)
        return self

    def add_fracture_config(self, config: 'HydraulicFractureConfig') -> 'ModelBuilder':
        self.fracture_configs.append(config)
        return self

    def add_fluid_properties_config(self, config: 'SimpleFluidPropertiesConfig') -> 'ModelBuilder':
        self.fluid_properties_configs = [c for c in self.fluid_properties_configs if c.name != config.name]
        self.fluid_properties_configs.append(config)
        return self

    def add_variables(self, variables: List[Union[str, Dict[str, Any]]]) -> 'ModelBuilder':
        vars_block = self._get_or_create_toplevel_moose_block("Variables")
        for var_config in variables:
            if isinstance(var_config, str):
                vars_block.add_sub_block(MooseBlock(var_config))
            elif isinstance(var_config, dict):
                name = var_config.get("name")
                if not name: raise ValueError("Variable config dict must have a 'name' key.")
                var_block = MooseBlock(name)
                if "params" in var_config:
                    for p_name, p_val in var_config["params"].items():
                        var_block.add_param(p_name, p_val)
                vars_block.add_sub_block(var_block)
            else:
                raise TypeError(f"Invalid variable configuration: {var_config}")
        print(f"Info: Added {len(variables)} variables.")
        return self

    def set_main_domain_parameters_2d(self, **kwargs):
        print("Warning: set_main_domain_parameters_2d is deprecated. Use build_stitched_mesh_for_fractures instead.")
        return self

    def add_nodeset_by_coord(self, nodeset_op_name: str, new_boundary_name: str,
                             coordinates: Union[Tuple[float, ...], str], **additional_params) -> 'ModelBuilder':
        params = {"new_boundary": new_boundary_name,
                  "coord": ' '.join(map(str, coordinates)) if isinstance(coordinates, tuple) else coordinates,
                  **additional_params}
        self._add_generic_mesh_generator(nodeset_op_name, "ExtraNodesetGenerator", params)
        return self

    def add_time_derivative_kernel(self, variable: str, kernel_name: Optional[str] = None) -> 'ModelBuilder':
        # (Content from original file)
        kernels_main_block = self._get_or_create_toplevel_moose_block("Kernels")
        name_to_use = kernel_name if kernel_name is not None else f"dot_{variable}"
        kernel_obj = MooseBlock(name_to_use, block_type="TimeDerivative")
        kernel_obj.add_param("variable", variable)
        kernels_main_block.add_sub_block(kernel_obj)
        return self

    def add_function_diffusion_kernel(self, kernel_name: str, variable: str, function_name: str,
                                      block_names: Union[str, List[str]]) -> 'ModelBuilder':
        # (Content from original file)
        kernels_main_block = self._get_or_create_toplevel_moose_block("Kernels")
        kernel_obj = MooseBlock(kernel_name, block_type="FunctionDiffusion")
        kernel_obj.add_param("variable", variable)
        kernel_obj.add_param("function", function_name)
        kernel_obj.add_param("block", ' '.join(block_names) if isinstance(block_names, list) else block_names)
        kernels_main_block.add_sub_block(kernel_obj)
        return self

    def add_anisotropic_diffusion_kernel(self, kernel_name: str, variable: str, block_names: Union[str, List[str]],
                                         tensor_coefficient: str) -> 'ModelBuilder':
        # (Content from original file)
        kernels_main_block = self._get_or_create_toplevel_moose_block("Kernels")
        kernel_obj = MooseBlock(kernel_name, block_type="AnisotropicDiffusion")
        kernel_obj.add_param("variable", variable)
        kernel_obj.add_param("block", ' '.join(block_names) if isinstance(block_names, list) else block_names)
        kernel_obj.add_param("tensor_coeff", tensor_coefficient)
        kernels_main_block.add_sub_block(kernel_obj)
        return self

    def add_porous_flow_darcy_base_kernel(self, kernel_name: str, variable: str,
                                          gravity_vector: str = '0 0 0') -> 'ModelBuilder':
        # (Content from original file)
        kernels_main_block = self._get_or_create_toplevel_moose_block("Kernels")
        kernel_obj = MooseBlock(kernel_name, block_type="PorousFlowFullySaturatedDarcyBase")
        kernel_obj.add_param("variable", variable)
        kernel_obj.add_param("gravity", gravity_vector)
        kernels_main_block.add_sub_block(kernel_obj)
        return self

    def add_stress_divergence_tensor_kernel(self, kernel_name: str, variable: str, component: int) -> 'ModelBuilder':
        # (Content from original file)
        kernels_main_block = self._get_or_create_toplevel_moose_block("Kernels")
        kernel_obj = MooseBlock(kernel_name, block_type="StressDivergenceTensors")
        kernel_obj.add_param("variable", variable)
        kernel_obj.add_param("component", component)
        kernels_main_block.add_sub_block(kernel_obj)
        return self

    def add_porous_flow_effective_stress_coupling_kernel(self, kernel_name: str, variable: str, component: int,
                                                         biot_coefficient: float) -> 'ModelBuilder':
        # (Content from original file)
        kernels_main_block = self._get_or_create_toplevel_moose_block("Kernels")
        kernel_obj = MooseBlock(kernel_name, block_type="PorousFlowEffectiveStressCoupling")
        kernel_obj.add_param("variable", variable)
        kernel_obj.add_param("component", component)
        kernel_obj.add_param("biot_coefficient", biot_coefficient)
        kernels_main_block.add_sub_block(kernel_obj)
        return self

    def add_porous_flow_mass_volumetric_expansion_kernel(self, kernel_name: str, variable: str,
                                                         fluid_component: int = 0) -> 'ModelBuilder':
        # (Content from original file)
        kernels_main_block = self._get_or_create_toplevel_moose_block("Kernels")
        kernel_obj = MooseBlock(kernel_name, block_type="PorousFlowMassVolumetricExpansion")
        kernel_obj.add_param("variable", variable)
        kernel_obj.add_param("fluid_component", fluid_component)
        kernels_main_block.add_sub_block(kernel_obj)
        return self

    def add_porous_flow_mass_volumetric_expansion_kernel(self,
                                                         kernel_name: str,
                                                         variable: str,
                                                         fluid_component: int,
                                                         displacements: Union[str, List[str]],
                                                         base_name: str,
                                                         block_names: Union[str, List[str]]) -> 'ModelBuilder':
        """
        Adds a PorousFlowMassVolumetricExpansion kernel to the Kernels block.

        This kernel is essential for poromechanics simulations as it accounts for the
        change in fluid mass resulting from the volumetric expansion or contraction
        of the solid matrix. This term is dependent on the time-derivative of the
        volumetric strain, which is calculated from the displacement variables.

        Args:
            kernel_name (str): The unique name for this kernel block in the MOOSE input file.
            variable (str): The name of the PorousFlow variable this kernel operates on (e.g., 'pp').
            fluid_component (int): The index of the fluid component this kernel applies to.
            displacements (Union[str, List[str]]): The displacement variable(s) (e.g., ['disp_x', 'disp_y']).
            base_name (str): The base name of the material property that computes volumetric strain,
                             typically matching the name of the ComputeSmallStrain material (e.g., 'strain').
            block_names (Union[str, List[str]]): A list or space-separated string of block names
                                                 or IDs where this kernel should be active.

        Returns:
            ModelBuilder: The instance of the model builder for method chaining.
        """
        # Get the main [Kernels] block, creating it if it doesn't exist.
        kernels_main_block = self._get_or_create_toplevel_moose_block("Kernels")

        # Create the specific MooseBlock for this kernel.
        kernel_obj = MooseBlock(kernel_name, block_type="PorousFlowMassVolumetricExpansion")

        # Add all the required parameters to the kernel block.
        kernel_obj.add_param("variable", variable)
        kernel_obj.add_param("fluid_component", fluid_component)
        kernel_obj.add_param("base_name", base_name)

        # Format list-based parameters into space-separated strings as expected by MOOSE.
        kernel_obj.add_param("block", ' '.join(block_names) if isinstance(block_names, list) else block_names)
        displacements_str = ' '.join(displacements) if isinstance(displacements, list) else displacements
        kernel_obj.add_param("displacements", f"'{displacements_str}'")

        # Add the fully configured kernel to the main [Kernels] block.
        kernels_main_block.add_sub_block(kernel_obj)
        print(f"Info: Added PorousFlowMassVolumetricExpansion Kernel '{kernel_name}'.")

        # Return the builder instance to allow for chaining commands.
        return self

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

    def add_boundary_condition(self, name: str, bc_type: str, variable: str, boundary_name: Union[str, List[str]],
                               params: Optional[Dict[str, Any]] = None) -> 'ModelBuilder':
        # (Content from original file)
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

    def set_hydraulic_fracturing_bcs(self, injection_well_boundary_name: str, injection_pressure_function_name: str,
                                     confine_disp_x_boundaries: Union[str, List[str]],
                                     confine_disp_y_boundaries: Union[str, List[str]], pressure_variable: str = "pp",
                                     disp_x_variable: str = "disp_x",
                                     disp_y_variable: str = "disp_y") -> 'ModelBuilder':
        # (Content from original file)
        bcs_main_block = self._get_or_create_toplevel_moose_block("BCs")
        bcs_main_block.sub_blocks.clear()
        self.add_boundary_condition(name="injection_pressure", bc_type="FunctionDirichletBC",
                                    variable=pressure_variable, boundary_name=injection_well_boundary_name,
                                    params={"function": injection_pressure_function_name})
        self.add_boundary_condition(name="confinex", bc_type="DirichletBC", variable=disp_x_variable,
                                    boundary_name=confine_disp_x_boundaries, params={"value": 0})
        self.add_boundary_condition(name="confiney", bc_type="DirichletBC", variable=disp_y_variable,
                                    boundary_name=confine_disp_y_boundaries, params={"value": 0})
        print("Info: Set standard hydraulic fracturing BCs using predefined set.")
        return self

    def add_user_object(self, name: str, uo_type: str, params: Optional[Dict[str, Any]] = None) -> 'ModelBuilder':
        # (Content from original file)
        uo_main_block = self._get_or_create_toplevel_moose_block("UserObjects")
        uo_main_block.sub_blocks = [sb for sb in uo_main_block.sub_blocks if sb.block_name != name]
        uo_sub_block = MooseBlock(name, block_type=uo_type)
        if params:
            for p_name, p_val in params.items():
                uo_sub_block.add_param(p_name, p_val)
        uo_main_block.add_sub_block(uo_sub_block)
        print(f"Info: Added/Updated UserObject '{name}'.")
        return self

    def set_porous_flow_dictator(self, dictator_name: str = "dictator",
                                 porous_flow_variables: Union[str, List[str]] = "pp", num_fluid_phases: int = 1,
                                 num_fluid_components: int = 1, **other_params) -> 'ModelBuilder':
        # (Content from original file)
        vars_str = ' '.join(porous_flow_variables) if isinstance(porous_flow_variables, list) else porous_flow_variables
        params = {"porous_flow_vars": vars_str, "number_fluid_phases": num_fluid_phases,
                  "number_fluid_components": num_fluid_components, **other_params}
        return self.add_user_object(name=dictator_name, uo_type="PorousFlowDictator", params=params)

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

    def add_piecewise_function_from_data1d(self, name: str, source_data1d: core1D.Data1D,
                                           other_params: Optional[Dict[str, Any]] = None) -> 'ModelBuilder':
        # (Content from original file)
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

    # Update the add_executioner_block method to handle adaptive stepper configuration
    def add_executioner_block(self,
                              end_time: float,
                              dt: float,
                              # The time_stepper_type is now an optional argument.
                              # If adaptive_stepper_config is provided, this will be overridden.
                              time_stepper_type: str = 'ConstantDT',
                              adaptive_stepper_config: Optional[AdaptiveTimeStepperConfig] = None,
                              **kwargs) -> 'ModelBuilder':
        """
        Adds a standard [Executioner] block. If an adaptive_stepper_config is provided,
        it automatically configures the advanced IterationAdaptiveDT TimeStepper.
        Otherwise, it defaults to a simpler TimeStepper (e.g., ConstantDT).

        Args:
            end_time: The simulation end time.
            dt: The initial or constant time step size.
            time_stepper_type: The fallback TimeStepper type if no adaptive config is given.
            adaptive_stepper_config: The configuration object for the adaptive stepper.
            **kwargs: Additional parameters for the [Executioner] block.
        """
        exec_block = self._get_or_create_toplevel_moose_block("Executioner")

        # Set base executioner parameters (e.g., solver tolerances)
        params = {
            "type": "Transient", "solve_type": "Newton", "end_time": end_time, "verbose": True,
            "l_tol": 1e-3, "l_max_its": 2000, "nl_max_its": 200, "nl_abs_tol": 1e-3, "nl_rel_tol": 1e-3,
            **kwargs
        }
        for p_name, p_val in params.items():
            exec_block.add_param(p_name, p_val)

        # --- Core "intelligent" logic for selecting the TimeStepper ---
        if adaptive_stepper_config:
            # If the user provides the advanced configuration object...
            print("Info: Configuring with IterationAdaptiveDT TimeStepper based on provided config.")

            # 1. Automatically call the *existing* method to create all required functions.
            #    This works because our new TimeStepperFunctionConfig is a subclass of Data1D.
            for func_config in adaptive_stepper_config.functions:
                self.add_piecewise_function_from_data1d(name=func_config.name, source_data1d=func_config)

            # 2. Extract all function names from the configuration.
            function_names = [f.name for f in adaptive_stepper_config.functions]

            # 3. Create the TimeStepper block with the correct type and parameters.
            ts_block = MooseBlock("TimeStepper", block_type="IterationAdaptiveDT")
            ts_block.add_param("dt", dt)
            ts_block.add_param("timestep_limiting_function", ' '.join(function_names))
            ts_block.add_param("force_step_every_function_point", True)
            exec_block.add_sub_block(ts_block)

        else:
            # If no advanced config is given, use a simple, robust TimeStepper to avoid errors.
            print(f"Info: No adaptive config provided. Defaulting to simple '{time_stepper_type}' TimeStepper.")
            ts_block = MooseBlock("TimeStepper", block_type=time_stepper_type)
            ts_block.add_param("dt", dt)
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

    @staticmethod
    def build_example_with_all_features(output_filepath: str = "example_full_build.i"):
        """
        UPDATED STATIC METHOD: Demonstrates building a complete input file
        using the new, robust, stitched-mesh architecture. This method showcases
        the intended workflow for setting up a complex poromechanics simulation.
        """
        # Because this is a static method, all necessary imports must be done
        # inside the method's scope.
        import numpy as np
        from fiberis.moose.config import MatrixConfig, SRVConfig, HydraulicFractureConfig, ZoneMaterialProperties, \
            PointValueSamplerConfig, LineValueSamplerConfig, SimpleFluidPropertiesConfig
        from fiberis.analyzer.Data1D import core1D

        # 1. Initialize the ModelBuilder
        builder = ModelBuilder(project_name="StitchedMeshFracExample")

        # 2. Define Material and Fluid Properties
        # Define material properties for different zones.
        matrix_mats = ZoneMaterialProperties(porosity=0.05, permeability="'1e-15 0 0 0 1e-15 0 0 0 1e-16'",
                                             youngs_modulus=3e10, poissons_ratio=0.25)
        srv_mats = ZoneMaterialProperties(porosity=0.1, permeability="'1e-13 0 0 0 1e-13 0 0 0 1e-14'")
        frac_mats = ZoneMaterialProperties(porosity=0.5, permeability="'1e-10 0 0 0 1e-10 0 0 0 1e-11'")
        # Define fluid properties.
        water_props = SimpleFluidPropertiesConfig(name="water", bulk_modulus=2.2E9, viscosity=1.0E-3, density0=1000.0)

        # 3. Add Configurations to the Builder
        # Set the main reservoir matrix configuration.
        builder.set_matrix_config(MatrixConfig(name="matrix", materials=matrix_mats))
        # Add configuration for the Stimulated Reservoir Volume (SRV).
        builder.add_srv_config(
            SRVConfig(name="SRV1", length=300, height=80, center_x=500, center_y=250, materials=srv_mats))
        # Add configuration for the hydraulic fracture.
        builder.add_fracture_config(
            HydraulicFractureConfig(name="Frac1", length=200, height=0.2, center_x=500, center_y=250,
                                    materials=frac_mats))
        # Add fluid properties configuration.
        builder.add_fluid_properties_config(water_props)

        # 4. Define Primary Variables
        builder.add_variables([
            {"name": "pp", "params": {"initial_condition": 26.4E6}},  # Porepressure
            "disp_x",  # Displacement in x
            "disp_y"  # Displacement in y
        ])

        # 5. Construct the Mesh using the new Stitched-Mesh approach
        print("\n--- Building Mesh ---")
        domain_length = 1000.0
        domain_bounds = (0, 500)
        # Define the y-coordinates of all horizontal features that require mesh seams.
        fracture_y_coords = [builder.fracture_configs[0].center_y]

        # Build the base mesh with layers stitched at fracture locations.
        builder.build_stitched_mesh_for_fractures(
            fracture_y_coords=fracture_y_coords,
            domain_bounds=domain_bounds,
            domain_length=domain_length,
            nx=50,
            ny_per_layer_half=15,  # Elements in each half-layer (above/below a seam)
            bias_y=1.2  # Bias meshing towards the seam
        )

        # Define the SRV and Fracture subdomains by assigning block IDs.
        builder.add_srv_zone_2d(config=builder.srv_configs[0], target_block_id=1)
        builder.add_hydraulic_fracture_2d(config=builder.fracture_configs[0], target_block_id=2)

        # Apply mesh refinement to the newly defined blocks.
        builder.refine_blocks(op_name="refine_features", block_ids=[1, 2], refinement_levels=[1, 2])

        # Define named boundaries (sidesets) for applying Boundary Conditions.
        builder.add_named_boundary("left", (0, 0, 0), (0, domain_bounds[1], 0))
        builder.add_named_boundary("right", (domain_length, 0, 0), (domain_length, domain_bounds[1], 0))
        builder.add_named_boundary("bottom", (0, 0, 0), (domain_length, 0, 0))
        builder.add_named_boundary("top", (0, domain_bounds[1], 0), (domain_length, domain_bounds[1], 0))

        # Define the injection well as a nodeset at a specific coordinate.
        builder.add_nodeset_by_coord("injection_well_nodes", "injection_well", (500, 250, 0))
        print("--- Mesh Construction Complete ---")

        # 6. Define Kernels (the physics)
        builder.add_time_derivative_kernel(variable="pp")
        builder.add_porous_flow_darcy_base_kernel(kernel_name="flux", variable="pp")
        builder.add_stress_divergence_tensor_kernel(kernel_name="grad_stress_x", variable="disp_x", component=0)
        builder.add_stress_divergence_tensor_kernel(kernel_name="grad_stress_y", variable="disp_y", component=1)
        builder.add_porous_flow_effective_stress_coupling_kernel(kernel_name="eff_stress_x", variable="disp_x",
                                                                 component=0, biot_coefficient=0.7)
        builder.add_porous_flow_effective_stress_coupling_kernel(kernel_name="eff_stress_y", variable="disp_y",
                                                                 component=1, biot_coefficient=0.7)
        builder.add_porous_flow_mass_volumetric_expansion_kernel(kernel_name="mass_exp", variable="pp")

        # 7. Define Materials block based on previously set configs
        builder.add_poromechanics_materials(fluid_properties_name="water", biot_coefficient=0.7,
                                            solid_bulk_compliance=1e-11)

        # 8. Define Functions (e.g., for time-dependent BCs)
        # Create a synthetic Data1D object for the injection pressure schedule.
        pressure_data1d = core1D.Data1D(taxis=np.array([0, 1800, 3600]), data=np.array([27e6, 45e6, 40e6]))
        builder.add_piecewise_function_from_data1d(name="injection_pressure_func", source_data1d=pressure_data1d)

        # 9. Define Boundary Conditions using the named boundaries
        builder.set_hydraulic_fracturing_bcs(
            injection_well_boundary_name="injection_well",
            injection_pressure_function_name="injection_pressure_func",
            confine_disp_x_boundaries="left right",  # Apply to multiple boundaries
            confine_disp_y_boundaries="top bottom",
        )

        # 10. Define AuxVariables and AuxKernels for outputting tensor components
        tensor_to_output_map = {"stress": "stress", "strain": "strain"}
        builder.add_standard_tensor_aux_vars_and_kernels(tensor_to_output_map)

        # 11. Define Postprocessors for data extraction
        builder.add_postprocessor(PointValueSamplerConfig(name="pp_well", variable="pp", point=(500, 250, 0)))
        builder.add_postprocessor(LineValueSamplerConfig(
            name="pressure_x_profile",
            variable="pp stress_yy",  # Can sample multiple variables
            start_point=(0, 250, 0),
            end_point=(1000, 250, 0),
            num_points=101,
            output_vector=True  # Ensure this is compatible with VectorPostprocessors
        ))

        # 12. Define Executioner, Preconditioning, and Outputs
        builder.add_executioner_block(end_time=3600, dt=100)
        builder.add_preconditioning_block(active_preconditioner='mumps')
        builder.add_outputs_block(exodus=True, csv=True)

        # 13. Generate the final input file
        builder.generate_input_file(output_filepath)

if __name__ == '__main__':
    # Preserved from original file
    import os
    output_dir = "test_files/moose_input_file_test"
    os.makedirs(output_dir, exist_ok=True)
    full_example_file = os.path.join(output_dir, "model_builder_full_example.i")
    ModelBuilder.build_example_with_all_features(full_example_file)