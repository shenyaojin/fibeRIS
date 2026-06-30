# Characterization tests for fiberis.moose.model_builder.ModelBuilder
#
# Golden-master / snapshot tests that build representative MOOSE models
# programmatically, generate the resulting block structure / input text, and pin
# the CURRENT output. All golden values were obtained by RUNNING the code, never
# hand-written. These tests must stay GREEN through a pure refactor of the
# input-generation layer (they do NOT run the MOOSE binary).

import numpy as np
import pytest

from fiberis.moose.model_builder import ModelBuilder, OptimizationLayeredModelBuilder
from fiberis.moose.config import (
    ZoneMaterialProperties,
    InitialConditionConfig,
    MatrixConfig,
    SRVConfig,
    HydraulicFractureConfig,
    CasingLayerConfig,
    CasingConfig,
    SimpleFluidPropertiesConfig,
    PointValueSamplerConfig,
    LineValueSamplerConfig,
    PostprocessorConfig,
    AdaptivityConfig,
    IndicatorConfig,
    MarkerConfig,
    AdaptiveTimeStepperConfig,
    TimeStepperFunctionConfig,
    TimeSequenceStepper,
)
from fiberis.analyzer.Data1D import core1D


# Helper to fetch a top-level block's rendered text by name.
def _render_block(builder, name):
    return builder._get_or_create_toplevel_moose_block(name).render()


# --- Construction & basic structure ---------------------------------------

def test_init_state():
    b = ModelBuilder("MyProject")
    assert b.project_name == "MyProject"
    assert b._top_level_blocks == []
    assert b.matrix_config is None
    assert b.srv_configs == []
    assert b.fracture_configs == []
    assert b._next_available_block_id == 1


def test_str_representation():
    b = ModelBuilder("P")
    b.add_variables(["pp"])
    s = str(b)
    assert "--- MOOSE Model Structure for: P ---" in s
    assert "[Variables]" in s
    assert "[pp]" in s
    assert "-----------------------------" in s


def test_get_or_create_toplevel_block_is_idempotent():
    b = ModelBuilder("P")
    blk1 = b._get_or_create_toplevel_moose_block("Mesh")
    blk2 = b._get_or_create_toplevel_moose_block("Mesh")
    assert blk1 is blk2
    assert len(b._top_level_blocks) == 1


def test_generate_unique_op_name():
    assert ModelBuilder._generate_unique_op_name("foo", []) == "foo"
    assert ModelBuilder._generate_unique_op_name("foo", ["foo"]) == "foo_1"
    assert ModelBuilder._generate_unique_op_name("foo", ["foo", "foo_1"]) == "foo_2"


# --- GlobalParams & Variables ---------------------------------------------

def test_add_global_params():
    b = ModelBuilder("P")
    b.add_global_params({"displacements": "disp_x disp_y", "PorousFlowDictator": "dictator"})
    rendered = _render_block(b, "GlobalParams")
    assert "[GlobalParams]" in rendered
    assert "displacements = 'disp_x disp_y'" in rendered
    assert "PorousFlowDictator = 'dictator'" in rendered


def test_add_variables_strings_and_dicts():
    b = ModelBuilder("P")
    b.add_variables([
        {"name": "pp", "params": {"initial_condition": 26.4E6}},
        "disp_x",
        "disp_y",
    ])
    rendered = _render_block(b, "Variables")
    assert "[pp]" in rendered
    assert "initial_condition = 26400000.0" in rendered
    assert "[disp_x]" in rendered
    assert "[disp_y]" in rendered


def test_add_variables_dict_without_name_raises():
    b = ModelBuilder("P")
    with pytest.raises(ValueError):
        b.add_variables([{"params": {"x": 1}}])


def test_add_variables_bad_type_raises():
    b = ModelBuilder("P")
    with pytest.raises(TypeError):
        b.add_variables([123])


# --- Mesh: stitched fractures ---------------------------------------------

def test_build_stitched_mesh_single_fracture():
    b = ModelBuilder("P")
    b.build_stitched_mesh_for_fractures(
        fracture_y_coords=250.0,
        domain_bounds=(0, 500),
        domain_length=1000.0,
        nx=50,
        ny_per_layer_half=15,
        bias_y=1.2,
    )
    rendered = _render_block(b, "Mesh")
    # Two half-panels stitched into one layer for a single interior seam.
    assert "[layer0_panel_a]" in rendered
    assert "[layer0_panel_b]" in rendered
    assert "type = StitchedMeshGenerator" in rendered
    assert "[stitched_layer_0]" in rendered
    assert "inputs = 'layer0_panel_a layer0_panel_b'" in rendered
    assert "stitch_boundaries_pairs = 'bottom top'" in rendered
    # Panel A uses bias 1/1.2; panel B uses 1.2.
    assert "bias_y = 0.8333333333333334" in rendered
    assert "bias_y = 1.2" in rendered
    # Geometry info recorded.
    assert b.geometry_info["mesh"]["domain_bounds"] == (0, 500)
    assert b.geometry_info["mesh"]["domain_length"] == 1000.0


def test_build_stitched_mesh_multiple_fractures_final_stitch():
    b = ModelBuilder("P")
    b.build_stitched_mesh_for_fractures(
        fracture_y_coords=[150.0, 350.0],
        domain_bounds=(0, 500),
        domain_length=1000.0,
    )
    rendered = _render_block(b, "Mesh")
    # 3 layers between 4 y-points -> final stitching of layers.
    assert "[stitched_layer_0]" in rendered
    assert "[stitched_layer_2]" in rendered
    assert "[final_stitch_0]" in rendered
    assert "clear_stitched_boundary_ids = true" in rendered


# --- Subdomains & block renaming ------------------------------------------

def test_add_srv_and_fracture_subdomains_and_block_map():
    b = ModelBuilder("P")
    mats = ZoneMaterialProperties(porosity=0.1, permeability=1e-13)
    srv = SRVConfig(name="SRV1", length=300, height=80, center_x=500,
                    center_y=250, materials=mats)
    frac = HydraulicFractureConfig(name="Frac1", length=200, height=0.2,
                                   center_x=500, center_y=250, materials=mats)
    b.add_srv_zone_2d(srv, target_block_id=1)
    b.add_hydraulic_fracture_2d(frac, target_block_id=2)
    rendered = _render_block(b, "Mesh")
    assert "[SRV1_bbox]" in rendered
    assert "type = SubdomainBoundingBoxGenerator" in rendered
    assert "block_id = 1" in rendered
    assert "[Frac1_bbox]" in rendered
    assert "block_id = 2" in rendered
    # SRV: center 500,250 length 300 height 80 -> bottom_left 350,210
    assert "bottom_left = '350.0 210.0 0'" in rendered
    assert "top_right = '650.0 290.0 0'" in rendered
    assert b._block_id_to_name_map == {1: "SRV1", 2: "Frac1"}
    assert b._next_available_block_id == 3


def test_finalize_mesh_block_renaming_adds_matrix_zero():
    b = ModelBuilder("P")
    b.set_matrix_config(MatrixConfig(name="matrix",
                        materials=ZoneMaterialProperties(porosity=0.05, permeability=1e-16)))
    mats = ZoneMaterialProperties(porosity=0.1, permeability=1e-13)
    b.add_srv_zone_2d(SRVConfig(name="SRV1", length=10, height=10, center_x=5,
                                center_y=5, materials=mats), target_block_id=1)
    b._finalize_mesh_block_renaming()
    rendered = _render_block(b, "Mesh")
    assert "type = RenameBlockGenerator" in rendered
    assert "old_block = '0 1'" in rendered
    assert "new_block = 'matrix SRV1'" in rendered


def test_refine_blocks():
    b = ModelBuilder("P")
    b.refine_blocks(op_name="refine", block_ids=[1, 2], refinement_levels=[1, 2])
    rendered = _render_block(b, "Mesh")
    assert "type = RefineBlockGenerator" in rendered
    assert "block = '1 2'" in rendered
    assert "refinement = '1 2'" in rendered


def test_refine_blocks_scalar_level_broadcast():
    b = ModelBuilder("P")
    b.refine_blocks(op_name="refine", block_ids=[1, 2, 3], refinement_levels=2)
    rendered = _render_block(b, "Mesh")
    assert "refinement = '2 2 2'" in rendered


def test_refine_blocks_mismatched_lengths_raises():
    b = ModelBuilder("P")
    with pytest.raises(ValueError):
        b.refine_blocks(op_name="r", block_ids=[1, 2], refinement_levels=[1])


# --- Boundaries & nodesets -------------------------------------------------

def test_add_named_boundary():
    b = ModelBuilder("P")
    b.add_named_boundary("left", (0, 0, 0), (0, 500, 0))
    rendered = _render_block(b, "Mesh")
    assert "[sideset_left]" in rendered
    assert "type = SideSetsAroundBoundingBoxGenerator" in rendered
    assert "boundary_names = 'left'" in rendered
    assert "bottom_left = '0 0 0'" in rendered
    assert {"name": "left", "start": (0, 0, 0), "end": (0, 500, 0)} in b.geometry_info["boundaries"]


def test_add_nodeset_by_coord():
    b = ModelBuilder("P")
    b.add_nodeset_by_coord("well_nodes", "injection_well", (500, 250, 0))
    rendered = _render_block(b, "Mesh")
    assert "[well_nodes]" in rendered
    assert "type = ExtraNodesetGenerator" in rendered
    assert "new_boundary = 'injection_well'" in rendered
    assert "coord = '500 250 0'" in rendered


def test_add_nodeset_by_bbox():
    b = ModelBuilder("P")
    b.add_nodeset_by_bbox("ns", "bnd", (0, 0, 0), (1, 1, 0))
    rendered = _render_block(b, "Mesh")
    assert "type = BoundingBoxNodeSetGenerator" in rendered
    assert "new_boundary = 'bnd'" in rendered
    assert "bottom_left = '0 0 0'" in rendered


def test_add_linear_pressure_boundary_creates_nodeset_and_bc():
    b = ModelBuilder("P")
    b.add_linear_pressure_boundary(
        boundary_name="inj", bottom_left=(0, 0, 0), top_right=(1, 1, 0),
        pressure_function_name="pfunc")
    mesh = _render_block(b, "Mesh")
    bcs = _render_block(b, "BCs")
    assert "[inj_generator]" in mesh
    assert "[inj_pressure_bc]" in bcs
    assert "type = FunctionDirichletBC" in bcs
    assert "function = 'pfunc'" in bcs


# --- Kernels ---------------------------------------------------------------

def test_add_time_derivative_kernel_default_name():
    b = ModelBuilder("P")
    b.add_time_derivative_kernel(variable="pp")
    rendered = _render_block(b, "Kernels")
    assert "[dot_pp]" in rendered
    assert "type = TimeDerivative" in rendered
    assert "variable = 'pp'" in rendered


def test_add_function_diffusion_kernel_block_list_joined():
    b = ModelBuilder("P")
    b.add_function_diffusion_kernel("diff", "pp", "kfunc", ["matrix", "SRV1"])
    rendered = _render_block(b, "Kernels")
    assert "type = FunctionDiffusion" in rendered
    assert "function = 'kfunc'" in rendered
    assert "block = 'matrix SRV1'" in rendered


def test_add_anisotropic_diffusion_kernel():
    b = ModelBuilder("P")
    b.add_anisotropic_diffusion_kernel("aniso", "pp", "matrix", "1 0 0 0 1 0 0 0 1")
    rendered = _render_block(b, "Kernels")
    assert "type = AnisotropicDiffusion" in rendered
    assert "tensor_coeff = '1 0 0 0 1 0 0 0 1'" in rendered


def test_add_porous_flow_darcy_base_kernel():
    b = ModelBuilder("P")
    b.add_porous_flow_darcy_base_kernel("flux", "pp")
    rendered = _render_block(b, "Kernels")
    assert "type = PorousFlowFullySaturatedDarcyBase" in rendered
    assert "gravity = '0 0 0'" in rendered


def test_add_stress_divergence_tensor_kernel():
    b = ModelBuilder("P")
    b.add_stress_divergence_tensor_kernel("grad_x", "disp_x", 0)
    rendered = _render_block(b, "Kernels")
    assert "type = StressDivergenceTensors" in rendered
    assert "component = 0" in rendered


def test_add_effective_stress_coupling_kernel():
    b = ModelBuilder("P")
    b.add_porous_flow_effective_stress_coupling_kernel("eff_x", "disp_x", 0, 0.7)
    rendered = _render_block(b, "Kernels")
    assert "type = PorousFlowEffectiveStressCoupling" in rendered
    assert "biot_coefficient = 0.7" in rendered


def test_add_mass_volumetric_expansion_and_time_derivative_kernels():
    b = ModelBuilder("P")
    b.add_porous_flow_mass_volumetric_expansion_kernel("mass_exp", "pp")
    b.add_porous_flow_mass_time_derivative_kernel("mass_dt", "pp")
    rendered = _render_block(b, "Kernels")
    assert "type = PorousFlowMassVolumetricExpansion" in rendered
    assert "type = PorousFlowMassTimeDerivative" in rendered
    assert "fluid_component = 0" in rendered


def test_add_custom_kernel():
    b = ModelBuilder("P")
    b.add_custom_kernel(kernel_type="MyKernel", kernel_name="kk", variable="pp",
                        params={"foo": "bar"}, extra=5)
    rendered = _render_block(b, "Kernels")
    assert "[kk]" in rendered
    assert "type = MyKernel" in rendered
    assert "foo = 'bar'" in rendered
    assert "extra = 5" in rendered


# --- UserObjects / dictator ------------------------------------------------

def test_add_user_object_replaces_same_name():
    b = ModelBuilder("P")
    b.add_user_object("uo", "TypeA", {"x": 1})
    b.add_user_object("uo", "TypeB", {"y": 2})
    rendered = _render_block(b, "UserObjects")
    assert "type = TypeB" in rendered
    assert "type = TypeA" not in rendered
    # Only one sub-block remains.
    assert b._get_or_create_toplevel_moose_block("UserObjects").sub_blocks.__len__() == 1


def test_set_porous_flow_dictator():
    b = ModelBuilder("P")
    b.set_porous_flow_dictator(porous_flow_variables=["pp", "disp_x"],
                               num_fluid_phases=1, num_fluid_components=1)
    rendered = _render_block(b, "UserObjects")
    assert "type = PorousFlowDictator" in rendered
    assert "porous_flow_vars = 'pp disp_x'" in rendered
    assert "number_fluid_phases = 1" in rendered


# --- Boundary conditions ---------------------------------------------------

def test_add_boundary_condition_with_list_boundary():
    b = ModelBuilder("P")
    b.add_boundary_condition("confinex", "DirichletBC", "disp_x",
                             ["left", "right"], params={"value": 0})
    rendered = _render_block(b, "BCs")
    assert "type = DirichletBC" in rendered
    assert "boundary = 'left right'" in rendered
    assert "value = 0" in rendered


def test_set_hydraulic_fracturing_bcs_clears_and_sets():
    b = ModelBuilder("P")
    b.add_boundary_condition("stale", "DirichletBC", "x", "b")
    b.set_hydraulic_fracturing_bcs(
        injection_well_boundary_name="injection_well",
        injection_pressure_function_name="pfunc",
        confine_disp_x_boundaries="left right",
        confine_disp_y_boundaries="top bottom")
    rendered = _render_block(b, "BCs")
    # Old BC cleared.
    assert "[stale]" not in rendered
    assert "[injection_pressure]" in rendered
    assert "function = 'pfunc'" in rendered
    assert "[confinex]" in rendered
    assert "[confiney]" in rendered


# --- Functions -------------------------------------------------------------

def test_add_piecewise_function_from_data1d():
    b = ModelBuilder("P")
    d = core1D.Data1D(taxis=np.array([0, 1800, 3600]),
                      data=np.array([27e6, 45e6, 40e6]))
    b.add_piecewise_function_from_data1d("inj_func", d)
    rendered = _render_block(b, "Functions")
    assert "[inj_func]" in rendered
    assert "type = PiecewiseLinear" in rendered
    assert "x = '0 1800 3600'" in rendered
    assert "y = '27000000.0 45000000.0 40000000.0'" in rendered


def test_add_piecewise_function_wrong_type_raises():
    b = ModelBuilder("P")
    with pytest.raises(TypeError):
        b.add_piecewise_function_from_data1d("f", "not a data1d")


def test_add_piecewise_function_empty_data1d_raises():
    b = ModelBuilder("P")
    d = core1D.Data1D()
    with pytest.raises(ValueError):
        b.add_piecewise_function_from_data1d("f", d)


# --- Fluid properties & Materials -----------------------------------------

def test_add_simple_fluid_properties():
    b = ModelBuilder("P")
    b.add_simple_fluid_properties(SimpleFluidPropertiesConfig(name="water"))
    rendered = _render_block(b, "FluidProperties")
    assert "[water]" in rendered
    assert "type = SimpleFluidProperties" in rendered
    assert "bulk_modulus = 2200000000.0" in rendered
    assert "viscosity = 0.001" in rendered
    assert "density0 = 1000.0" in rendered


def test_add_simple_fluid_properties_wrong_type_raises():
    b = ModelBuilder("P")
    with pytest.raises(TypeError):
        b.add_simple_fluid_properties("not a config")


def test_add_poromechanics_materials_fracture_workflow():
    b = ModelBuilder("P")
    matrix_mats = ZoneMaterialProperties(porosity=0.05, permeability="'1e-15 0 0 0 1e-15 0 0 0 1e-16'",
                                         youngs_modulus=3e10, poissons_ratio=0.25)
    srv_mats = ZoneMaterialProperties(porosity=0.1, permeability="'1e-13 0 0'")
    b.set_matrix_config(MatrixConfig(name="matrix", materials=matrix_mats))
    b.add_srv_config(SRVConfig(name="SRV1", length=10, height=10, center_x=5,
                               center_y=5, materials=srv_mats))
    b.add_fluid_properties_config(SimpleFluidPropertiesConfig(name="water"))
    b.add_poromechanics_materials(fluid_properties_name="water",
                                  biot_coefficient=0.7, solid_bulk_compliance=1e-11)
    rendered = _render_block(b, "Materials")
    assert "[porosity_matrix]" in rendered
    assert "type = PorousFlowPorosityConst" in rendered
    assert "[permeability_matrix]" in rendered
    assert "[elasticity_tensor_matrix]" in rendered
    assert "youngs_modulus = 30000000000.0" in rendered
    assert "poissons_ratio = 0.25" in rendered
    assert "[biot_modulus]" in rendered
    assert "type = PorousFlowConstantBiotModulus" in rendered
    assert "fluid_bulk_modulus = 2200000000.0" in rendered
    assert "block = 'matrix SRV1'" in rendered
    assert "type = ComputeLinearElasticStress" in rendered
    assert "type = PorousFlowVolumetricStrain" in rendered


def test_add_poromechanics_materials_missing_fluid_raises():
    b = ModelBuilder("P")
    b.set_matrix_config(MatrixConfig(name="matrix",
                        materials=ZoneMaterialProperties(porosity=0.05, permeability=1e-16)))
    with pytest.raises(ValueError):
        b.add_poromechanics_materials(fluid_properties_name="nope",
                                      biot_coefficient=0.7, solid_bulk_compliance=1e-11)


def test_add_poromechanics_materials_npz_permeability_path(tmp_path):
    # A permeability given as a path to an .npz triggers a time-dependent
    # FunctionAux + PorousFlowPermeabilityTensorFromVar workflow.
    npz = tmp_path / "perm.npz"
    np.savez(npz, taxis=np.array([0., 1., 2.]),
             data=np.array([1e-13, 2e-13, 3e-13]),
             start_time="2020-01-01T00:00:00")
    mats = ZoneMaterialProperties(porosity=0.1, permeability=str(npz))
    b = ModelBuilder("P")
    b.set_matrix_config(MatrixConfig(name="matrix", materials=mats))
    b.add_fluid_properties_config(SimpleFluidPropertiesConfig(name="water"))
    b.add_poromechanics_materials(fluid_properties_name="water",
                                  biot_coefficient=0.7, solid_bulk_compliance=1e-11)
    materials = _render_block(b, "Materials")
    aux_vars = _render_block(b, "AuxVariables")
    aux_kernels = _render_block(b, "AuxKernels")
    functions = _render_block(b, "Functions")
    assert "type = PorousFlowPermeabilityTensorFromVar" in materials
    assert "perm = 'scalar_perm_matrix'" in materials
    assert "[scalar_perm_matrix]" in aux_vars
    assert "type = FunctionAux" in aux_kernels
    assert "[perm_func_matrix]" in functions


def test_set_main_domain_parameters_2d_deprecated(capsys):
    b = ModelBuilder("P")
    result = b.set_main_domain_parameters_2d(foo=1)
    assert result is b
    assert "deprecated" in capsys.readouterr().out


def test_set_adaptivity_options_fallback_no_config():
    b = ModelBuilder("P")
    b.set_adaptivity_options(enable=True)
    rendered = _render_block(b, "Adaptivity")
    assert "marker = 'default_marker'" in rendered
    assert "[default_indicator]" in rendered
    assert "refine = 0.5" in rendered


def test_set_adaptivity_options_disable_when_absent_is_noop():
    b = ModelBuilder("P")
    result = b.set_adaptivity_options(enable=False)
    assert result is b
    assert b._top_level_blocks == []


# --- AuxVariables / AuxKernels --------------------------------------------

def test_add_standard_tensor_aux_vars_and_kernels():
    b = ModelBuilder("P")
    b.add_standard_tensor_aux_vars_and_kernels({"stress": "stress"})
    aux_vars = _render_block(b, "AuxVariables")
    aux_kernels = _render_block(b, "AuxKernels")
    assert "[stress_xx]" in aux_vars
    assert "[stress_yy]" in aux_vars
    assert "order = 'CONSTANT'" in aux_vars
    assert "family = 'MONOMIAL'" in aux_vars
    assert "type = RankTwoAux" in aux_kernels
    assert "rank_two_tensor = 'stress'" in aux_kernels
    assert "index_i = 0" in aux_kernels


def test_add_standard_tensor_aux_rate_variant():
    b = ModelBuilder("P")
    b.add_standard_tensor_aux_vars_and_kernels({"strain": "strain", "strain_rate": "strain_rate"})
    aux_kernels = _render_block(b, "AuxKernels")
    assert "type = TimeDerivativeAux" in aux_kernels
    assert "[strain_rate_xx]" in aux_kernels
    assert "functor = 'strain_xx'" in aux_kernels


def test_add_standard_tensor_aux_rate_missing_base_raises():
    b = ModelBuilder("P")
    with pytest.raises(ValueError):
        b.add_standard_tensor_aux_vars_and_kernels({"strain_rate": "strain_rate"})


# --- Initial conditions ----------------------------------------------------

def test_add_initial_conditions_from_configs():
    b = ModelBuilder("P")
    ic = InitialConditionConfig(name="ppic", ic_type="ConstantIC",
                                variable="pp", params={"value": 1.0})
    b.set_matrix_config(MatrixConfig(name="matrix",
                        materials=ZoneMaterialProperties(porosity=0.05, permeability=1e-16),
                        initial_conditions=[ic]))
    b.add_initial_conditions_from_configs()
    rendered = _render_block(b, "ICs")
    assert "[ppic_matrix]" in rendered
    assert "type = ConstantIC" in rendered
    assert "variable = 'pp'" in rendered
    assert "block = 'matrix'" in rendered
    assert "value = 1.0" in rendered


# --- Adaptivity ------------------------------------------------------------

def test_set_adaptivity_options_with_config():
    b = ModelBuilder("P")
    cfg = AdaptivityConfig(
        marker_to_use="mymarker", steps=3,
        indicators=[IndicatorConfig("ind", "GradientJumpIndicator", {"variable": "pp"})],
        markers=[MarkerConfig("mymarker", "ErrorFractionMarker",
                              {"indicator": "ind", "refine": 0.3})])
    b.set_adaptivity_options(enable=True, config=cfg)
    rendered = _render_block(b, "Adaptivity")
    assert "marker = 'mymarker'" in rendered
    assert "steps = 3" in rendered
    assert "[Indicators]" in rendered
    assert "type = GradientJumpIndicator" in rendered
    assert "[Markers]" in rendered
    assert "type = ErrorFractionMarker" in rendered


def test_set_adaptivity_options_default_template():
    b = ModelBuilder("P")
    b.set_adaptivity_options(enable=True,
                             default_template_settings={"monitored_variable": "pp"})
    rendered = _render_block(b, "Adaptivity")
    assert "marker = 'marker_for_pp'" in rendered
    assert "[indicator_on_pp]" in rendered
    assert "refine = 0.3" in rendered
    assert "coarsen = 0.05" in rendered


def test_set_adaptivity_options_disable_removes_block():
    b = ModelBuilder("P")
    b.set_adaptivity_options(enable=True, default_template_settings={})
    b.set_adaptivity_options(enable=False)
    assert all(blk.block_name != "Adaptivity" for blk in b._top_level_blocks)


def test_set_adaptivity_options_wrong_config_type_raises():
    b = ModelBuilder("P")
    with pytest.raises(TypeError):
        b.set_adaptivity_options(enable=True, config="not a config")


# --- Postprocessors --------------------------------------------------------

def test_add_point_value_postprocessor():
    b = ModelBuilder("P")
    b.add_postprocessor(PointValueSamplerConfig(name="pp_well", variable="pp",
                                                point=(500, 250, 0)))
    rendered = _render_block(b, "Postprocessors")
    assert "[pp_well]" in rendered
    assert "type = PointValue" in rendered
    assert "variable = 'pp'" in rendered
    assert "point = '500 250 0'" in rendered


def test_add_line_value_postprocessor_goes_to_vector_block():
    b = ModelBuilder("P")
    b.add_postprocessor(LineValueSamplerConfig(
        name="profile", variable="pp stress_yy",
        start_point=(0, 250, 0), end_point=(1000, 250, 0),
        num_points=101, output_vector=True))
    rendered = _render_block(b, "VectorPostprocessors")
    assert "[profile]" in rendered
    assert "type = LineValueSampler" in rendered
    assert "start_point = '0 250 0'" in rendered
    assert "num_points = 101" in rendered


def test_add_generic_postprocessor():
    b = ModelBuilder("P")
    b.add_postprocessor(PostprocessorConfig(name="avg", pp_type="ElementAverageValue",
                                            params={"variable": "pp"}))
    rendered = _render_block(b, "Postprocessors")
    assert "[avg]" in rendered
    assert "type = ElementAverageValue" in rendered


def test_add_postprocessor_wrong_type_raises():
    b = ModelBuilder("P")
    with pytest.raises(TypeError):
        b.add_postprocessor("not a config")


# --- Executioner / Preconditioning / Outputs ------------------------------

def test_add_executioner_block_constant_dt():
    b = ModelBuilder("P")
    b.add_executioner_block(end_time=3600, time_stepper_type="ConstantDT", dt=100)
    rendered = _render_block(b, "Executioner")
    assert "type = 'Transient'" in rendered
    assert "solve_type = 'Newton'" in rendered
    assert "end_time = 3600" in rendered
    assert "[TimeStepper]" in rendered
    assert "type = ConstantDT" in rendered
    assert "dt = 100" in rendered


def test_add_executioner_block_constant_dt_missing_dt_raises():
    b = ModelBuilder("P")
    with pytest.raises(ValueError):
        b.add_executioner_block(end_time=10, time_stepper_type="ConstantDT")


def test_add_executioner_block_time_sequence_stepper():
    b = ModelBuilder("P")
    b.add_executioner_block(end_time=4, time_stepper_type="TimeSequenceStepper",
                            stepper_config=TimeSequenceStepper([0, 1, 2, 4]))
    rendered = _render_block(b, "Executioner")
    assert "type = TimeSequenceStepper" in rendered
    assert "time_sequence = '0 1 2 4'" in rendered


def test_add_executioner_block_iteration_adaptive_dt():
    b = ModelBuilder("P")
    func = TimeStepperFunctionConfig(name="sched", x_values=[0, 100], y_values=[10, 10])
    cfg = AdaptiveTimeStepperConfig(functions=[func])
    b.add_executioner_block(end_time=100, time_stepper_type="IterationAdaptiveDT",
                            dt=10, stepper_config=cfg)
    exec_rendered = _render_block(b, "Executioner")
    func_rendered = _render_block(b, "Functions")
    assert "type = IterationAdaptiveDT" in exec_rendered
    assert "force_step_every_function_point = true" in exec_rendered
    assert "timestep_limiting_function = 'sched'" in exec_rendered
    # The schedule function is added to [Functions].
    assert "[sched]" in func_rendered


def test_add_executioner_block_adaptive_wrong_config_raises():
    b = ModelBuilder("P")
    with pytest.raises(TypeError):
        b.add_executioner_block(end_time=10, time_stepper_type="IterationAdaptiveDT",
                                dt=1, stepper_config="bad")


def test_add_preconditioning_block():
    b = ModelBuilder("P")
    b.add_preconditioning_block(active_preconditioner="mumps")
    rendered = _render_block(b, "Preconditioning")
    assert "active = 'mumps'" in rendered
    assert "[mumps]" in rendered
    assert "type = SMP" in rendered
    assert "[basic]" in rendered


def test_add_outputs_block_default():
    b = ModelBuilder("P")
    b.add_outputs_block(exodus=True, csv=True)
    rendered = _render_block(b, "Outputs")
    assert "[exodus]" in rendered
    assert "type = Exodus" in rendered
    assert "[csv]" in rendered
    assert "type = CSV" in rendered


def test_add_outputs_block_exodus_execute_on_and_replace():
    b = ModelBuilder("P")
    b.add_outputs_block(exodus=True, csv=False)
    b.add_outputs_block(exodus=True, csv=True, exodus_execute_on="FINAL")
    rendered = _render_block(b, "Outputs")
    assert "execute_on = 'FINAL'" in rendered
    assert "[csv]" in rendered
    # Only a single exodus block exists after replacement.
    outputs = b._get_or_create_toplevel_moose_block("Outputs")
    exodus_blocks = [sb for sb in outputs.sub_blocks if sb.block_name == "exodus"]
    assert len(exodus_blocks) == 1


# --- Casing model ----------------------------------------------------------

def _casing_builder():
    b = ModelBuilder("Cas")
    mats1 = ZoneMaterialProperties(porosity=0.1, permeability=1e-15,
                                   youngs_modulus=3e10, poissons_ratio=0.25)
    mats2 = ZoneMaterialProperties(porosity=0.2, permeability=1e-13,
                                   youngs_modulus=2e10, poissons_ratio=0.2)
    layers = [CasingLayerConfig("top", 10.0, mats1),
              CasingLayerConfig("bot", 20.0, mats2)]
    cc = CasingConfig("casing", layers, injection_well_name="well",
                      injection_well_x_coord=50.0)
    b.add_casing_config(cc)
    return b


def test_build_mesh_for_casing_model():
    b = _casing_builder()
    b.build_mesh_for_casing_model(domain_length=100.0, nx=10, ny=10)
    rendered = _render_block(b, "Mesh")
    assert "[base_mesh]" in rendered
    # Total height 30 -> ymin/ymax -15/15.
    assert "ymin = -15.0" in rendered
    assert "ymax = 15.0" in rendered
    assert "[top_bbox]" in rendered
    assert "[bot_bbox]" in rendered
    assert "[injection_well_nodes]" in rendered
    assert "new_boundary = 'well'" in rendered
    assert "[final_block_rename]" in rendered
    assert "new_block = 'top bot'" in rendered
    assert b._block_id_to_name_map == {1: "top", 2: "bot"}
    assert b.geometry_info["mesh"]["type"] == "casing_model_srv_layers"


def test_build_mesh_for_casing_model_without_config_raises():
    b = ModelBuilder("P")
    with pytest.raises(ValueError):
        b.build_mesh_for_casing_model(domain_length=100.0, nx=10, ny=10)


def test_add_poromechanics_materials_casing_workflow():
    b = _casing_builder()
    b.build_mesh_for_casing_model(domain_length=100.0, nx=10, ny=10)
    b.add_fluid_properties_config(SimpleFluidPropertiesConfig(name="water"))
    b.add_poromechanics_materials(fluid_properties_name="water",
                                  biot_coefficient=0.7, solid_bulk_compliance=1e-11)
    rendered = _render_block(b, "Materials")
    # Per-layer materials.
    assert "[porosity_top]" in rendered
    assert "[permeability_top]" in rendered
    assert "[elasticity_tensor_top]" in rendered
    assert "[porosity_bot]" in rendered
    assert "block = 'top bot'" in rendered


# --- save/load round trip & full input generation -------------------------

def test_save_and_load_npz_roundtrip(tmp_path):
    b = ModelBuilder("RoundTrip")
    b.add_variables(["pp"])
    b.add_time_derivative_kernel(variable="pp")
    fp = tmp_path / "state.npz"
    b.save_npz(str(fp))
    assert fp.exists()
    loaded = ModelBuilder.load_npz(str(fp))
    assert isinstance(loaded, ModelBuilder)
    assert loaded.project_name == "RoundTrip"
    assert str(loaded) == str(b)


def test_save_npz_appends_extension(tmp_path):
    b = ModelBuilder("P")
    b.add_variables(["pp"])
    target = tmp_path / "noext"
    b.save_npz(str(target))
    assert (tmp_path / "noext.npz").exists()


def test_generate_input_file_empty_raises(tmp_path):
    b = ModelBuilder("Empty")
    with pytest.raises(ValueError):
        b.generate_input_file(str(tmp_path / "empty.i"))


def test_generate_input_file_full(tmp_path):
    b = ModelBuilder("Full")
    b.add_variables(["pp", "disp_x"])
    b.add_time_derivative_kernel(variable="pp")
    b.add_executioner_block(end_time=10, time_stepper_type="ConstantDT", dt=1)
    out = tmp_path / "full.i"
    b.generate_input_file(str(out))
    text = out.read_text()
    assert text.startswith("# MOOSE input file generated by ModelBuilder for project: Full")
    assert "[Variables]" in text
    assert "[Kernels]" in text
    assert "[Executioner]" in text
    # Each top-level block is closed with [].
    assert "[]" in text


def test_build_example_with_all_features(tmp_path):
    # End-to-end golden master of the documented example workflow.
    out = tmp_path / "example.i"
    ModelBuilder.build_example_with_all_features(str(out))
    assert out.exists()
    text = out.read_text()
    assert "project: StitchedMeshFracExample" in text
    assert "[Mesh]" in text
    assert "type = StitchedMeshGenerator" in text
    assert "[Variables]" in text
    assert "[Kernels]" in text
    assert "[Materials]" in text
    assert "[BCs]" in text
    assert "[Functions]" in text
    assert "[Postprocessors]" in text
    assert "[VectorPostprocessors]" in text
    assert "[Executioner]" in text
    assert "[Preconditioning]" in text
    assert "[Outputs]" in text
    assert "type = RenameBlockGenerator" in text


def test_extract_geometry():
    b = ModelBuilder("P")
    mats = ZoneMaterialProperties(porosity=0.1, permeability=1e-13)
    b.build_stitched_mesh_for_fractures(250.0, (0, 500), 1000.0)
    b.add_srv_zone_2d(SRVConfig(name="SRV1", length=10, height=10, center_x=5,
                                center_y=5, materials=mats), target_block_id=1)
    geo = b.extract_geometry()
    assert "mesh" in geo
    assert geo["mesh"]["domain_length"] == 1000.0
    assert len(geo["srv_zones"]) == 1


# --- OptimizationLayeredModelBuilder --------------------------------------

def _opt_builder():
    b = OptimizationLayeredModelBuilder("Opt")
    mats = ZoneMaterialProperties(porosity=0.1, permeability=1e-15,
                                  youngs_modulus=3e10, poissons_ratio=0.25)
    layers = [CasingLayerConfig("l1", 10.0, mats),
              CasingLayerConfig("l2", 10.0, mats)]
    cc = CasingConfig("casing", layers, injection_well_name="well",
                      injection_well_x_coord=50.0)
    b.set_casing_config(cc)
    return b


def test_optimization_builder_set_casing_config():
    b = _opt_builder()
    assert b.casing_config is not None
    assert len(b.casing_config.layers) == 2


def test_optimization_builder_adjoint_variables():
    b = _opt_builder()
    b.add_adjoint_variables()
    rendered = _render_block(b, "Variables")
    assert "[pp_adjoint]" in rendered
    assert "solver_sys = 'adjoint'" in rendered
    assert "[disp_x_adjoint]" in rendered


def test_optimization_setup_forward_model():
    b = _opt_builder()
    b.setup_optimization_forward_model(perm_y=1e-20, perm_z=0.0)
    funcs = _render_block(b, "Functions")
    aux_kernels = _render_block(b, "AuxKernels")
    vpps = _render_block(b, "VectorPostprocessors")
    assert "[func_kyy]" in funcs
    assert "type = ParsedFunction" in funcs
    assert "[perm_l1]" in funcs
    assert "type = ParsedOptimizationFunction" in funcs
    assert "type = FunctionAux" in aux_kernels
    assert "type = ElementOptimizationAnisotropicDiffusionInnerProduct" in vpps


def test_optimization_setup_forward_model_without_config_raises():
    b = OptimizationLayeredModelBuilder("Opt")
    with pytest.raises(ValueError):
        b.setup_optimization_forward_model()


def test_optimization_executioner_block():
    b = _opt_builder()
    b.add_optimization_executioner_block(end_time=10, time_stepper_type="ConstantDT", dt=1)
    rendered = _render_block(b, "Executioner")
    assert "type = 'TransientAndAdjoint'" in rendered
    assert "forward_system = 'nl0'" in rendered
    assert "adjoint_system = 'adjoint'" in rendered


def test_optimization_preconditioning_block():
    b = _opt_builder()
    b.add_preconditioning_block()
    rendered = _render_block(b, "Preconditioning")
    assert "[nl0]" in rendered
    assert "nl_sys = 'nl0'" in rendered
    assert "[adjoint]" in rendered
    assert "nl_sys = 'adjoint'" in rendered


def test_optimization_problem_block():
    b = _opt_builder()
    b.add_optimization_problem_block()
    rendered = _render_block(b, "Problem")
    assert "nl_sys_names = 'nl0 adjoint'" in rendered
    assert "kernel_coverage_check = false" in rendered


def test_optimization_reporters_and_dirac():
    b = _opt_builder()
    b.add_optimization_reporters_and_dirac(measurement_variable="disp_y")
    reporters = _render_block(b, "Reporters")
    dirac = _render_block(b, "DiracKernels")
    assert "[data]" in reporters
    assert "type = OptimizationData" in reporters
    assert "[params]" in reporters
    assert "real_vector_names = 'perm_1 perm_2'" in reporters
    assert "[misfit]" in dirac
    assert "variable = 'disp_y_adjoint'" in dirac
