#%% For debug.
import os
import numpy as np
from fiberis.moose.model_builder import ModelBuilder
from fiberis.moose.runner import MooseRunner
from fiberis.moose.config import (
    MatrixConfig, SRVConfig, HydraulicFractureConfig, ZoneMaterialProperties,
    SimpleFluidPropertiesConfig, PointValueSamplerConfig, LineValueSamplerConfig
)
from fiberis.analyzer.Data1D.core1D import Data1D

#%% --- 1. Basic Setup ---
output_dir = "output/moose_full_sim_output_fixed_v14"
os.makedirs(output_dir, exist_ok=True)
input_file = os.path.join(output_dir, "full_multi_frac_sim_fixed.i")

builder = ModelBuilder(project_name="FullMultiFractureSimFixed_v14")

# --- 2. Mesh Generation ---
fracture_y_coords = [300.0, -400.0]
domain_bounds = (-500.0, 500.0)
domain_length = 1000.0

builder.build_stitched_mesh_for_fractures(
    fracture_y_coords=fracture_y_coords,
    domain_bounds=domain_bounds,
    domain_length=domain_length,
    nx=100,
    ny_per_layer_half=20,
    bias_y=1.5
)

# --- 3. Define Materials and Geometry ---
matrix_mats = ZoneMaterialProperties(porosity=0.01, permeability="'1E-20 0 0  0 1E-20 0  0 0 1E-21'")
srv_mats = ZoneMaterialProperties(porosity=0.1, permeability="'1E-17 0 0  0 1E-17 0  0 0 1E-18'")
fracture_mats = ZoneMaterialProperties(porosity=0.1, permeability="'1E-12 0 0  0 1E-12 0  0 0 1E-13'")

builder.set_matrix_config(MatrixConfig(name="matrix", materials=matrix_mats))

center_x_val = domain_length / 2.0
geometries = [
    SRVConfig(name="srv_top", length=300, height=50, center_x=center_x_val, center_y=300, materials=srv_mats),
    HydraulicFractureConfig(name="hf_top", length=250, height=0.2, center_x=center_x_val, center_y=300,
                            materials=fracture_mats),
    SRVConfig(name="srv_bot", length=300, height=50, center_x=center_x_val, center_y=-400, materials=srv_mats),
    HydraulicFractureConfig(name="hf_bot", length=250, height=0.2, center_x=center_x_val, center_y=-400,
                            materials=fracture_mats)
]

sorted_geometries = sorted(geometries, key=lambda x: x.height, reverse=True)
next_block_id = 1
for geom_config in sorted_geometries:
    if isinstance(geom_config, SRVConfig):
        builder.add_srv_config(geom_config)
        builder.add_srv_zone_2d(geom_config, target_block_id=next_block_id)
    elif isinstance(geom_config, HydraulicFractureConfig):
        builder.add_fracture_config(geom_config)
        builder.add_hydraulic_fracture_2d(geom_config, target_block_id=next_block_id)
    next_block_id += 1

builder.add_nodeset_by_coord(nodeset_op_name="injection_top", new_boundary_name="injection_well_top",
                             coordinates=(center_x_val, 300, 0))
builder.add_nodeset_by_coord(nodeset_op_name="injection_bot", new_boundary_name="injection_well_bot",
                             coordinates=(center_x_val, -400, 0))

# --- 4. Add Full Physics Fields ---
builder.add_variables([
    {"name": "pp", "params": {"initial_condition": 2.64E7}},
    {"name": "disp_x", "params": {"initial_condition": 0}},
    {"name": "disp_y", "params": {"initial_condition": 0}}
])

builder.set_porous_flow_dictator(dictator_name="dictator", porous_flow_variables="pp")
builder.add_global_params({"PorousFlowDictator": "dictator", "displacements": "'disp_x disp_y'"})

biot_coeff_val = 0.7
builder.add_time_derivative_kernel(variable="pp")
builder.add_porous_flow_darcy_base_kernel(kernel_name="flux", variable="pp")
builder.add_stress_divergence_tensor_kernel(kernel_name="grad_stress_x", variable="disp_x", component=0)
builder.add_stress_divergence_tensor_kernel(kernel_name="grad_stress_y", variable="disp_y", component=1)
builder.add_porous_flow_effective_stress_coupling_kernel(kernel_name="eff_stress_x", variable="disp_x", component=0,
                                                         biot_coefficient=biot_coeff_val)
builder.add_porous_flow_effective_stress_coupling_kernel(kernel_name="eff_stress_y", variable="disp_y", component=1,
                                                         biot_coefficient=biot_coeff_val)
builder.add_porous_flow_mass_volumetric_expansion_kernel(kernel_name="mass_exp", variable="pp")

fluid_props = SimpleFluidPropertiesConfig(name="water", bulk_modulus=2.2E9, viscosity=1.0E-3, density0=1000.0)
builder.add_fluid_properties_config(fluid_props)
builder.add_poromechanics_materials(
    fluid_properties_name="water",
    biot_coefficient=biot_coeff_val,
    solid_bulk_compliance=2E-11
)

dummy_pressure_curve = Data1D(
    taxis=np.linspace(0, 3600, 100),         # Generates 100 time points from 0 to 3600
    data=np.linspace(2.7E7, 4.0E7, 100)      # Generates 100 data points from 2.7E7 to 4.0E7
)
builder.add_piecewise_function_from_data1d(name="injection_pressure_func", source_data1d=dummy_pressure_curve)

builder.set_hydraulic_fracturing_bcs(
    injection_well_boundary_name="injection_well_top injection_well_bot",
    injection_pressure_function_name="injection_pressure_func",
    confine_disp_x_boundaries="left right",
    confine_disp_y_boundaries="top bottom"
)

builder.add_standard_tensor_aux_vars_and_kernels({"stress": "stress", "total_strain": "strain"})

# Point samplers
builder.add_postprocessor(PointValueSamplerConfig(name="pp_inj_top", variable="pp", point=(center_x_val, 300, 0)))
builder.add_postprocessor(PointValueSamplerConfig(name="pp_inj_bot", variable="pp", point=(center_x_val, -400, 0)))

# vector samplers
builder.add_postprocessor(
    LineValueSamplerConfig(name="pressure_profile_top_hf", variable="pp", start_point=(center_x_val - 125, 300, 0),
                           end_point=(center_x_val + 125, 300, 0), num_points=101,
                           other_params={'sort_by': 'x'}))
builder.add_postprocessor(
    LineValueSamplerConfig(name="pressure_profile_bot_hf", variable="pp", start_point=(center_x_val - 125, -400, 0),
                           end_point=(center_x_val + 125, -400, 0), num_points=101,
                           other_params={'sort_by': 'x'}))

# --- 5. Solver and Output ---
builder.add_executioner_block(type="Transient", solve_type="NEWTON",
                              time_stepper_type='ConstantDT')

builder.add_preconditioning_block(active_preconditioner='mumps')
builder.add_outputs_block(exodus=True, csv=True)

# --- 6. Generate Input File ---
builder.generate_input_file(input_file)

#%% --- 7. Run Simulation ---
print("\n--- Starting MOOSE Simulation Runner ---")
try:
    moose_executable = "/rcp/rcp42/home/shenyaojin/Documents/bakken_mariner/moose_env/moose/modules/porous_flow/porous_flow-opt"
    if not os.path.exists(moose_executable):
        print(f"ERROR: MOOSE executable not found at '{moose_executable}'. Please update the path.")

    runner = MooseRunner(moose_executable_path=moose_executable)
    success, stdout, stderr = runner.run(
        input_file_path=input_file,
        output_directory=output_dir,
        num_processors=2,
        log_file_name="simulation.log"
    )
    if success:
        print("\nSimulation completed successfully!")
    else:
        print("\nSimulation failed.")
        print("--- STDERR from MOOSE ---")
        print(stderr)
except Exception as e:
    print(f"\nAn error occurred during the simulation run: {e}")