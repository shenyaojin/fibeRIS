# This model builder is a specialized version for Fervo's history matching tasks.
# CHANGELOG:
# 12/09/2025: Add initial condition (IC) for each block in the model, which can improve the numerical stability.


# Shenyao Jin, 11/24/2025

import numpy as np
import os
import datetime
from fiberis.analyzer.Data1D.Data1D_Gauge import Data1DGauge
from fiberis.analyzer.Data2D.core2D import Data2D
from typing import List
from fiberis.moose.config import (
    MatrixConfig, SRVConfig, HydraulicFractureConfig, ZoneMaterialProperties, SimpleFluidPropertiesConfig,
    PointValueSamplerConfig, LineValueSamplerConfig, TimeSequenceStepper, InitialConditionConfig
)
from fiberis.moose.model_builder import ModelBuilder
from fiberis.io.reader_moose_vpp import MOOSEVectorPostProcessorReader

def build_baseline_model(**kwargs) -> ModelBuilder:
    """
    In this function, I extract the common baseline model building steps from 106r2_SingleFracScanner_test_func.py
    to create a reusable baseline model generator. This script should be much more stable and no mistakes should be made
    by QCing the original script multiple times.
    --Shenyao Jin

    :param project_name: The name of the project. Defaults to "BaselineModel".
    :param model_width: The width of the model domain in meters. Defaults to 200.0 * 0.3048.
    :param fracture_y_coords: A list of y-coordinates for the fractures. Defaults to [0.0].
    :param model_length: The length of the model domain in meters. Defaults to 800.0 * 0.3048.
    :param nx: The number of elements in the x-direction. Defaults to 200.
    :param ny_per_layer_half: The number of elements in the y-direction for each half-layer. Defaults to 80.
    :param bias_y: The mesh bias in the y-direction. Defaults to 1.2.
    :param matrix_perm: The permeability of the matrix. Defaults to 1e-18.
    :param srv_perm: The permeability of the SRV. Defaults to 1e-15.
    :param fracture_perm: The permeability of the fracture. Defaults to 1e-13.
    :param srv_length_ft: The length of the SRV in feet. Defaults to 400.
    :param srv_height_ft: The height of the SRV in feet. Defaults to 20.
    :param hf_length_ft: The length of the hydraulic fracture in feet. Defaults to 250.
    :param hf_height_ft: The height of the hydraulic fracture in feet. Defaults to 0.2.
    :param initial_pressure: The initial pressure of the model. Defaults to 5.17E7.
    :param monitoring_point_shift_ft: The shift of the monitoring point in feet. Defaults to 80.
    :param start_offset_y: The start offset of the fiber sampler in the y-direction. Defaults to 20.
    :param end_offset_y: The end offset of the fiber sampler in the y-direction. Defaults to 20.
    :param num_fiber_points: The number of points in the fiber sampler. Defaults to 200.
    :return: A ModelBuilder object representing the baseline model.
    """
    # Define default parameters
    conversion_factor = 0.3048  # feet to meters

    # "data/fiberis_format/post_processing/injection_pressure_full_profile.npz" <- injection pressure profile
    # Load gauge data for MOOSE, I have already packed the data in fiberis format.
    gauge_data_for_moose = Data1DGauge()
    gauge_data_for_moose.load_npz("data_fervo/fiberis_format/post_processing/gauge_data_for_simulation_synthetic_fault_pressure.npz")
    gauge_data_for_moose.data = 6894.76 * gauge_data_for_moose.data  # Convert psi to Pa

    # Start building the model
    builder = ModelBuilder(project_name=kwargs.get("project_name", "BaselineModel"))
    domain_bounds = (- kwargs.get('model_width', 200.0 * conversion_factor),
                     + kwargs.get('model_width', 200.0 * conversion_factor))

    frac_coords = kwargs.get('fracture_y_coords', [0.0 * conversion_factor])
    # If frac_coords is a list (for multi-fracture mesh), use the first element for this single-fracture model's center.
    frac_y_center = frac_coords[0] if isinstance(frac_coords, list) else frac_coords
    domain_length = kwargs.get('model_length', 800.0 * conversion_factor)
    builder.build_stitched_mesh_for_fractures(
        fracture_y_coords=frac_coords,
        domain_bounds=domain_bounds,
        domain_length=domain_length,
        nx=kwargs.get('nx', 200),
        ny_per_layer_half=kwargs.get('ny_per_layer_half', 110),
        bias_y=kwargs.get('bias_y', 1.1)
    )

    matrix_perm = kwargs.get('matrix_perm', 1e-18)
    srv_perm = kwargs.get('srv_perm', 1e-15)
    fracture_perm = kwargs.get('fracture_perm', 1e-13)

    # The tensor format for permeability in fiberis:
    matrix_perm_str = f"{matrix_perm} 0 0 0 {matrix_perm} 0 0 0 {matrix_perm}"
    srv_perm_str = f"{srv_perm} 0 0 0 {srv_perm} 0 0 0 {srv_perm}"
    fracture_perm_str = f"{fracture_perm} 0 0 0 {fracture_perm} 0 0 0 {fracture_perm}"

    # Material properties
    matrix_mats = ZoneMaterialProperties(porosity=0.01, permeability=matrix_perm_str)
    srv_mats = ZoneMaterialProperties(porosity=0.1, permeability=srv_perm_str)
    fracture_mats = ZoneMaterialProperties(porosity=0.16, permeability=fracture_perm_str)

    # Define Initial Conditions
    matrix_pressure_ic = InitialConditionConfig(
        name="initial_pressure_matrix",
        ic_type="ConstantIC",
        variable="pp",
        params={"value": gauge_data_for_moose.data[0]}
    )
    srv_frac_pressure_ic = InitialConditionConfig(
        name="initial_pressure_srv_frac",
        ic_type="ConstantIC",
        variable="pp",
        params={"value": gauge_data_for_moose.data[0]}
    )

    builder.set_matrix_config(MatrixConfig(name="matrix", materials=matrix_mats, initial_conditions=[matrix_pressure_ic]))

    center_x_val = domain_length / 2.0
    srv_length_ft = kwargs.get('srv_length_ft', 280)
    srv_height_ft = kwargs.get('srv_height_ft', 20)  # <- Changed here. From 50 to 20
    hf_length_ft = kwargs.get('hf_length_ft', 250)
    hf_height_ft = kwargs.get('hf_height_ft', 0.2)

    geometries = [
        SRVConfig(name="srv", length=srv_length_ft * conversion_factor, height=srv_height_ft * conversion_factor,
                  center_x=center_x_val, center_y=frac_y_center, materials=srv_mats, initial_conditions=[srv_frac_pressure_ic]),
        HydraulicFractureConfig(name="hf", length=hf_length_ft * conversion_factor,
                                height=hf_height_ft * conversion_factor, center_x=center_x_val,
                                center_y=frac_y_center, materials=fracture_mats, initial_conditions=[srv_frac_pressure_ic])
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

    builder.add_nodeset_by_coord(nodeset_op_name="injection", new_boundary_name="injection",
                                 coordinates=(center_x_val, frac_y_center, 0))

    builder.add_variables([
        "pp",
        {"name": "disp_x", "params": {"initial_condition": 0}},
        {"name": "disp_y", "params": {"initial_condition": 0}}
    ])

    builder.set_porous_flow_dictator(dictator_name="dictator", porous_flow_variables="pp")
    builder.add_global_params({"PorousFlowDictator": "dictator", "displacements": "'disp_x disp_y'"})

    biot_coeff = 0.7
    builder.add_porous_flow_mass_time_derivative_kernel(kernel_name="dt", variable="pp")
    builder.add_porous_flow_darcy_base_kernel(kernel_name="flux", variable="pp")
    builder.add_stress_divergence_tensor_kernel(kernel_name="grad_stress_x", variable="disp_x", component=0)
    builder.add_stress_divergence_tensor_kernel(kernel_name="grad_stress_y", variable="disp_y", component=1)
    builder.add_porous_flow_effective_stress_coupling_kernel(kernel_name="eff_stress_x", variable="disp_x", component=0,
                                                             biot_coefficient=biot_coeff)
    builder.add_porous_flow_effective_stress_coupling_kernel(kernel_name="eff_stress_y", variable="disp_y", component=1,
                                                             biot_coefficient=biot_coeff)
    builder.add_porous_flow_mass_volumetric_expansion_kernel(kernel_name="mass_exp", variable="pp")

    fluid_property = SimpleFluidPropertiesConfig(name="water", bulk_modulus=2.2E9, viscosity=1.0E-3, density0=1000.0)
    builder.add_fluid_properties_config(fluid_property)
    builder.add_poromechanics_materials(
        fluid_properties_name="water",
        biot_coefficient=biot_coeff,
        solid_bulk_compliance=2E-11
    )

    builder.add_piecewise_function_from_data1d(name="injection_pressure_func", source_data1d=gauge_data_for_moose)

    builder.set_hydraulic_fracturing_bcs(
        injection_well_boundary_name="injection",
        injection_pressure_function_name="injection_pressure_func",
        confine_disp_x_boundaries="left right",
        confine_disp_y_boundaries="top bottom"
    )

    builder.add_standard_tensor_aux_vars_and_kernels({"stress": "stress", "total_strain": "strain"})

    # Add post-processors to model builder
    # This part is from baseline_model_builder.py (v1) which provides better post-processing options.
    shift_list_ft = kwargs.get('shift_list_ft', [80.0])
    angle = kwargs.get('angle', 30.0)  # Angle in degrees for clockwise rotation

    for shift_val_ft in shift_list_ft:
        # Center point sampler, pressure
        builder.add_postprocessor(
            PointValueSamplerConfig(
                name=f"hf_center_dispx_sampler_{shift_val_ft}ft",
                variable="disp_x",
                point=(center_x_val, frac_y_center, 0)
            )
        )

        # Center point sampler, strain_yy
        builder.add_postprocessor(
            PointValueSamplerConfig(
                name=f"hf_center_dispy_sampler_{shift_val_ft}ft",
                variable="disp_y",
                point=(center_x_val, frac_y_center, 0)
            )
        )

        # Monitoring point sampler, pressure
        builder.add_postprocessor(
            PointValueSamplerConfig(
                name=f"monitor_point_dispx_sampler_{shift_val_ft}ft",
                variable="disp_x",
                point=(center_x_val + shift_val_ft * conversion_factor, frac_y_center, 0)
            )
        )

        # Monitoring point sampler, strain_yy
        builder.add_postprocessor(
            PointValueSamplerConfig(
                name=f"monitor_point_dispy_sampler_{shift_val_ft}ft",
                variable="disp_y",
                point=(center_x_val + shift_val_ft * conversion_factor, frac_y_center, 0)
            )
        )

        # Line sampler calculation
        x_center = center_x_val + shift_val_ft * conversion_factor
        y_start = domain_bounds[0] + kwargs.get("start_offset_y", 20) * conversion_factor
        y_end = domain_bounds[1] - kwargs.get("end_offset_y", 20) * conversion_factor
        y_center = (y_start + y_end) / 2.0

        theta = np.deg2rad(angle)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        # Rotate start point
        y_start_rel = y_start - y_center
        x_start_rot = x_center + y_start_rel * sin_theta
        y_start_rot = y_center + y_start_rel * cos_theta

        # Rotate end point
        y_end_rel = y_end - y_center
        x_end_rot = x_center + y_end_rel * sin_theta
        y_end_rot = y_center + y_end_rel * cos_theta

        # Boundary checks
        x_min_bound, x_max_bound = 10, domain_length - 10
        y_min_bound, y_max_bound = domain_bounds[0] + 10, domain_bounds[1] - 10

        start_point = (
            np.clip(x_start_rot, x_min_bound, x_max_bound),
            np.clip(y_start_rot, y_min_bound, y_max_bound),
            0
        )
        end_point = (
            np.clip(x_end_rot, x_min_bound, x_max_bound),
            np.clip(y_end_rot, y_min_bound, y_max_bound),
            0
        )

        # Line sampler along the fracture, pressure
        builder.add_postprocessor(
            LineValueSamplerConfig(
                name=f"fiber_pressure_sampler_{shift_val_ft}ft",
                variable="pp",
                start_point=start_point,
                end_point=end_point,
                num_points=kwargs.get("num_fiber_points", 200),
                other_params={'sort_by': 'y'}
            )
        )

        # Line sampler along the fracture, strain components
        builder.add_postprocessor(
            LineValueSamplerConfig(
                name=f"fiber_strain_sampler_{shift_val_ft}ft",
                variable="strain_xx strain_yy strain_xy",
                start_point=start_point,
                end_point=end_point,
                num_points=kwargs.get("num_fiber_points", 200),
                other_params={'sort_by': 'y'}
            )
        )

    # Time sequence stepper
    total_time = gauge_data_for_moose.taxis[-1] - gauge_data_for_moose.taxis[0]
    # Down sample two dataframes to speed up the simulation.
    timestepper_profile = Data1DGauge()
    timestepper_profile = gauge_data_for_moose.copy()
    timestepper_profile.adaptive_downsample(kwargs.get("timestepper_max_points", 140))

    dt_control_func = TimeSequenceStepper()
    dt_control_func.from_data1d(timestepper_profile)

    # Define the time stepper block
    builder.add_executioner_block(
        end_time=total_time,
        dt=3600 * 24 * 5,
        time_stepper_type='TimeSequenceStepper',
        stepper_config=dt_control_func
    )

    builder.add_initial_conditions_from_configs()
    builder.add_preconditioning_block(active_preconditioner='mumps')
    builder.add_outputs_block(exodus=True, csv=True, exodus_execute_on='INITIAL FINAL')

    return builder


def post_processor_info_extractor(**kwargs) -> List[Data2D]:
    """
    This function extracts post-processor information from the simulation results.
    Will return four Data2D objects: one for pressure, and one for each strain component (xx, yy, xy).

    :param output_dir: The directory containing the MOOSE output CSV files.
    :return: A list of Data2D objects representing the extracted post-processor data.
    """
    output_dir = kwargs.get("output_dir")
    if not output_dir:
        raise ValueError("output_dir must be provided in kwargs")

    vector_reader = MOOSEVectorPostProcessorReader()
    max_processor_id, _ = vector_reader.get_max_indices(output_dir)

    pressure_data2d = None
    strain_xx_data2d = None
    strain_yy_data2d = None
    strain_xy_data2d = None

    for i in range(max_processor_id + 1):
        # Check sampler name by reading first variable
        vector_reader.read(directory=output_dir, post_processor_id=i, variable_index=1)

        if "fiber_pressure_sampler" in vector_reader.sampler_name:
            pressure_data2d = vector_reader.to_analyzer()
            pressure_data2d.name = "pressure"
        elif "fiber_strain_sampler" in vector_reader.sampler_name:
            # It's the strain sampler, now extract all components
            vector_reader.read(directory=output_dir, post_processor_id=i, variable_index=1)
            strain_xx_data2d = vector_reader.to_analyzer()
            strain_xx_data2d.name = "strain_xx"

            vector_reader.read(directory=output_dir, post_processor_id=i, variable_index=2)
            strain_yy_data2d = vector_reader.to_analyzer()
            strain_yy_data2d.name = "strain_yy"

            vector_reader.read(directory=output_dir, post_processor_id=i, variable_index=3)
            strain_xy_data2d = vector_reader.to_analyzer()
            strain_xy_data2d.name = "strain_xy"

    if pressure_data2d is None:
        raise FileNotFoundError("Could not find and extract 'fiber_pressure_sampler' data.")
    if strain_xx_data2d is None:
        raise FileNotFoundError("Could not find and extract 'fiber_strain_sampler' (strain_xx) data.")
    if strain_yy_data2d is None:
        raise FileNotFoundError("Could not find and extract 'fiber_strain_sampler' (strain_yy) data.")
    if strain_xy_data2d is None:
        raise FileNotFoundError("Could not find and extract 'fiber_strain_sampler' (strain_xy) data.")

    return [pressure_data2d, strain_xx_data2d, strain_yy_data2d, strain_xy_data2d]

if __name__ == "__main__":
    print("Don't run this script directly. It is meant to be imported as a module.\n--Shenyao")