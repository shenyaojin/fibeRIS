# This model builder will create a simple baseline model for history matching tasks.
# I will illustrate this model in my paper, so please refer to the paper published in
# It uses the data from Mariner. For confidential purposes, no real data is included in fiberis.
# Please replace the data loading section with your own data source.
# Shenyao Jin, shenyaojin@mines.edu, 11/16/2025

import numpy as np
import os
from fiberis.analyzer.Data1D.Data1D_Gauge import Data1DGauge
from fiberis.analyzer.Data2D.core2D import Data2D
from typing import List
from fiberis.moose.config import (
    MatrixConfig, SRVConfig, HydraulicFractureConfig, ZoneMaterialProperties, SimpleFluidPropertiesConfig,
    PointValueSamplerConfig, LineValueSamplerConfig, TimeSequenceStepper
)
from fiberis.moose.model_builder import ModelBuilder
from fiberis.io.reader_moose_vpp import MOOSEVectorPostProcessorReader

def build_baseline_model(**kwargs) -> ModelBuilder:
    """
    This function builds a baseline model for history matching tasks.
    It contains a one-fracture with one-SRV setup.
    Most of the parameters are changeable through kwargs.

    -- Built by Shenyao Jin, 11/16/2025

    :param kwargs: Parameters to customize the model. See the code for details.
    :return: A ModelBuilder object representing the baseline model, which is ready to render and run.
    """

    builder = ModelBuilder(project_name=kwargs.get("project_name", "baseline_model"))
    frac_coords = 0  # Should be in the center of the model

    conversion_factor = 0.3048  # feet to meters, I just don't want to change all the numbers
    domain_bounds = (- kwargs.get('model_width', 50.0 * conversion_factor),
                     + kwargs.get('model_width', 50.0 * conversion_factor))

    domain_length = kwargs.get('model_length', 400.0 * conversion_factor)

    builder.build_stitched_mesh_for_fractures(
        fracture_y_coords=frac_coords,
        domain_bounds=domain_bounds,
        domain_length=domain_length,
        nx=kwargs.get('nx', 200),
        ny_per_layer_half=kwargs.get('ny_per_layer_half', 100),
        bias_y=kwargs.get('bias_y', 1.05)
    )

    matrix_perm = kwargs.get('matrix_perm', 1e-18)
    srv_perm = kwargs.get('srv_perm', 1e-16)
    fracture_perm = kwargs.get('fracture_perm', 1e-14)

    matrix_perm_str = f"{matrix_perm} 0 0 0 {matrix_perm} 0 0 0 {matrix_perm}"
    srv_perm_str = f"{srv_perm} 0 0 0 {srv_perm} 0 0 0 {srv_perm}"
    fracture_perm_str = f"{fracture_perm} 0 0 0 {fracture_perm} 0 0 0 {fracture_perm}"

    matrix_mats = ZoneMaterialProperties(porosity=0.01, permeability=matrix_perm_str)
    srv_mats = ZoneMaterialProperties(porosity=0.1, permeability=srv_perm_str)
    fracture_mats = ZoneMaterialProperties(porosity=0.16, permeability=fracture_perm_str)

    builder.set_matrix_config(MatrixConfig(name="matrix", materials=matrix_mats))

    center_x_val = domain_length / 2.0
    srv_length_ft2 = kwargs.get('srv_length_ft', 285)
    srv_height_ft2 = kwargs.get('srv_height_ft', 5)
    hf_length_ft = kwargs.get('hf_length_ft', 250)
    hf_height_ft = kwargs.get('hf_height_ft', 0.2)

    geometries = [
        SRVConfig(name="srv", length=srv_length_ft2 * conversion_factor, height=srv_height_ft2 * conversion_factor,
                  center_x=center_x_val, center_y=frac_coords, materials=srv_mats),
        HydraulicFractureConfig(name="hf", length=hf_length_ft * conversion_factor,
                                height=hf_height_ft * conversion_factor, center_x=center_x_val,
                                center_y=frac_coords, materials=fracture_mats)
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
                                 coordinates=(center_x_val, frac_coords, 0))

    # Load pressure gauge data from Mariner dataset for injection and time stepping
    # This section is adapted from scripts/DSS_history_match/108_misfit_func.py for better convergence.
    pressure_gauge_g1_path = kwargs.get("pressure_gauge_g1_path", "data/fiberis_format/prod/gauges/pressure_g1.npz")
    injection_gauge_pressure_path = kwargs.get("injection_gauge_pressure_path", "data/fiberis_format/prod/gauges/gauge4_data_prod.npz")

    # Load and preprocess gauge data
    gauge_data_interference = Data1DGauge()
    gauge_data_interference.load_npz(pressure_gauge_g1_path)

    injection_gauge_pressure = Data1DGauge()
    injection_gauge_pressure.load_npz(injection_gauge_pressure_path)
    # Select the production data up to the point where the interference data begins.
    injection_gauge_pressure.select_time(injection_gauge_pressure.start_time, gauge_data_interference.start_time)
    injection_gauge_pressure.remove_abnormal_data(threshold=300, method='mean')

    # Create copies for processing
    injection_gauge_pressure_copy = injection_gauge_pressure.copy()
    gauge_data_interference_copy = gauge_data_interference.copy()
    injection_gauge_pressure_copy.adaptive_downsample(300)
    gauge_data_interference_copy.adaptive_downsample(600)

    # Shift the interference gauge data to align with DSS data (one is wellhead, the other is downhole)
    if len(injection_gauge_pressure.data) > 0:
        difference_val = injection_gauge_pressure.data[-1] - gauge_data_interference.data[0]
        gauge_data_interference_copy.data += difference_val

    # Merge the two profiles
    injection_gauge_pressure_copy.right_merge(gauge_data_interference_copy)
    injection_gauge_pressure_copy.rename("injection pressure full profile")

    # Use this processed data for the injection pressure function
    gauge_data_for_moose = injection_gauge_pressure_copy.copy()
    # Quick fix: remove abnormal high pressures
    gauge_data_for_moose.data = 6894.76 * gauge_data_for_moose.data  # Convert psi to Pa

    builder.add_variables([
        {"name": "pp", "params": {"initial_condition": kwargs.get('initial_pressure', gauge_data_for_moose.data[0])}},
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

    builder.add_piecewise_function_from_data1d(
        name = "injection_pressure_func",
        source_data1d = gauge_data_for_moose
    )

    builder.set_hydraulic_fracturing_bcs(
        injection_well_boundary_name="injection",
        injection_pressure_function_name="injection_pressure_func",
        confine_disp_x_boundaries="left right",
        confine_disp_y_boundaries="top bottom"
    )

    builder.add_standard_tensor_aux_vars_and_kernels({"stress": "stress", "total_strain": "strain"})

    # Add post-processors
    # There will be (1) one sample at the center of the fracture, (2) one at monitoring point
    # (3) The fiber will be treated as a linear sampler.

    shift_val_ft = kwargs.get('monitoring_point_shift_ft', 80) # The distance from monitor well to the center of the fracture

    # Center point sampler, pressure
    builder.add_postprocessor(
        PointValueSamplerConfig(
            name = "hf_center_pressure_sampler",
            variable = "pp",
            point = (center_x_val, frac_coords, 0)
        )
    )

    # Center point sampler, strain_yy
    builder.add_postprocessor(
        PointValueSamplerConfig(
            name = "hf_center_strain_yy_sampler",
            variable = "strain_yy",
            point = (center_x_val, frac_coords, 0)
        )
    )

    # Monitoring point sampler, pressure
    builder.add_postprocessor(
        PointValueSamplerConfig(
            name = "monitor_point_pressure_sampler",
            variable = "pp",
            point = (center_x_val + shift_val_ft * conversion_factor, frac_coords, 0)
        )
    )

    # Monitoring point sampler, strain_yy
    builder.add_postprocessor(
        PointValueSamplerConfig(
            name = "monitor_point_strain_yy_sampler",
            variable = "strain_yy",
            point = (center_x_val + shift_val_ft * conversion_factor, frac_coords, 0)
        )
    )

    # Line sampler along the fracture, pressure
    builder.add_postprocessor(
        LineValueSamplerConfig(
            name = "fiber_pressure_sampler",
            variable = "pp",
            start_point = (center_x_val + shift_val_ft * conversion_factor, domain_bounds[0] + kwargs.get("start_offset_y", 20) * conversion_factor, 0),
            end_point = (center_x_val + shift_val_ft * conversion_factor, domain_bounds[1] - kwargs.get("end_offset_y", 20) * conversion_factor, 0),
            num_points = kwargs.get("num_fiber_points", 200),
            other_params = {'sort_by': 'y'}
        )
    )

    # Line sampler along the fracture, strain_yy
    builder.add_postprocessor(
        LineValueSamplerConfig(
            name = "fiber_strain_yy_sampler",
            variable = "strain_yy",
            start_point = (center_x_val + shift_val_ft * conversion_factor, domain_bounds[0] + kwargs.get("start_offset_y", 20) * conversion_factor, 0),
            end_point = (center_x_val + shift_val_ft * conversion_factor, domain_bounds[1] - kwargs.get("end_offset_y", 20) * conversion_factor, 0),
            num_points = kwargs.get("num_fiber_points", 200),
            other_params = {'sort_by': 'y'}
        )
    )

    # Define the time stepper
    total_time = gauge_data_for_moose.taxis[-1] - gauge_data_for_moose.taxis[0]
    # Down sample two dataframes to reduce computational cost
    # Logic here can be improved in the future.
    gauge_data_interference_stepper = gauge_data_interference_copy.copy()
    injection_gauge_pressure_stepper = injection_gauge_pressure_copy.copy()
    gauge_data_interference_stepper.adaptive_downsample(120)
    injection_gauge_pressure_stepper.adaptive_downsample(20)
    timestepper_profile = injection_gauge_pressure_stepper.copy()
    timestepper_profile.select_time(timestepper_profile.start_time, gauge_data_interference_stepper.start_time)
    timestepper_profile.right_merge(gauge_data_interference_stepper)

    # Define the time stepper function
    dt_control_func = TimeSequenceStepper()
    dt_control_func.from_data1d(timestepper_profile)

    # Define the time stepper block
    builder.add_executioner_block(
        end_time=total_time,
        dt=3600 * 24 * 5,
        time_stepper_type='TimeSequenceStepper',
        stepper_config=dt_control_func
    )

    builder.add_preconditioning_block(active_preconditioner='mumps')
    builder.add_outputs_block(exodus=False, csv=True)

    return builder

def post_processor_info_extractor(**kwargs) -> List[Data2D]:
    """
    This function extracts post-processor information from the simulation results.
    Will return two Data2D objects: one for pressure, one for strain_yy.

    :param kwargs: Parameters to customize the extraction. See the code for details.
                   'output_dir' (str): The directory containing the MOOSE output CSV files.
    :return: A list of Data2D objects representing the extracted post-processor data.
    """
    output_dir = kwargs.get("output_dir")
    if not output_dir:
        raise ValueError("output_dir must be provided in kwargs")

    vector_reader = MOOSEVectorPostProcessorReader()
    max_processor_id, _ = vector_reader.get_max_indices(output_dir)

    pressure_data2d = None
    strain_data2d = None

    for i in range(max_processor_id + 1):
        vector_reader.read(directory=output_dir, post_processor_id=i, variable_index=1)
        
        if "fiber_pressure_sampler" in vector_reader.sampler_name:
            pressure_data2d = vector_reader.to_analyzer()
        elif "fiber_strain_yy_sampler" in vector_reader.sampler_name:
            strain_data2d = vector_reader.to_analyzer()

    if pressure_data2d is None:
        raise FileNotFoundError("Could not find and extract 'fiber_pressure_sampler' data.")
    if strain_data2d is None:
        raise FileNotFoundError("Could not find and extract 'fiber_strain_yy_sampler' data.")

    return [pressure_data2d, strain_data2d]


if __name__ == "__main__":
    print("Don't run this file... import it as a module instead.")