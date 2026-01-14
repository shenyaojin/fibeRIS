# version 2 of baseline model generator, aim to remove error in previous version
# Modified from 107 file, using same structure as v1 generator
# Shenyao Jin, 01/14/2026

import numpy as np
import os
import datetime
import matplotlib.pyplot as plt
from fiberis.analyzer.Data1D.Data1D_Gauge import Data1DGauge
from fiberis.analyzer.Data2D.Data2D_XT_DSS import DSS2D
from typing import List
from fiberis.moose.config import (
    MatrixConfig, SRVConfig, HydraulicFractureConfig, ZoneMaterialProperties, SimpleFluidPropertiesConfig,
    PointValueSamplerConfig, LineValueSamplerConfig, TimeSequenceStepper, InitialConditionConfig
)
from fiberis.moose.model_builder import ModelBuilder
from fiberis.io.reader_moose_vpp import MOOSEVectorPostProcessorReader

def build_baseline_model(**kwargs) -> ModelBuilder:
    """
    This model builder aims to fix the issues in previous baseline model generator.
    -- Shenyao Jin, 01/14/2026


    :param kwargs:
    :return: ModelBuilder object
    """
    conversion_factor = 0.3048  # feet to meters

    # LEGACY: preprocessing gauge data
    DSS_datapath = "data/fiberis_format/s_well/dss_data/Mariner 14x-36-POW-S - RFS strain change.npz"
    pressure_gauge_g1_path = "data/fiberis_format/prod/gauges/pressure_g1.npz"
    injection_gauge_pressure_path = "data/fiberis_format/prod/gauges/gauge4_data_prod.npz"

    DSSdata = DSS2D()
    DSSdata.load_npz(DSS_datapath)
    mds = DSSdata.daxis
    ind = (mds > 7500) & (mds < 15000)
    drift_val = np.median(DSSdata.data[ind, :], axis=0)
    DSSdata.data -= drift_val.reshape((1, -1))
    DSSdata.select_time(0, 400000)
    DSSdata.select_depth(12000, 16360)

    DSSdata_copy = DSSdata.copy()
    DSSdata_copy.select_depth(14500, 15500)

    # %% 2. Load and preprocess gauge data
    gauge_data_interference = Data1DGauge()
    gauge_data_interference.load_npz(pressure_gauge_g1_path)
    gauge_data_interference.select_time(DSSdata.start_time, DSSdata.get_end_time())

    injection_gauge_pressure = Data1DGauge()
    injection_gauge_pressure.load_npz(injection_gauge_pressure_path)
    print(injection_gauge_pressure.start_time, DSSdata.start_time)
    injection_gauge_pressure.select_time(injection_gauge_pressure.start_time, DSSdata.start_time)
    injection_gauge_pressure.remove_abnormal_data(threshold=300, method='mean')

    injection_gauge_pressure_copy = injection_gauge_pressure.copy()
    gauge_data_interference_copy = gauge_data_interference.copy()
    injection_gauge_pressure_copy.adaptive_downsample(300)
    gauge_data_interference_copy.adaptive_downsample(600)

    # Shift the interference gauge data to align with DSS data (one is wellhead, the other is downhole)
    difference_val = injection_gauge_pressure.data[-1] - gauge_data_interference.data[0]
    gauge_data_interference_copy.data += difference_val

    injection_gauge_pressure_copy.right_merge(gauge_data_interference_copy)
    injection_gauge_pressure_copy.rename("injection pressure full profile")
    # DATA Processing complete

    builder = ModelBuilder(project_name=kwargs.get("project_name", "BaselineModel"))
    frac_coords = kwargs.get('fracture_y_coords', [0.0 * conversion_factor])
    # If frac_coords is a list (for multi-fracture mesh), use the first element for this single-fracture model's center.
    frac_y_center = frac_coords[0] if isinstance(frac_coords, list) else frac_coords
    domain_bounds = (- kwargs.get('model_width', 200.0 * conversion_factor),
                     + kwargs.get('model_width', 200.0 * conversion_factor))

    domain_length = kwargs.get('model_length', 800.0 * conversion_factor)
    builder.build_stitched_mesh_for_fractures(
        fracture_y_coords=frac_coords,
        domain_bounds=domain_bounds,
        domain_length=domain_length,
        nx=kwargs.get('nx', 200),
        ny_per_layer_half=kwargs.get('ny_per_layer_half', 80),
        bias_y=kwargs.get('bias_y', 1.2)
    )

    # Define material properties
    matrix_perm = kwargs.get('matrix_perm', 1e-18)
    srv_perm = kwargs.get('srv_perm', 1e-15)
    srv_perm2 = kwargs.get('srv_perm2', 1e-14)
    fracture_perm = kwargs.get('fracture_perm', 1e-13)

    # The tensor format for permeability in fiberis:
    matrix_perm_str = f"{matrix_perm} 0 0 0 {matrix_perm} 0 0 0 {matrix_perm}"
    srv_perm_str = f"{srv_perm} 0 0 0 {srv_perm} 0 0 0 {srv_perm}"
    srv_perm_str2 = f"{srv_perm2} 0 0 0 {srv_perm2} 0 0 0 {srv_perm2}"
    fracture_perm_str = f"{fracture_perm} 0 0 0 {fracture_perm} 0 0 0 {fracture_perm}"

    matrix_mats = ZoneMaterialProperties(porosity=0.01, permeability=matrix_perm_str)
    srv_mats = ZoneMaterialProperties(porosity=0.1, permeability=srv_perm_str)
    srv_mats2 = ZoneMaterialProperties(porosity=0.1, permeability=srv_perm_str2)
    fracture_mats = ZoneMaterialProperties(porosity=0.16, permeability=fracture_perm_str)

    builder.set_matrix_config(MatrixConfig(name="matrix", materials=matrix_mats))

    center_x_val = domain_length / 2.0
    srv_length_ft2 = kwargs.get('srv_length_ft', 400)
    srv_height_ft2 = kwargs.get('srv_height_ft', 50)
    srv_length_ft1 = kwargs.get('srv_length_ft1', 250)
    srv_height_ft1 = kwargs.get('srv_height_ft1', 150) # This parameter is too large based on literature.
    hf_length_ft = kwargs.get('hf_length_ft', 250)
    hf_height_ft = kwargs.get('hf_height_ft', 0.2)

    geometries = [
        SRVConfig(name="srv_tall", length=srv_length_ft1 * conversion_factor, height=srv_height_ft1 * conversion_factor,
                  center_x=center_x_val, center_y=frac_y_center, materials=srv_mats2),
        SRVConfig(name="srv_wide", length=srv_length_ft2 * conversion_factor, height=srv_height_ft2 * conversion_factor,
                  center_x=center_x_val, center_y=frac_y_center, materials=srv_mats),
        HydraulicFractureConfig(name="hf", length=hf_length_ft * conversion_factor,
                                height=hf_height_ft * conversion_factor, center_x=center_x_val,
                                center_y=frac_y_center, materials=fracture_mats)
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
        {"name": "pp", "params": {"initial_condition": kwargs.get('initial_pressure', 5.17E7)}},
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

    fluid_property = SimpleFluidPropertiesConfig(name="water", bulk_modulus=2.2E9, viscosity=1.0E-3, density0=1000.0)
    builder.add_fluid_properties_config(fluid_property)
    builder.add_poromechanics_materials(
        fluid_properties_name="water",
        biot_coefficient=biot_coeff,
        solid_bulk_compliance=2E-11
    )

    gauge_data_for_moose = injection_gauge_pressure_copy.copy()
    gauge_data_for_moose.data = 6894.76 * gauge_data_for_moose.data  # Convert psi to Pa

    builder.add_piecewise_function_from_data1d(name="injection_pressure_func", source_data1d=gauge_data_for_moose)

    builder.set_hydraulic_fracturing_bcs(
        injection_well_boundary_name="injection",
        injection_pressure_function_name="injection_pressure_func",
        confine_disp_x_boundaries="left right",
        confine_disp_y_boundaries="top bottom"
    )

    builder.add_standard_tensor_aux_vars_and_kernels({"stress": "stress", "total_strain": "strain"})

    spacing_dss = (DSSdata.daxis[1] - DSSdata.daxis[0]) * conversion_factor

    # Add post-processors to model builder
    # This part is from baseline_model_builder.py (v1) which provides better post-processing options.
    shift_val_ft = kwargs.get('monitoring_point_shift_ft', 80)

    # Center point sampler, pressure
    builder.add_postprocessor(
        PointValueSamplerConfig(
            name="hf_center_pressure_sampler",
            variable="pp",
            point=(center_x_val, frac_y_center, 0)
        )
    )

    # Center point sampler, strain_yy
    builder.add_postprocessor(
        PointValueSamplerConfig(
            name="hf_center_strain_yy_sampler",
            variable="strain_yy",
            point=(center_x_val, frac_y_center, 0)
        )
    )

    # Monitoring point sampler, pressure
    builder.add_postprocessor(
        PointValueSamplerConfig(
            name="monitor_point_pressure_sampler",
            variable="pp",
            point=(center_x_val + shift_val_ft * conversion_factor, frac_y_center, 0)
        )
    )

    # Monitoring point sampler, strain_yy
    builder.add_postprocessor(
        PointValueSamplerConfig(
            name="monitor_point_strain_yy_sampler",
            variable="strain_yy",
            point=(center_x_val + shift_val_ft * conversion_factor, frac_y_center, 0)
        )
    )

    # Line sampler along the fracture, pressure
    builder.add_postprocessor(
        LineValueSamplerConfig(
            name="fiber_pressure_sampler",
            variable="pp",
            start_point=(center_x_val + shift_val_ft * conversion_factor,
                         domain_bounds[0] + kwargs.get("start_offset_y", 20) * conversion_factor, 0),
            end_point=(center_x_val + shift_val_ft * conversion_factor,
                       domain_bounds[1] - kwargs.get("end_offset_y", 20) * conversion_factor, 0),
            num_points=kwargs.get("num_fiber_points", 200),
            other_params={'sort_by': 'y'}
        )
    )

    # Line sampler along the fracture, strain_yy
    builder.add_postprocessor(
        LineValueSamplerConfig(
            name="fiber_strain_yy_sampler",
            variable="strain_yy",
            start_point=(center_x_val + shift_val_ft * conversion_factor,
                         domain_bounds[0] + kwargs.get("start_offset_y", 20) * conversion_factor, 0),
            end_point=(center_x_val + shift_val_ft * conversion_factor,
                       domain_bounds[1] - kwargs.get("end_offset_y", 20) * conversion_factor, 0),
            num_points=kwargs.get("num_fiber_points", 200),
            other_params={'sort_by': 'y'}
        )
    )

    # Time stepper
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