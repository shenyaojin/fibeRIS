# This model builder will create a simple baseline model for history matching tasks.
# I will illustrate this model in my paper, so please refer to the paper published in
# It uses the data from Mariner. For confidential purposes, no real data is included in fiberis.
# Please replace the data loading section with your own data source.
# Shenyao Jin, shenyaojin@mines.edu, 11/16/2025

import numpy as np

from fiberis.analyzer.Data1D.Data1D_Gauge import Data1DGauge
from fiberis.moose.config import (
    MatrixConfig, SRVConfig, HydraulicFractureConfig, ZoneMaterialProperties, SimpleFluidPropertiesConfig,
    PointValueSamplerConfig, LineValueSamplerConfig, TimeSequenceStepper
)
from fiberis.moose.model_builder import ModelBuilder

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
        ny_per_layer_half=kwargs.get('ny_per_layer_half', 80),
        bias_y=kwargs.get('bias_y', 1.2)
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

    # Load pressure gauge data from Mariner dataset
    pg_dataframe = Data1DGauge()
    pg_dataframe.load_npz(kwargs.get("data_path",
                                     "data/fiberis_format/post_processing/history_matching_pressure_profile_full.npz"))
    # here you should replace with your own data source

    pg_dataframe.rename("gauge_hf_pressure")
    pg_dataframe.data = pg_dataframe.data * 6894.76  # psi to kPa

    builder.add_variables([
        {"name": "pp", "params": {"initial_condition": kwargs.get('initial_pressure', pg_dataframe.data[0])}},
        {"name": "disp_x", "params": {"initial_condition": 0}},
        {"name": "disp_y", "params": {"initial_condition": 0}}
    ])

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
        source_data1d = pg_dataframe
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


