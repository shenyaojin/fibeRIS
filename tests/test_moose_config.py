# Characterization tests for fiberis.moose.config
#
# Golden-master tests pinning the CURRENT defaults and serialization behavior of
# the config dataclasses / parameter blocks. Values were obtained by RUNNING the
# code. Tests must stay GREEN through a pure refactor.

import numpy as np
import pytest

from fiberis.moose.config import (
    ZoneMaterialProperties,
    InitialConditionConfig,
    MatrixConfig,
    HydraulicFractureConfig,
    SRVConfig,
    CasingLayerConfig,
    CasingConfig,
    IndicatorConfig,
    MarkerConfig,
    AdaptivityConfig,
    PostprocessorConfigBase,
    PointValueSamplerConfig,
    LineValueSamplerConfig,
    PostprocessorConfig,
    SimpleFluidPropertiesConfig,
    TimeStepperFunctionConfig,
    AdaptiveTimeStepperConfig,
    TimeSequenceStepper,
)
from fiberis.analyzer.Data1D.core1D import Data1D


# --- ZoneMaterialProperties -----------------------------------------------

def test_zone_material_properties_defaults():
    z = ZoneMaterialProperties(porosity=0.1, permeability=1e-15)
    assert z.porosity == 0.1
    assert z.permeability == 1e-15
    assert z.youngs_modulus is None
    assert z.poissons_ratio is None


def test_zone_material_properties_full():
    z = ZoneMaterialProperties(porosity=0.2, permeability="'1e-15 0 0'",
                               youngs_modulus=3e10, poissons_ratio=0.25)
    assert z.permeability == "'1e-15 0 0'"
    assert z.youngs_modulus == 3e10
    assert z.poissons_ratio == 0.25


# --- InitialConditionConfig ------------------------------------------------

def test_initial_condition_config_defaults():
    ic = InitialConditionConfig(name="ic1", ic_type="ConstantIC", variable="pp")
    assert ic.name == "ic1"
    assert ic.ic_type == "ConstantIC"
    assert ic.variable == "pp"
    assert ic.params == {}


def test_initial_condition_config_with_params():
    ic = InitialConditionConfig(name="ic1", ic_type="ConstantIC",
                                variable="pp", params={"value": 1.0})
    assert ic.params == {"value": 1.0}


# --- MatrixConfig ----------------------------------------------------------

def test_matrix_config():
    mats = ZoneMaterialProperties(porosity=0.05, permeability=1e-16)
    m = MatrixConfig(name="matrix", materials=mats)
    assert m.name == "matrix"
    assert m.materials is mats
    assert m.initial_conditions == []


# --- HydraulicFractureConfig ----------------------------------------------

def test_hydraulic_fracture_config_defaults():
    mats = ZoneMaterialProperties(porosity=0.5, permeability=1e-10)
    f = HydraulicFractureConfig(name="Frac1", length=200, height=0.2,
                                center_x=500, center_y=250, materials=mats)
    assert f.name == "Frac1"
    assert f.length == 200
    assert f.height == 0.2
    assert f.center_x == 500
    assert f.center_y == 250
    assert f.orientation_angle == 0.0
    assert f.mesh_length_param is None
    assert f.mesh_height_param is None
    assert f.initial_conditions == []


def test_hydraulic_fracture_config_nonzero_orientation_prints(capsys):
    mats = ZoneMaterialProperties(porosity=0.5, permeability=1e-10)
    HydraulicFractureConfig(name="F", length=1, height=1, center_x=0,
                            center_y=0, materials=mats, orientation_angle=30.0)
    out = capsys.readouterr().out
    assert "non-zero orientation_angle" in out


# --- SRVConfig -------------------------------------------------------------

def test_srv_config():
    mats = ZoneMaterialProperties(porosity=0.1, permeability=1e-13)
    s = SRVConfig(name="SRV1", length=300, height=80, center_x=500,
                  center_y=250, materials=mats)
    assert s.name == "SRV1"
    assert s.length == 300
    assert s.height == 80
    assert s.initial_conditions == []
    assert s.mesh_length_param is None


# --- Casing configs --------------------------------------------------------

def test_casing_layer_config():
    mats = ZoneMaterialProperties(porosity=0.1, permeability=1e-15)
    layer = CasingLayerConfig(name="sandstone", height=10.0, materials=mats)
    assert layer.name == "sandstone"
    assert layer.height == 10.0
    assert layer.materials is mats


def test_casing_config():
    mats = ZoneMaterialProperties(porosity=0.1, permeability=1e-15)
    layers = [CasingLayerConfig(name="l1", height=5.0, materials=mats),
              CasingLayerConfig(name="l2", height=7.0, materials=mats)]
    c = CasingConfig(name="casing", layers=layers,
                     injection_well_name="well", injection_well_x_coord=50.0)
    assert c.name == "casing"
    assert len(c.layers) == 2
    assert c.injection_well_name == "well"
    assert c.injection_well_x_coord == 50.0


# --- Adaptivity configs ----------------------------------------------------

def test_indicator_and_marker_config():
    ind = IndicatorConfig(name="ind", type="GradientJumpIndicator",
                          params={"variable": "pp"})
    mk = MarkerConfig(name="mk", type="ErrorFractionMarker",
                      params={"refine": 0.3})
    assert ind.name == "ind"
    assert ind.type == "GradientJumpIndicator"
    assert ind.params == {"variable": "pp"}
    assert mk.params == {"refine": 0.3}


def test_adaptivity_config_defaults_and_adders():
    a = AdaptivityConfig(marker_to_use="m", steps=2)
    assert a.marker_to_use == "m"
    assert a.steps == 2
    assert a.indicators == []
    assert a.markers == []

    ind = IndicatorConfig(name="i", type="T", params={})
    mk = MarkerConfig(name="m", type="T", params={})
    a.add_indicator(ind)
    a.add_marker(mk)
    assert a.indicators == [ind]
    assert a.markers == [mk]


def test_adaptivity_config_add_wrong_type_raises():
    a = AdaptivityConfig(marker_to_use="m", steps=1)
    with pytest.raises(TypeError):
        a.add_indicator("not an indicator")
    with pytest.raises(TypeError):
        a.add_marker("not a marker")


# --- Postprocessor configs -------------------------------------------------

def test_postprocessor_config_base_default_execute_on():
    b = PostprocessorConfigBase(name="b", pp_type="X")
    # NOTE: mutable default argument; current default value pinned here.
    assert b.execute_on == ['initial', 'timestep_end', 'final']
    assert b.variable is None
    assert b.variables is None
    assert b.other_params == {}


def test_point_value_sampler_tuple_to_string():
    p = PointValueSamplerConfig(name="pp_well", variable="pp", point=(500, 250, 0))
    assert p.pp_type == "PointValue"
    assert p.point == "500 250 0"
    assert p.variable == "pp"
    # execute_on default for this subclass is None (overrides base default).
    assert p.execute_on is None


def test_point_value_sampler_string_point_passthrough():
    p = PointValueSamplerConfig(name="pp", variable="pp", point="1 2 3")
    assert p.point == "1 2 3"


def test_line_value_sampler_serialization():
    l = LineValueSamplerConfig(name="ln", variable="pp",
                               start_point=(0, 1, 0), end_point=(2, 3, 0),
                               num_points=51, output_vector=True)
    assert l.pp_type == "LineValueSampler"
    assert l.other_params["start_point"] == "0 1 0"
    assert l.other_params["end_point"] == "2 3 0"
    assert l.other_params["num_points"] == 51
    assert l.other_params["output_vector_postprocessor"] is True
    assert l.start_point_str == "0 1 0"
    assert l.end_point_str == "2 3 0"
    assert l.num_sample_points == 51


def test_line_value_sampler_no_output_vector_key_omitted():
    l = LineValueSamplerConfig(name="ln", variable="pp",
                               start_point="0 0 0", end_point="1 1 1")
    # When output_vector is False, the key is not added.
    assert "output_vector_postprocessor" not in l.other_params
    assert l.other_params["num_points"] == 100


def test_generic_postprocessor_config_pops_standard_keys():
    params = {"execute_on": "final", "variable": "pp", "foo": "bar"}
    g = PostprocessorConfig(name="g", pp_type="ElementAverageValue", params=params)
    assert g.pp_type == "ElementAverageValue"
    assert g.execute_on == "final"
    assert g.variable == "pp"
    assert g.other_params == {"foo": "bar"}
    # NOTE: the input dict is mutated in place (pop), 'foo' remains.
    assert params == {"foo": "bar"}


# --- SimpleFluidPropertiesConfig ------------------------------------------

def test_simple_fluid_properties_defaults():
    f = SimpleFluidPropertiesConfig(name="water")
    assert f.name == "water"
    assert f.bulk_modulus == 2.2E9
    assert f.viscosity == 1.0E-3
    assert f.density0 == 1000.0
    assert f.thermal_expansion == 0.0002
    assert f.cp == 4194.0
    assert f.cv == 4186.0
    assert f.porepressure_coefficient == 1.0


# --- TimeStepperFunctionConfig --------------------------------------------

def test_timestepper_function_config_maps_to_data1d():
    tf = TimeStepperFunctionConfig(name="cs", x_values=[0, 1, 2],
                                   y_values=[0.1, 0.2, 0.3])
    assert tf.name == "cs"
    np.testing.assert_array_equal(tf.taxis, np.array([0, 1, 2]))
    np.testing.assert_array_equal(tf.data, np.array([0.1, 0.2, 0.3]))


def test_timestepper_function_config_from_data1d():
    d = Data1D(taxis=np.array([0., 10., 30.]), data=np.array([1., 2., 3.]))
    tf = TimeStepperFunctionConfig.load_timestep_from_data1d("cs2", d)
    # dt = diff(taxis)/2 = [5, 10], last appended -> [5, 10, 10]
    np.testing.assert_array_equal(tf.data, np.array([5., 10., 10.]))
    np.testing.assert_array_equal(tf.taxis, np.array([0., 10., 30.]))


def test_timestepper_function_config_from_data1d_too_few_points_raises():
    d = Data1D(taxis=np.array([0.]), data=np.array([1.]))
    with pytest.raises(ValueError):
        TimeStepperFunctionConfig.load_timestep_from_data1d("x", d)


# --- AdaptiveTimeStepperConfig --------------------------------------------

def test_adaptive_time_stepper_config():
    tf = TimeStepperFunctionConfig(name="cs", x_values=[0, 1], y_values=[0.1, 0.1])
    cfg = AdaptiveTimeStepperConfig(functions=[tf])
    assert cfg.functions == [tf]


# --- TimeSequenceStepper ---------------------------------------------------

def test_time_sequence_stepper_from_list():
    ts = TimeSequenceStepper([0, 1, 2.5])
    assert ts.time_sequence == "0 1 2.5"


def test_time_sequence_stepper_empty_default():
    ts = TimeSequenceStepper()
    assert ts.time_sequence == ""


def test_time_sequence_stepper_bad_type_raises():
    with pytest.raises(TypeError):
        TimeSequenceStepper(time_sequence="not a list")


def test_time_sequence_stepper_from_data1d():
    ts = TimeSequenceStepper()
    d = Data1D(taxis=np.array([0., 1., 2.]), data=np.array([1., 1., 1.]))
    ts.from_data1d(d)
    assert ts.time_sequence == "0.0 1.0 2.0"


def test_time_sequence_stepper_from_data1d_empty_raises():
    ts = TimeSequenceStepper()
    d = Data1D(taxis=None, data=None)
    with pytest.raises(ValueError):
        ts.from_data1d(d)
