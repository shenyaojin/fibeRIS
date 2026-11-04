import numpy as np

from fiberis.moose.model_builder import ModelBuilder
from fiberis.moose.config import InjectionStageConfig
from fiberis.analyzer.Data1D.Data1D_Gauge import Data1DGauge


def _build(stages, tmp_path):
    builder = ModelBuilder(project_name="staged_test")
    builder._add_generic_mesh_generator("base_mesh", "CartesianMeshGenerator",
                                        {"dim": 2, "dx": "'10 10 10'", "dy": "'10 10 10'"}, input_op="")
    gauge = Data1DGauge()
    gauge.taxis = np.array([0.0, 100.0])
    gauge.data = np.array([0.0, 1.0])
    gauge.name = "p"
    builder.add_piecewise_function_from_data1d(name="pfunc", source_data1d=gauge)
    builder.add_staged_pressure_injection(stages=stages, pressure_function_name="pfunc")
    path = tmp_path / "deck.i"
    builder.generate_input_file(str(path))
    return path.read_text()


def test_point_stage_uses_extra_nodeset(tmp_path):
    text = _build([InjectionStageConfig("s1", (10, 10, 0), 0, 50)], tmp_path)
    assert "ExtraNodesetGenerator" in text
    assert "BoundingBoxNodeSetGenerator" not in text


def test_line_stage_uses_bounding_box_nodeset(tmp_path):
    stages = [InjectionStageConfig("s1", (10, 10, 0), 0, 50, bounding_box=((9.99, 9, -0.1), (10.01, 21, 0.1))),
              InjectionStageConfig("s2", (20, 10, 0), 50, 100)]
    text = _build(stages, tmp_path)
    assert text.count("BoundingBoxNodeSetGenerator") == 1
    assert "bottom_left = '9.99 9 -0.1'" in text or "bottom_left = 9.99 9 -0.1" in text
    assert "ExtraNodesetGenerator" in text  # s2 is still a point
