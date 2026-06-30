# Characterization tests for fiberis.moose.editor.MooseModelEditor
#
# Golden-master tests for the public editor API exercised on a small model built
# programmatically. Behavior pinned by RUNNING the code. Tests must stay GREEN
# through a pure refactor.

import pytest

from fiberis.moose.model_builder import ModelBuilder
from fiberis.moose.editor import MooseModelEditor


def _small_builder():
    """Build a tiny model with a couple of nested blocks for editing."""
    b = ModelBuilder(project_name="EditorTest")
    b.add_global_params({"displacements": "disp_x disp_y"})
    b.add_variables(["pp", "disp_x"])
    b.add_time_derivative_kernel(variable="pp", kernel_name="dot_pp")
    b.add_boundary_condition(name="confinex", bc_type="DirichletBC",
                             variable="disp_x", boundary_name="left",
                             params={"value": 0})
    return b


def test_set_parameter_on_top_level_block():
    b = _small_builder()
    editor = MooseModelEditor(b)
    editor.set_parameter(["GlobalParams"], "displacements", "disp_x disp_y disp_z")
    assert editor.get_parameter(["GlobalParams"], "displacements") == "disp_x disp_y disp_z"


def test_set_parameter_on_nested_block():
    b = _small_builder()
    editor = MooseModelEditor(b)
    # Path: [BCs] -> [confinex], add/update 'value'.
    editor.set_parameter(["BCs", "confinex"], "value", 42)
    assert editor.get_parameter(["BCs", "confinex"], "value") == 42


def test_set_parameter_adds_new_param():
    b = _small_builder()
    editor = MooseModelEditor(b)
    editor.set_parameter(["Kernels", "dot_pp"], "extra", "thing")
    assert editor.get_parameter(["Kernels", "dot_pp"], "extra") == "thing"


def test_set_parameter_missing_path_raises():
    b = _small_builder()
    editor = MooseModelEditor(b)
    with pytest.raises(ValueError):
        editor.set_parameter(["DoesNotExist"], "x", 1)


def test_get_parameter_missing_block_raises():
    b = _small_builder()
    editor = MooseModelEditor(b)
    with pytest.raises(ValueError):
        editor.get_parameter(["Nope", "Nada"], "x")


def test_get_parameter_missing_param_raises_keyerror():
    b = _small_builder()
    editor = MooseModelEditor(b)
    with pytest.raises(KeyError):
        editor.get_parameter(["GlobalParams"], "no_such_param")


def test_find_parameter_returns_all_paths():
    b = _small_builder()
    editor = MooseModelEditor(b)
    # 'variable' is present on the kernel and the BC sub-blocks.
    paths = editor.find_parameter("variable")
    assert ["Kernels", "dot_pp"] in paths
    assert ["BCs", "confinex"] in paths


def test_find_parameter_not_found_returns_empty():
    b = _small_builder()
    editor = MooseModelEditor(b)
    assert editor.find_parameter("totally_absent_param") == []


def test_find_block_recursive_empty_path_returns_none():
    b = _small_builder()
    editor = MooseModelEditor(b)
    assert editor._find_block_recursive([], b._top_level_blocks) is None


def test_print_model_structure_smoke(capsys):
    b = _small_builder()
    editor = MooseModelEditor(b)
    editor.print_model_structure()
    out = capsys.readouterr().out
    assert "MOOSE Model Structure for: EditorTest" in out
    assert "[GlobalParams]" in out
