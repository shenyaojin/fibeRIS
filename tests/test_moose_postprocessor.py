# Characterization tests for fiberis.moose.postprocessor.
#
# The MooseOutputReader class depends on meshio reading real Exodus II files
# (which we do not have), so it is exercised via a lightweight in-memory mesh
# (mirroring the module's own __main__ MockMooseOutputReader approach) to pin
# the pure-python extraction logic. MoosePointSamplerSet is pure CSV parsing
# and is characterized directly with synthesized MOOSE-style CSV files.
#
# These must remain GREEN through a pure refactor.

import os

import numpy as np
import pytest

from fiberis.moose import postprocessor
from fiberis.moose.postprocessor import MooseOutputReader, MoosePointSamplerSet


# ---------------------------------------------------------------------------
# In-memory mesh fixtures for MooseOutputReader (avoids meshio / Exodus files)
# ---------------------------------------------------------------------------

class _FakeMesh:
    def __init__(self, points, point_data, field_data=None, cell_data=None):
        self.points = np.asarray(points, dtype=float)
        self.point_data = point_data
        self.field_data = field_data if field_data is not None else {}
        self.cell_data = cell_data if cell_data is not None else {}


def _reader_with_mesh(mesh):
    """Construct a MooseOutputReader around a pre-built mesh, replaying the
    time-step inference logic from __init__ without touching meshio."""
    reader = MooseOutputReader.__new__(MooseOutputReader)
    reader.mesh = mesh
    reader.points = mesh.points
    reader._kdtree = None
    reader.time_steps = None
    if mesh.field_data and "time_whole" in mesh.field_data:
        reader.time_steps = np.array(mesh.field_data["time_whole"])
        if reader.time_steps.ndim == 0:
            reader.time_steps = np.array([reader.time_steps.item()])
    elif mesh.point_data and mesh.point_data.keys():
        first_var = next(iter(mesh.point_data))
        pdata = mesh.point_data[first_var]
        if isinstance(pdata, list):
            reader.time_steps = np.arange(len(pdata))
        elif isinstance(pdata, np.ndarray):
            reader.time_steps = np.array([0.0])
    return reader


# Five points forming a small 2D mesh (z=0), matching the module's __main__.
_POINTS = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0.5, 0.5, 0.0]]


def _steady_state_reader():
    mesh = _FakeMesh(
        _POINTS,
        point_data={"diffused": np.array([0.0, 1.0, 0.0, 1.0, 0.5])},
        field_data={},
    )
    return _reader_with_mesh(mesh)


def _transient_reader():
    # Two time steps, scalar nodal variable 'diffused'.
    mesh = _FakeMesh(
        _POINTS,
        point_data={
            "diffused": [
                np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
                np.array([10.0, 11.0, 12.0, 13.0, 14.0]),
            ]
        },
        field_data={"time_whole": [0.0, 0.5]},
    )
    return _reader_with_mesh(mesh)


class TestMooseOutputReaderSteadyState:
    def test_time_steps_inferred_as_single_zero(self):
        reader = _steady_state_reader()
        # No time_whole, point_data is an ndarray -> assume single step at t=0.
        np.testing.assert_array_equal(reader.get_time_steps(), [0.0])

    def test_nodal_variable_names(self):
        assert _steady_state_reader().get_nodal_variable_names() == ["diffused"]

    def test_cell_variable_names_empty(self):
        assert _steady_state_reader().get_cell_variable_names() == []

    def test_find_nearest_node_indices(self):
        reader = _steady_state_reader()
        # Query the exact center point and a corner.
        idx = reader.find_nearest_node_indices([(0.5, 0.5), (0.0, 0.0)])
        assert idx == [4, 0]

    def test_find_nearest_node_indices_empty(self):
        assert _steady_state_reader().find_nearest_node_indices([]) == []

    def test_extract_point_data_dict_numpy(self):
        reader = _steady_state_reader()
        result = reader.extract_point_data_over_time(
            "diffused", [(0.0, 0.0), (0.5, 0.5)], output_format="dict_numpy"
        )
        assert set(result.keys()) == {"point_0_node_0", "point_1_node_4"}
        np.testing.assert_array_equal(result["point_0_node_0"], [0.0])
        np.testing.assert_array_equal(result["point_1_node_4"], [0.5])

    def test_extract_point_data_pandas_df(self):
        reader = _steady_state_reader()
        df = reader.extract_point_data_over_time(
            "diffused", [(1.0, 1.0)], output_format="pandas_df"
        )
        # Node 3 is nearest (1,1). Index is the single time step [0.0].
        assert list(df.index) == [0.0]
        assert "point_0_node_3" in df.columns
        assert df["point_0_node_3"].tolist() == [1.0]

    def test_extract_missing_variable_returns_none(self):
        reader = _steady_state_reader()
        assert reader.extract_point_data_over_time("nope", [(0.0, 0.0)]) is None

    def test_waterfall_single_2d_array(self):
        reader = _steady_state_reader()
        arr = reader.extract_line_data_for_waterfall(
            "diffused", (0.0, 0.0), (1.0, 0.0), 2,
            output_format="single_2d_array_time_vs_space",
        )
        # 1 time step x 2 line points -> shape (1, 2).
        assert arr.shape == (1, 2)
        # Endpoints (0,0)->node0=0.0 and (1,0)->node1=1.0.
        np.testing.assert_array_equal(arr, [[0.0, 1.0]])


class TestMooseOutputReaderTransient:
    def test_time_steps_from_field_data(self):
        reader = _transient_reader()
        np.testing.assert_array_equal(reader.get_time_steps(), [0.0, 0.5])

    def test_extract_point_data_over_two_steps(self):
        reader = _transient_reader()
        result = reader.extract_point_data_over_time(
            "diffused", [(0.0, 0.0)], output_format="dict_numpy"
        )
        np.testing.assert_array_equal(result["point_0_node_0"], [0.0, 10.0])

    def test_waterfall_dict_numpy_keyed_by_time(self):
        reader = _transient_reader()
        result = reader.extract_line_data_for_waterfall(
            "diffused", (0.0, 0.0), (1.0, 0.0), 2, output_format="dict_numpy"
        )
        assert sorted(result.keys()) == [0.0, 0.5]
        np.testing.assert_array_equal(result[0.0], [0.0, 1.0])
        np.testing.assert_array_equal(result[0.5], [10.0, 11.0])

    def test_waterfall_list_of_arrays(self):
        reader = _transient_reader()
        result = reader.extract_line_data_for_waterfall(
            "diffused", (0.0, 0.0), (1.0, 0.0), 2, output_format="list_of_arrays"
        )
        assert isinstance(result, list)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# MoosePointSamplerSet -- pure CSV parsing
# ---------------------------------------------------------------------------

def _write_sampler_csv(path, header, rows):
    lines = [",".join(header)]
    for row in rows:
        lines.append(",".join(str(v) for v in row))
    path.write_text("\n".join(lines) + "\n")


class TestMoosePointSamplerSet:
    def test_loads_each_variable_column(self, tmp_path):
        _write_sampler_csv(
            tmp_path / "ps.csv",
            ["time", "ppA", "ppB"],
            [(0.0, 1.0, 4.0), (1.0, 2.0, 5.0), (2.0, 3.0, 6.0)],
        )
        s = MoosePointSamplerSet(str(tmp_path))
        assert sorted(s.get_sampler_names()) == ["ppA", "ppB"]

    def test_sampler_data_values(self, tmp_path):
        _write_sampler_csv(
            tmp_path / "ps.csv",
            ["time", "ppA", "ppB"],
            [(0.0, 1.0, 4.0), (1.0, 2.0, 5.0), (2.0, 3.0, 6.0)],
        )
        s = MoosePointSamplerSet(str(tmp_path))
        ppA = s.get_sampler("ppA")
        np.testing.assert_array_equal(ppA.taxis, [0.0, 1.0, 2.0])
        np.testing.assert_array_equal(ppA.data, [1.0, 2.0, 3.0])
        # Type used to hold the data.
        assert type(ppA).__name__ == "Data1D_MOOSEps"

    def test_get_sampler_missing_returns_none(self, tmp_path):
        _write_sampler_csv(tmp_path / "ps.csv", ["time", "ppA"], [(0.0, 1.0)])
        s = MoosePointSamplerSet(str(tmp_path))
        assert s.get_sampler("nonexistent") is None

    def test_numbered_vector_files_not_excluded_bug(self, tmp_path):
        # NOTE: possible bug. The exclusion regex is r'.*?_\\d+\.csv$' with a
        # doubled backslash, so '\\d' is treated as a literal backslash+d and
        # never matches digit-numbered vector-sampler files. As a result a
        # numbered file like 'vec_0000.csv' is NOT excluded and its columns are
        # loaded as point samplers.
        _write_sampler_csv(tmp_path / "ps.csv", ["time", "ppA"], [(0.0, 1.0)])
        _write_sampler_csv(tmp_path / "vec_0000.csv", ["id", "vvar"], [(0, 9.0)])
        s = MoosePointSamplerSet(str(tmp_path))
        # 'vvar' from the numbered file leaks in alongside 'ppA'.
        assert "vvar" in s.get_sampler_names()
        assert "ppA" in s.get_sampler_names()

    def test_missing_directory_yields_no_samplers(self, tmp_path):
        s = MoosePointSamplerSet(str(tmp_path / "does_not_exist"))
        assert s.get_sampler_names() == []

    def test_empty_directory_yields_no_samplers(self, tmp_path):
        s = MoosePointSamplerSet(str(tmp_path))
        assert s.get_sampler_names() == []

    def test_plot_all_samplers_saves_png(self, tmp_path):
        _write_sampler_csv(
            tmp_path / "ps.csv", ["time", "ppA"], [(0.0, 1.0), (1.0, 2.0)]
        )
        s = MoosePointSamplerSet(str(tmp_path))
        save_dir = tmp_path / "plots"
        s.plot_all_samplers(save_dir=str(save_dir), show_plots=False)
        assert os.path.exists(os.path.join(str(save_dir), "sampler_ppA.png"))

    def test_plot_all_samplers_no_samplers_noop(self, tmp_path):
        s = MoosePointSamplerSet(str(tmp_path))  # empty dir -> no samplers
        # Should simply return without error.
        s.plot_all_samplers(save_dir=str(tmp_path / "plots"))
        assert s.get_sampler_names() == []
