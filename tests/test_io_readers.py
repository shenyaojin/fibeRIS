# Characterization tests for the fiberis.io reader_*.py modules.
#
# Strategy:
#   * Readers whose raw input format matches a bundled fixture are exercised on
#     the real read path and their output shapes/dtypes/values are pinned.
#       - reader_hfts2_h5.HFTS2DAS2D        -> examples/data/2d/original_data/*.h5
#       - reader_mariner_dssh5.MarinerDSS2D -> same h5 (read currently FAILS; pinned)
#       - reader_MOOSEcsv_pp1d.MOOSEcsv_pp1d -> synthesized MOOSE CSV
#       - reader_moose_ps.MOOSEPointSamplerReader -> synthesized MOOSE CSV
#       - reader_moose_vpp.MOOSEVectorPostProcessorReader -> synthesized VPP CSVs
#   * Readers that require vendor raw data we do not have get an import/smoke
#     test plus a skip for the read path.
#
# These must remain GREEN through a pure refactor.

import datetime
import importlib
import os

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _h5_dir(examples_data_dir):
    return os.path.join(examples_data_dir, "2d", "original_data")


# ---------------------------------------------------------------------------
# HFTS2 DAS h5 reader -- REAL DATA read path
# ---------------------------------------------------------------------------

class TestHFTS2DAS2DRealData:
    def _read(self, examples_data_dir):
        from fiberis.io.reader_hfts2_h5 import HFTS2DAS2D

        reader = HFTS2DAS2D()
        reader.read(_h5_dir(examples_data_dir))
        return reader

    def test_to_analyzer_shapes_and_dtype(self, examples_data_dir):
        reader = self._read(examples_data_dir)
        ana = reader.to_analyzer()
        # DSS2D analyzer object.
        assert type(ana).__name__ == "DSS2D"
        # 9 h5 files, each (6100, 30); merged along time -> (6100, 258).
        assert ana.data.shape == (6100, 258)
        assert ana.data.dtype == np.int32
        assert ana.daxis.shape == (6100,)
        assert ana.taxis.shape == (258,)

    def test_daxis_from_depth_table(self, examples_data_dir):
        ana = self._read(examples_data_dir).to_analyzer()
        # daxis comes from the calibrated HFTS2 depth table, not the h5 file.
        np.testing.assert_allclose(ana.daxis[0], -318.93235)
        np.testing.assert_allclose(ana.daxis[-1], 5881.690674)

    def test_taxis_is_relative_starting_at_zero(self, examples_data_dir):
        ana = self._read(examples_data_dir).to_analyzer()
        assert ana.taxis[0] == 0.0
        np.testing.assert_allclose(ana.taxis[-1], 2587.14)

    def test_start_time_is_utc_from_first_chunk(self, examples_data_dir):
        ana = self._read(examples_data_dir).to_analyzer()
        assert ana.start_time == datetime.datetime(
            2019, 3, 28, 16, 4, 19, 700000, tzinfo=datetime.timezone.utc
        )

    def test_pinned_data_values(self, examples_data_dir):
        ana = self._read(examples_data_dir).to_analyzer()
        assert ana.data[0, 0] == 49986
        assert int(ana.data.sum()) == -3576841782

    def test_read_missing_folder_returns_quietly(self, tmp_path):
        from fiberis.io.reader_hfts2_h5 import HFTS2DAS2D

        reader = HFTS2DAS2D()
        # No .h5 files -> reader prints a warning and returns; temp_dssobject
        # stays the empty default DSS2D constructed in __init__.
        reader.read(str(tmp_path))
        assert reader.temp_dssobject.data is None


def test_hfts2curve_is_constructible_but_unimplemented():
    # HFTS2CURVE.read/write/to_analyzer are stubs (pass). Pin that the class
    # is importable/constructible and that read() returns None.
    from fiberis.io.reader_hfts2_h5 import HFTS2CURVE

    curve = HFTS2CURVE()
    assert curve.read("anything", "var") is None
    assert curve.to_analyzer() is None


# ---------------------------------------------------------------------------
# Mariner single-file DSS h5 reader -- REAL DATA read path (currently fails)
# ---------------------------------------------------------------------------

class TestMarinerDSS2DRealData:
    def _h5_file(self, examples_data_dir):
        return os.path.join(_h5_dir(examples_data_dir), "sensor_2019-03-28T160419Z.h5")

    def test_read_raises_on_missing_start_time_metadata(self, examples_data_dir):
        # NOTE: possible bug / data limitation. read_h5 returns start_time=None
        # for these HFTS-style h5 files (no 'stamps' key), and MarinerDSS2D.read
        # unconditionally calls start_time.decode(), raising AttributeError.
        from fiberis.io.reader_mariner_dssh5 import MarinerDSS2D

        reader = MarinerDSS2D()
        with pytest.raises(AttributeError):
            reader.read(self._h5_file(examples_data_dir))

    def test_partial_state_before_failure(self, examples_data_dir):
        # Before the start_time conversion fails, taxis/daxis/data are assigned.
        from fiberis.io.reader_mariner_dssh5 import MarinerDSS2D

        reader = MarinerDSS2D()
        try:
            reader.read(self._h5_file(examples_data_dir))
        except AttributeError:
            pass
        assert reader.data is not None
        assert reader.data.shape == (6100, 30)
        assert reader.taxis is not None
        assert reader.taxis[0] == 0.0


# ---------------------------------------------------------------------------
# MOOSE CSV 1D reader -- synthesized faithful CSV
# ---------------------------------------------------------------------------

def _write_moose_point_sampler_csv(path):
    """A MOOSE PointSampler-style CSV: a 'time' column plus variable columns."""
    path.write_text(
        "time,pp_mon1,pp_mon2\n"
        "0.0,10.0,20.0\n"
        "1.0,11.0,21.0\n"
        "2.0,12.0,22.0\n"
    )
    return str(path)


class TestMOOSEcsvPP1D:
    def test_list_available_keys(self, tmp_path):
        from fiberis.io.reader_MOOSEcsv_pp1d import MOOSEcsv_pp1d

        csv = _write_moose_point_sampler_csv(tmp_path / "sampler_out.csv")
        keys = MOOSEcsv_pp1d().list_available_keys(csv)
        # 'time' is excluded from the returned keys.
        assert keys == ["pp_mon1", "pp_mon2"]

    def test_read_selects_named_column(self, tmp_path):
        from fiberis.io.reader_MOOSEcsv_pp1d import MOOSEcsv_pp1d

        csv = _write_moose_point_sampler_csv(tmp_path / "sampler_out.csv")
        reader = MOOSEcsv_pp1d()
        reader.read(csv, key="pp_mon2")
        np.testing.assert_array_equal(reader.taxis, [0.0, 1.0, 2.0])
        np.testing.assert_array_equal(reader.data, [20.0, 21.0, 22.0])
        assert reader.label == "pp_mon2"

    def test_default_start_time(self, tmp_path):
        from fiberis.io.reader_MOOSEcsv_pp1d import MOOSEcsv_pp1d

        # start_time defaults to 2024-01-01 (arbitrary simulation time).
        assert MOOSEcsv_pp1d().start_time == datetime.datetime(2024, 1, 1)

    def test_read_missing_key_raises(self, tmp_path):
        from fiberis.io.reader_MOOSEcsv_pp1d import MOOSEcsv_pp1d

        csv = _write_moose_point_sampler_csv(tmp_path / "sampler_out.csv")
        with pytest.raises(ValueError):
            MOOSEcsv_pp1d().read(csv, key="does_not_exist")

    def test_read_none_key_raises(self, tmp_path):
        from fiberis.io.reader_MOOSEcsv_pp1d import MOOSEcsv_pp1d

        csv = _write_moose_point_sampler_csv(tmp_path / "sampler_out.csv")
        with pytest.raises(ValueError):
            MOOSEcsv_pp1d().read(csv, key=None)

    def test_to_analyzer(self, tmp_path):
        from fiberis.io.reader_MOOSEcsv_pp1d import MOOSEcsv_pp1d

        csv = _write_moose_point_sampler_csv(tmp_path / "sampler_out.csv")
        reader = MOOSEcsv_pp1d()
        reader.read(csv, key="pp_mon1")
        ana = reader.to_analyzer()
        assert type(ana).__name__ == "Data1D"
        assert ana.name == "pp_mon1"
        np.testing.assert_array_equal(ana.data, [10.0, 11.0, 12.0])

    def test_write_then_roundtrip_npz(self, tmp_path):
        from fiberis.io.reader_MOOSEcsv_pp1d import MOOSEcsv_pp1d

        csv = _write_moose_point_sampler_csv(tmp_path / "sampler_out.csv")
        reader = MOOSEcsv_pp1d()
        reader.read(csv, key="pp_mon2")
        out = str(tmp_path / "out")  # no .npz extension -> appended
        reader.write(out)
        loaded = np.load(out + ".npz", allow_pickle=True)
        np.testing.assert_array_equal(loaded["data"], [20.0, 21.0, 22.0])
        assert str(loaded["label"]) == "pp_mon2"


# ---------------------------------------------------------------------------
# MOOSE Point Sampler reader (folder-based auto-discovery)
# ---------------------------------------------------------------------------

class TestMOOSEPointSamplerReader:
    def test_read_auto_discovers_and_loads(self, tmp_path):
        from fiberis.io.reader_moose_ps import MOOSEPointSamplerReader

        _write_moose_point_sampler_csv(tmp_path / "sampler_out.csv")
        reader = MOOSEPointSamplerReader()
        reader.read(str(tmp_path), variable_index=1)
        np.testing.assert_array_equal(reader.taxis, [0.0, 1.0, 2.0])
        np.testing.assert_array_equal(reader.data, [10.0, 11.0, 12.0])
        assert reader.variable_name == "pp_mon1"

    def test_get_max_index(self, tmp_path):
        from fiberis.io.reader_moose_ps import MOOSEPointSamplerReader

        _write_moose_point_sampler_csv(tmp_path / "sampler_out.csv")
        # 2 data columns -> max index 2.
        assert MOOSEPointSamplerReader().get_max_index(str(tmp_path)) == 2

    def test_read_ambiguous_when_two_unnumbered_csvs(self, tmp_path):
        from fiberis.io.reader_moose_ps import MOOSEPointSamplerReader

        _write_moose_point_sampler_csv(tmp_path / "a.csv")
        _write_moose_point_sampler_csv(tmp_path / "b.csv")
        # More than one candidate sampler file -> FileNotFoundError.
        with pytest.raises(FileNotFoundError):
            MOOSEPointSamplerReader().read(str(tmp_path))

    def test_read_out_of_bounds_index_raises(self, tmp_path):
        from fiberis.io.reader_moose_ps import MOOSEPointSamplerReader

        _write_moose_point_sampler_csv(tmp_path / "sampler_out.csv")
        # The internal IndexError is caught and re-raised as ValueError by the
        # broad except in read().
        with pytest.raises(ValueError):
            MOOSEPointSamplerReader().read(str(tmp_path), variable_index=99)

    def test_to_analyzer(self, tmp_path):
        from fiberis.io.reader_moose_ps import MOOSEPointSamplerReader

        _write_moose_point_sampler_csv(tmp_path / "sampler_out.csv")
        reader = MOOSEPointSamplerReader()
        reader.read(str(tmp_path), variable_index=2)
        ana = reader.to_analyzer()
        assert type(ana).__name__ == "Data1D_MOOSEps"
        assert ana.name == "pp_mon2"
        np.testing.assert_array_equal(ana.data, [20.0, 21.0, 22.0])


# ---------------------------------------------------------------------------
# MOOSE VectorPostProcessor reader (CSV series)
# ---------------------------------------------------------------------------

def _write_moose_vpp_series(folder):
    """A MOOSE VectorPostProcessor-style series.

    The time-series file is the single non-numbered CSV; each numbered file
    holds one timestep with columns [id, var, x, y, z].
    """
    (folder / "run_out.csv").write_text("time\n0.0\n1.0\n2.0\n")
    for i in range(3):
        (folder / f"line_sampler_{i:04d}.csv").write_text(
            "id,pressure,x,y,z\n"
            f"0,{100.0 + i},0.0,0.0,0.0\n"
            f"1,{200.0 + i},1.0,0.0,0.0\n"
            f"2,{300.0 + i},2.0,0.0,0.0\n"
        )


class TestMOOSEVectorPostProcessorReader:
    def test_get_max_indices(self, tmp_path):
        from fiberis.io.reader_moose_vpp import MOOSEVectorPostProcessorReader

        _write_moose_vpp_series(tmp_path)
        # One sampler -> processor id 0; columns id,pressure,x,y,z (5) -> var idx 1.
        assert MOOSEVectorPostProcessorReader().get_max_indices(str(tmp_path)) == (0, 1)

    def test_read_assembles_space_by_time_matrix(self, tmp_path):
        from fiberis.io.reader_moose_vpp import MOOSEVectorPostProcessorReader

        _write_moose_vpp_series(tmp_path)
        reader = MOOSEVectorPostProcessorReader()
        reader.read(str(tmp_path), post_processor_id=0, variable_index=1)
        np.testing.assert_array_equal(reader.taxis, [0.0, 1.0, 2.0])
        # data is stacked along axis=1 -> (n_space, n_time) = (3, 3).
        assert reader.data.shape == (3, 3)
        np.testing.assert_array_equal(
            reader.data,
            [[100.0, 101.0, 102.0],
             [200.0, 201.0, 202.0],
             [300.0, 301.0, 302.0]],
        )
        assert reader.sampler_name == "line_sampler"
        assert reader.variable_name == "pressure"

    def test_daxis_is_cumulative_distance(self, tmp_path):
        from fiberis.io.reader_moose_vpp import MOOSEVectorPostProcessorReader

        _write_moose_vpp_series(tmp_path)
        reader = MOOSEVectorPostProcessorReader()
        reader.read(str(tmp_path))
        # daxis = sqrt((x-x0)^2 + (y-y0)^2) with x=[0,1,2], y=0.
        np.testing.assert_allclose(reader.daxis, [0.0, 1.0, 2.0])

    def test_read_no_samplers_raises(self, tmp_path):
        from fiberis.io.reader_moose_vpp import MOOSEVectorPostProcessorReader

        (tmp_path / "run_out.csv").write_text("time\n0.0\n")
        with pytest.raises(FileNotFoundError):
            MOOSEVectorPostProcessorReader().read(str(tmp_path))

    def test_to_analyzer(self, tmp_path):
        from fiberis.io.reader_moose_vpp import MOOSEVectorPostProcessorReader

        _write_moose_vpp_series(tmp_path)
        reader = MOOSEVectorPostProcessorReader()
        reader.read(str(tmp_path))
        ana = reader.to_analyzer()
        assert type(ana).__name__ == "Data2D"
        assert ana.name == "line_sampler_pressure"


# ---------------------------------------------------------------------------
# Smoke / import tests for readers that need vendor raw data we do not have.
# ---------------------------------------------------------------------------

# (module_name, class_name) pairs that import cleanly and are constructible.
_CONSTRUCTIBLE_VENDOR_READERS = [
    ("reader_bearskin_injection", "BearskinInjection"),
    ("reader_bearskin_pp1d", "BearskinPP1D"),
    ("reader_gold4pb_3d", "Gold4PB3D"),
    ("reader_gold4pb_projection", "Gold4PBProjection"),
    ("reader_mariner_das2d", "MarinerDAS2D"),
    ("reader_mariner_fiberdata_production_dat2d", "MarinerDSSdat2D"),
    ("reader_mariner_pp1d", "MarinerPP1D"),
    ("reader_mariner_pressureg1", "MarinerPressureG1"),
    ("reader_moose_tensor_from_data2d", "MOOSETensorFromData2D"),
    ("reader_moose_tensor_vpp", "MOOSETensorVPPReader"),
]


@pytest.mark.parametrize("module_name,class_name", _CONSTRUCTIBLE_VENDOR_READERS)
def test_vendor_reader_importable_and_constructible(module_name, class_name):
    module = importlib.import_module(f"fiberis.io.{module_name}")
    cls = getattr(module, class_name)
    instance = cls()
    # All readers descend from DataIO and expose the standard attributes.
    from fiberis.io.core import DataIO

    assert isinstance(instance, DataIO)
    assert hasattr(instance, "read")
    pytest.skip(f"requires raw vendor data not bundled for {module_name}")


def test_mariner_rfs_abandoned_is_abstract():
    # NOTE: the 'abandoned' RFS reader does not implement the abstract
    # to_analyzer() method, so it cannot be instantiated. Pin this current
    # (intentionally abandoned) state.
    from fiberis.io.reader_mariner_rfs_abandoned import Mariner2DRFS2D

    with pytest.raises(TypeError):
        Mariner2DRFS2D()
    pytest.skip("reader_mariner_rfs_abandoned is abstract; requires raw vendor data not bundled")


# These two modules currently fail to import because of missing names in the
# fiberis.analyzer package. NOTE: possible bug -- the imported symbols
# (DataG3D / Data1DGauge) are not exported. Pinned as expected failures.
@pytest.mark.parametrize(
    "module_name,missing_symbol",
    [
        ("reader_mariner_3d", "DataG3D"),
        ("reader_mariner_gauge1d", "Data1DGauge"),
    ],
)
def test_reader_with_broken_import(module_name, missing_symbol):
    with pytest.raises(ImportError):
        importlib.import_module(f"fiberis.io.{module_name}")
    pytest.skip(
        f"{module_name} currently fails to import (missing {missing_symbol}); "
        "requires raw vendor data not bundled anyway"
    )
