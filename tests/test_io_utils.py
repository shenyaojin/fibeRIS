# Characterization tests for fiberis.utils.io_utils.

import h5py
import numpy as np
import numpy.testing as npt
import pytest

from fiberis.utils import io_utils as io


# ---------------------------------------------------------------------------
# Fixtures: write small HDF5 / csvh files into tmp_path
# ---------------------------------------------------------------------------

@pytest.fixture
def das_h5(tmp_path):
    """An HDF5 file using the simple ('data', 'depth', ...) key set."""
    fn = str(tmp_path / "das.h5")
    with h5py.File(fn, "w") as f:
        f.create_dataset("data", data=np.arange(12).reshape(3, 4).astype(float))
        f.create_dataset("depth", data=np.array([1.0, 2.0, 3.0]))
        f.create_dataset("stamps_unix", data=np.array([100.0, 101.0, 102.0]))
        f.create_dataset("stamps", data=np.array([1000.0, 1001.0]))
    return fn


# ---------------------------------------------------------------------------
# read_h5
# ---------------------------------------------------------------------------

def test_read_h5_simple_keys(das_h5):
    data, daxis, taxis, start_time = io.read_h5(das_h5)
    assert data.shape == (3, 4)
    npt.assert_array_equal(daxis, [1.0, 2.0, 3.0])
    npt.assert_array_equal(taxis, [100.0, 101.0, 102.0])
    # start_time taken as the FIRST element of the 'stamps' dataset.
    npt.assert_allclose(start_time, 1000.0, atol=1e-12)


def test_read_h5_missing_keys_returns_none(tmp_path):
    fn = str(tmp_path / "empty.h5")
    with h5py.File(fn, "w") as f:
        f.create_dataset("foo", data=np.array([1.0, 2.0]))
    data, daxis, taxis, start_time = io.read_h5(fn)
    assert data is None
    assert daxis is None
    assert taxis is None
    assert start_time is None


def test_read_h5_nested_acquisition_keys(tmp_path):
    fn = str(tmp_path / "nested.h5")
    with h5py.File(fn, "w") as f:
        grp = f.create_group("Acquisition/Raw[0]")
        grp.create_dataset("RawData", data=np.ones((2, 2)))
        grp.create_dataset("RawDataTime", data=np.array([1.0, 2.0]))
    data, daxis, taxis, start_time = io.read_h5(fn)
    npt.assert_array_equal(data, np.ones((2, 2)))
    assert daxis is None  # no 'depth' key
    npt.assert_array_equal(taxis, [1.0, 2.0])


# ---------------------------------------------------------------------------
# load_h5_by_group_name
# ---------------------------------------------------------------------------

def test_load_h5_by_group_name_found(das_h5):
    out = io.load_h5_by_group_name(das_h5, "data")
    assert out.shape == (3, 4)


def test_load_h5_by_group_name_missing_returns_empty(das_h5, capsys):
    out = io.load_h5_by_group_name(das_h5, "does_not_exist")
    assert isinstance(out, np.ndarray)
    assert out.size == 0
    assert "not found" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# list_h5_keys
# ---------------------------------------------------------------------------

def test_list_h5_keys(das_h5):
    keys = io.list_h5_keys(das_h5)
    assert sorted(keys) == ["data", "depth", "stamps", "stamps_unix"]


# ---------------------------------------------------------------------------
# load_hfts2_depthtable
# ---------------------------------------------------------------------------

def test_load_hfts2_depthtable(tmp_path, capsys):
    fn = tmp_path / "dt.csvh"
    fn.write_text(
        "# comment\n"
        "~DATA\n"
        "header1\n"
        "header2\n"
        "1,-317.911398\n"
        "2,-317.0\n"
        "3,bad\n"
        "4,-315.5\n"
    )
    depths = io.load_hfts2_depthtable(str(fn))
    # Two header lines after ~DATA are skipped; the unparseable 'bad' row is
    # dropped with a warning, so only three valid depths remain.
    npt.assert_allclose(depths, [-317.911398, -317.0, -315.5], atol=1e-6)
    assert "Could not parse" in capsys.readouterr().out


def test_load_hfts2_depthtable_no_data_section(tmp_path):
    fn = tmp_path / "nodata.csvh"
    fn.write_text("# only comments\n# nothing else\n")
    depths = io.load_hfts2_depthtable(str(fn))
    assert isinstance(depths, np.ndarray)
    assert depths.size == 0
