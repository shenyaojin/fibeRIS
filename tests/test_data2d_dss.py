# Characterization tests for fiberis.analyzer.Data2D.Data2D_XT_DSS.DSS2D
#
# DSS2D subclasses Data2D and adds simple_plot() and downsampling().
# Golden values were obtained by running the code. Tests must stay green
# through a pure refactor.

import datetime
import os

import numpy as np
import pytest
from numpy.testing import assert_array_equal

import matplotlib.pyplot as plt

from fiberis.analyzer.Data2D.Data2D_XT_DSS import DSS2D
from fiberis.analyzer.Data2D.core2D import Data2D


@pytest.fixture
def dss():
    """Deterministic 6-depth x 8-time DSS2D (data = 0..47 reshaped)."""
    data = np.arange(48, dtype=float).reshape(6, 8)
    taxis = np.arange(8, dtype=float)
    daxis = np.arange(6, dtype=float) * 10.0
    return DSS2D(data=data, taxis=taxis, daxis=daxis,
                 start_time=datetime.datetime(2023, 1, 1), name="dss")


class TestDSS2DInheritance:
    def test_is_data2d_subclass(self, dss):
        assert isinstance(dss, Data2D)

    def test_inherited_methods_work(self, dss):
        vals, t = dss.get_value_by_time(3.0)
        assert t == 3.0
        assert dss.get_max_taxis() == 7.0


class TestDownsampling:
    def test_downsampling_default(self, dss):
        # default factor_t=2, factor_d=2 -> stride slicing.
        dss.downsampling()
        assert dss.data.shape == (3, 4)
        assert_array_equal(dss.taxis, np.array([0.0, 2.0, 4.0, 6.0]))
        assert_array_equal(dss.daxis, np.array([0.0, 20.0, 40.0]))

    def test_downsampling_custom_factors(self, dss):
        dss.downsampling(factor_t=2, factor_d=3)
        assert dss.data.shape == (2, 4)
        assert_array_equal(dss.taxis, np.array([0.0, 2.0, 4.0, 6.0]))
        assert_array_equal(dss.daxis, np.array([0.0, 30.0]))
        # data[::3, ::2]: rows 0 and 3, columns 0,2,4,6.
        assert_array_equal(dss.data[0], np.array([0.0, 2.0, 4.0, 6.0]))
        assert_array_equal(dss.data[1], np.array([24.0, 26.0, 28.0, 30.0]))

    def test_downsampling_records_history(self, dss):
        n_before = len(dss.history.records)
        dss.downsampling()
        assert len(dss.history.records) == n_before + 1


class TestSimplePlot:
    def teardown_method(self):
        plt.close("all")

    def test_simple_plot_runs(self, dss):
        dss.simple_plot()  # no raise (Agg backend; plt.show is a no-op)
        assert plt.get_fignums()  # a figure was created

    def test_simple_plot_custom_kwargs(self, dss):
        dss.simple_plot(cmap="plasma", title="custom", xlabel="t", ylabel="d")
        assert plt.get_fignums()


class TestDSSLoadRealExample:
    def test_load_npz_into_dss(self, examples_data_dir):
        path = os.path.join(examples_data_dir, "2d", "fiberis_format",
                            "DASdata_example.npz")
        d = DSS2D()
        d.load_npz(path)
        assert d.data.shape == (6100, 258)
        assert isinstance(d, Data2D)
