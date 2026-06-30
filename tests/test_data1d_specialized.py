# Characterization (golden-master) tests for the specialized Data1D subclasses:
#   - Data1DGauge        (Data1D_Gauge.py)
#   - Data1DPumpingCurve (Data1D_PumpingCurve.py)
#   - Data1D_MOOSEps     (Data1D_MOOSEps.py)
#
# These pin the CURRENT observable behavior of the subclass-specific methods
# plus verify that inherited Data1D behavior continues to work. Golden values
# were obtained by running the code, not by assuming what is "correct".

import datetime

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from fiberis.analyzer.Data1D.core1D import Data1D
from fiberis.analyzer.Data1D.Data1D_Gauge import Data1DGauge
from fiberis.analyzer.Data1D.Data1D_PumpingCurve import Data1DPumpingCurve
from fiberis.analyzer.Data1D.Data1D_MOOSEps import Data1D_MOOSEps


# ===========================================================================
# Data1DGauge
# ===========================================================================
class TestData1DGauge:
    @pytest.fixture
    def gauge(self, sample_start_time):
        return Data1DGauge(data=np.array([1.0, 2.0, 3.0]),
                           taxis=np.array([0.0, 1.0, 2.0]),
                           start_time=sample_start_time, name="gauge")

    def test_is_subclass_of_data1d(self):
        assert issubclass(Data1DGauge, Data1D)

    def test_instance_is_data1d(self, gauge):
        assert isinstance(gauge, Data1D)

    def test_calculate_pressure_dropdown_stub_returns_zero(self, gauge):
        # NOTE: documents current behavior; method is an unimplemented stub
        # that always returns 0 regardless of arguments.
        assert gauge.calculate_pressure_dropdown() == 0
        assert gauge.calculate_pressure_dropdown(start_time=0, end_time=1) == 0

    def test_inherits_crop(self, gauge, sample_start_time):
        gauge.crop(1.0, 2.0)
        assert_array_equal(gauge.taxis, np.array([0.0, 1.0]))
        assert_array_equal(gauge.data, np.array([2.0, 3.0]))
        assert gauge.start_time == sample_start_time + datetime.timedelta(seconds=1)

    def test_inherits_get_value_by_time(self, gauge):
        assert gauge.get_value_by_time(0.5) == 1.5

    def test_inherits_get_end_time(self, gauge, sample_start_time):
        assert gauge.get_end_time(use_timestamp=False) == 2.0


# ===========================================================================
# Data1DPumpingCurve
# ===========================================================================
class TestData1DPumpingCurve:
    @pytest.fixture
    def pc(self, sample_start_time):
        return Data1DPumpingCurve(data=np.array([0.0, 5.0, 10.0]),
                                  taxis=np.array([0.0, 2.0, 4.0]),
                                  start_time=sample_start_time, name="pc")

    def test_is_subclass_of_data1d(self):
        assert issubclass(Data1DPumpingCurve, Data1D)

    def test_get_start_time_default(self, pc, sample_start_time):
        # min_index is always 0, so start time == start_time + taxis[0] == start_time
        assert pc.get_start_time() == sample_start_time

    def test_get_start_time_with_threshold_kwarg_ignored(self, pc, sample_start_time):
        # NOTE: documents current behavior; the 'threshold' kwarg is read but
        # never used to select an index (min_index is hard-coded to 0).
        assert pc.get_start_time(threshold=999.0) == sample_start_time

    def test_get_start_time_no_start_time_raises(self):
        pc = Data1DPumpingCurve(data=np.array([0.0, 5.0]),
                                taxis=np.array([0.0, 2.0]))
        with pytest.raises(ValueError):
            pc.get_start_time()

    def test_get_end_time_datetime_default(self, pc, sample_start_time):
        # PumpingCurve overrides get_end_time with signature usedatetime=True
        assert pc.get_end_time() == sample_start_time + datetime.timedelta(seconds=4.0)

    def test_get_end_time_seconds(self, pc):
        # NOTE: documents current behavior; the seconds branch returns the raw
        # taxis element (a numpy float), unlike the base class which wraps it
        # in np.float64 and uses the 'use_timestamp' parameter name.
        assert pc.get_end_time(usedatetime=False) == 4.0

    def test_get_end_time_threshold_kwarg_ignored(self, pc, sample_start_time):
        # end_index is hard-coded to len(data)-1 regardless of threshold
        assert pc.get_end_time(threshold=999.0) == \
            sample_start_time + datetime.timedelta(seconds=4.0)

    def test_get_end_time_no_start_time_raises(self):
        pc = Data1DPumpingCurve(data=np.array([0.0, 5.0]),
                                taxis=np.array([0.0, 2.0]))
        with pytest.raises(ValueError):
            pc.get_end_time()

    def test_inherits_crop(self, pc, sample_start_time):
        pc.crop(2.0, 4.0)
        assert_array_equal(pc.data, np.array([5.0, 10.0]))
        assert pc.start_time == sample_start_time + datetime.timedelta(seconds=2)

    def test_inherits_copy(self, pc):
        c = pc.copy()
        assert isinstance(c, Data1DPumpingCurve)
        assert c is not pc


# ===========================================================================
# Data1D_MOOSEps
# ===========================================================================
class TestData1DMOOSEps:
    def test_is_subclass_of_data1d(self):
        assert issubclass(Data1D_MOOSEps, Data1D)

    def test_init_only_takes_name(self):
        m = Data1D_MOOSEps(name="moose")
        assert m.name == "moose"
        assert m.data is None
        assert m.taxis is None
        assert m.start_time is None

    def test_init_without_name(self):
        m = Data1D_MOOSEps()
        assert m.name is None

    def test_init_adds_history_records(self):
        # parent __init__ adds one record, the subclass adds another.
        m = Data1D_MOOSEps(name="moose")
        assert len(m.history.records) >= 2

    def test_inherits_load_npz(self, tmp_path):
        fn = tmp_path / "moose.npz"
        np.savez(fn, data=np.array([1.0, 2.0, 3.0]),
                 taxis=np.array([0.0, 1.0, 2.0]),
                 start_time=datetime.datetime(2023, 1, 1, 12, 0, 0))
        m = Data1D_MOOSEps(name="moose")
        m.load_npz(str(fn))
        assert m.data.size == 3
        # load_npz overrides name with the basename of the file
        assert m.name == "moose.npz"

    def test_inherits_processing_after_load(self, tmp_path, sample_start_time):
        fn = tmp_path / "moose2.npz"
        np.savez(fn, data=np.arange(10.0), taxis=np.arange(10.0),
                 start_time=sample_start_time)
        m = Data1D_MOOSEps(name="m")
        m.load_npz(str(fn))
        m.down_sample(2)
        assert m.data.size == 5
        assert m.get_value_by_time(4.0) == 4.0
