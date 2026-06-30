# Characterization tests for fiberis.io.core.DataIO.
#
# These lock in the current observable behavior of the abstract base reader
# class (constructor-initialized attributes, the manual setters, and the
# logging integration with InfoManagementSystem). They must remain GREEN
# through a pure refactor.

import datetime

import numpy as np
import pytest

from fiberis.io import core
from fiberis.utils.history_utils import InfoManagementSystem


class _ConcreteIO(core.DataIO):
    """Minimal concrete subclass so the ABC can be instantiated for testing."""

    def read(self, **kwargs):  # pragma: no cover - trivial stub
        return None

    def write(self, filename, *args):  # pragma: no cover - trivial stub
        return None

    def to_analyzer(self):  # pragma: no cover - trivial stub
        return None


def test_dataio_is_abstract_cannot_instantiate_directly():
    # DataIO declares abstract methods read/write/to_analyzer, so direct
    # instantiation must raise TypeError.
    with pytest.raises(TypeError):
        core.DataIO()


def test_constructor_initializes_all_axes_to_none():
    obj = _ConcreteIO()
    assert obj.daxis is None
    assert obj.taxis is None
    assert obj.data is None
    assert obj.start_time is None
    assert obj.filename is None
    # 3D geometry handling attributes
    assert obj.xaxis is None
    assert obj.yaxis is None
    assert obj.zaxis is None


def test_constructor_creates_log_system():
    obj = _ConcreteIO()
    assert isinstance(obj.log_system, InfoManagementSystem)
    # A fresh log system starts with no records.
    assert obj.log_system.get_records() == []


def test_set_daxis_sets_value_and_records_log():
    obj = _ConcreteIO()
    daxis = np.array([0.0, 1.0, 2.0])
    obj.set_daxis(daxis)
    np.testing.assert_array_equal(obj.daxis, daxis)

    records = obj.log_system.get_records()
    assert len(records) == 1
    assert records[0]["description"] == "daxis is set."
    assert records[0]["level"] == "INFO"


def test_set_taxis_sets_value_and_records_log():
    obj = _ConcreteIO()
    taxis = np.linspace(0, 10, 11)
    obj.set_taxis(taxis)
    np.testing.assert_array_equal(obj.taxis, taxis)
    records = obj.log_system.get_records()
    assert records[-1]["description"] == "taxis is set."


def test_set_data_sets_value_and_records_log():
    obj = _ConcreteIO()
    data = np.arange(6).reshape(2, 3)
    obj.set_data(data)
    np.testing.assert_array_equal(obj.data, data)
    records = obj.log_system.get_records()
    assert records[-1]["description"] == "data is set."


def test_set_start_time_sets_value_and_records_log(sample_start_time):
    obj = _ConcreteIO()
    obj.set_start_time(sample_start_time)
    assert obj.start_time == sample_start_time
    records = obj.log_system.get_records()
    assert records[-1]["description"] == "start_time is set."


def test_record_log_joins_args_with_spaces():
    obj = _ConcreteIO()
    obj.record_log("hello", "world", 42)
    records = obj.log_system.get_records()
    assert records[-1]["description"] == "hello world 42"
    assert records[-1]["level"] == "INFO"


def test_record_log_respects_explicit_level():
    obj = _ConcreteIO()
    obj.record_log("a problem", level="ERROR")
    records = obj.log_system.get_records()
    assert records[-1]["level"] == "ERROR"


def test_record_log_record_has_timestamp_datetime():
    obj = _ConcreteIO()
    obj.record_log("ts check")
    record = obj.log_system.get_records()[-1]
    assert isinstance(record["timestamp"], datetime.datetime)


def test_multiple_setters_accumulate_records_in_order():
    obj = _ConcreteIO()
    obj.set_daxis(np.array([1.0]))
    obj.set_taxis(np.array([2.0]))
    obj.set_data(np.array([3.0]))
    descriptions = [r["description"] for r in obj.log_system.get_records()]
    assert descriptions == ["daxis is set.", "taxis is set.", "data is set."]


def test_print_log_runs_without_error(capsys):
    obj = _ConcreteIO()
    obj.record_log("message for printing")
    obj.print_log()
    captured = capsys.readouterr()
    # We only pin that it produced output mentioning the recorded message;
    # exact formatting is owned by InfoManagementSystem.
    assert "message for printing" in captured.out
