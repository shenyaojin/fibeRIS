# Characterization tests for fiberis.utils.history_utils.InfoManagementSystem.
#
# Pins the current behavior of the operation-logging system: record storage,
# level normalization, filtering, printing, file persistence, and dunders.

import datetime
import os

import pytest

from fiberis.utils.history_utils import InfoManagementSystem


@pytest.fixture
def ims():
    return InfoManagementSystem()


# ---------------------------------------------------------------------------
# Construction / basic storage
# ---------------------------------------------------------------------------

def test_starts_empty(ims):
    assert ims.records == []
    assert ims.get_records() == []


def test_add_record_default_level(ims):
    ims.add_record("hello")
    assert len(ims.records) == 1
    rec = ims.records[0]
    assert rec["description"] == "hello"
    assert rec["level"] == "INFO"
    assert isinstance(rec["timestamp"], datetime.datetime)


def test_add_record_level_uppercased(ims):
    ims.add_record("a", level="warning")
    ims.add_record("b", level="Error")
    assert ims.records[0]["level"] == "WARNING"
    assert ims.records[1]["level"] == "ERROR"


def test_add_record_nonstring_description_coerced(ims):
    ims.add_record(12345)
    assert ims.records[0]["description"] == "12345"


def test_add_record_nonstring_level_coerced_and_uppercased(ims):
    ims.add_record("x", level=42)
    # NOTE: non-string level path applies str(...).upper().
    assert ims.records[0]["level"] == "42"


def test_records_appended_in_order(ims):
    ims.add_record("first")
    ims.add_record("second")
    ims.add_record("third")
    descs = [r["description"] for r in ims.get_records()]
    assert descs == ["first", "second", "third"]


# ---------------------------------------------------------------------------
# get_records / filtering
# ---------------------------------------------------------------------------

def test_get_records_returns_copy(ims):
    ims.add_record("a")
    got = ims.get_records()
    got.append("garbage")
    # Mutating the returned list must not affect internal storage.
    assert len(ims.records) == 1


def test_get_records_filter_by_level(ims):
    ims.add_record("info1", level="INFO")
    ims.add_record("warn1", level="WARNING")
    ims.add_record("info2", level="INFO")
    infos = ims.get_records(lambda r: r["level"] == "INFO")
    assert len(infos) == 2
    assert {r["description"] for r in infos} == {"info1", "info2"}


def test_get_records_filter_no_match(ims):
    ims.add_record("a", level="INFO")
    assert ims.get_records(lambda r: r["level"] == "ERROR") == []


# ---------------------------------------------------------------------------
# print_records
# ---------------------------------------------------------------------------

def test_print_records_empty(ims, capsys):
    ims.print_records()
    out = capsys.readouterr().out
    assert "No records to display." in out


def test_print_records_format(ims, capsys):
    ims.add_record("the message", level="WARNING")
    ims.print_records()
    out = capsys.readouterr().out
    assert "[WARNING]" in out
    assert "the message" in out


def test_print_records_with_filter(ims, capsys):
    ims.add_record("keep", level="ERROR")
    ims.add_record("drop", level="INFO")
    ims.print_records(lambda r: r["level"] == "ERROR")
    out = capsys.readouterr().out
    assert "keep" in out
    assert "drop" not in out


# ---------------------------------------------------------------------------
# save_records_to_txt
# ---------------------------------------------------------------------------

def test_save_records_empty_returns_empty_string(ims, capsys):
    result = ims.save_records_to_txt()
    assert result == ""
    assert "No records to save." in capsys.readouterr().out


def test_save_records_to_directory(ims, tmp_path):
    ims.add_record("saved message", level="INFO")
    path = ims.save_records_to_txt(str(tmp_path))
    assert path != ""
    assert os.path.isfile(path)
    assert os.path.dirname(path) == str(tmp_path)
    content = open(path).read()
    assert "saved message" in content
    assert "Level:     INFO" in content
    assert "-" * 50 in content


def test_save_records_to_explicit_filepath(ims, tmp_path):
    ims.add_record("explicit", level="DEBUG")
    target = str(tmp_path / "my_log.txt")
    path = ims.save_records_to_txt(target)
    assert path == target
    assert os.path.isfile(target)
    assert "explicit" in open(target).read()


def test_save_records_with_filter(ims, tmp_path):
    ims.add_record("err", level="ERROR")
    ims.add_record("inf", level="INFO")
    target = str(tmp_path / "filtered.txt")
    ims.save_records_to_txt(target, filter_fn=lambda r: r["level"] == "ERROR")
    content = open(target).read()
    assert "err" in content
    assert "inf" not in content


# ---------------------------------------------------------------------------
# clear_records
# ---------------------------------------------------------------------------

def test_clear_records(ims):
    ims.add_record("a")
    ims.add_record("b")
    ims.clear_records()
    assert ims.records == []
    assert ims.get_records() == []


# ---------------------------------------------------------------------------
# __repr__ / __str__
# ---------------------------------------------------------------------------

def test_repr(ims):
    assert repr(ims) == "InfoManagementSystem(records_count=0)"
    ims.add_record("a")
    assert repr(ims) == "InfoManagementSystem(records_count=1)"


def test_str_empty(ims):
    assert str(ims) == "No records in history."


def test_str_with_records(ims):
    ims.add_record("hello world", level="INFO")
    s = str(ims)
    assert s.startswith("History Log (1 records):")
    assert "[INFO]" in s
    assert "hello world" in s
