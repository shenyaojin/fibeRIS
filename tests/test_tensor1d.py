"""Characterization tests for fiberis.analyzer.TensorProcessor.coreT1D.

IMPORTANT CURRENT-BEHAVIOR NOTE:
    coreT1D.py uses the type hint ``Tuple[int, int]`` (lines 122 and 155) but
    only imports ``Optional, Union, List, Any, Dict`` from ``typing`` -- it does
    NOT import ``Tuple``. Because the annotation is evaluated at class-definition
    time, the *module itself fails to import* with a ``NameError`` before the
    ``Tensor1D`` class can ever be constructed.

    This is a genuine bug. Per the characterization-test contract we pin the
    CURRENT observable behavior (import fails) rather than the intended behavior.
    When the bug is fixed (e.g. by adding ``Tuple`` to the typing import), the
    ``test_module_currently_fails_to_import`` test below will start failing and
    should be replaced with real behavioral tests of ``Tensor1D``.
"""

import importlib

import pytest


def test_module_currently_fails_to_import():
    # NOTE: possible bug -- `Tuple` is used in annotations but never imported,
    # so importing the module raises NameError at class-definition time.
    with pytest.raises(NameError) as excinfo:
        importlib.import_module("fiberis.analyzer.TensorProcessor.coreT1D")
    assert "Tuple" in str(excinfo.value)


def test_tensor1d_class_is_unimportable():
    # The class cannot be imported either, for the same reason.
    with pytest.raises(NameError):
        from fiberis.analyzer.TensorProcessor.coreT1D import Tensor1D  # noqa: F401
