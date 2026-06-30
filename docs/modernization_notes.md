# fibeRIS Modernization Notes

This document tracks the modernization effort for `fibeRIS`. The strategy is
**safety net first, refactor second**: build a comprehensive characterization
(golden-master) test suite that pins the *current* observable behavior, then
refactor and upgrade dependencies under its protection.

## Phase 1 — Test safety net (DONE)

A characterization suite was added across the whole package. Tests capture the
behavior the code *currently* exhibits (obtained by running the code, not by
guessing), so any behavior-preserving refactor keeps them green, while a fix to
one of the bugs below flips a specifically-marked test red on purpose.

- **Suite size:** 770 passing, 13 skipped (skips require unbundled vendor data).
- **Coverage:** ~70% of the package (branch coverage enabled), up from a near-zero
  baseline on most modules.
- **Tooling:** activated `pytest` / `coverage` / `mypy` config in `pyproject.toml`,
  added `tests/conftest.py` (shared fixtures, headless matplotlib), and a GitHub
  Actions CI workflow running the suite on Python 3.10–3.12.

Per-area coverage highlights: `analyzer` core classes 83–99%, `utils` 74–100%,
`simulator` 75%, `moose` config/editor 100% and `model_builder` 71%, `io/core`
92%.

## Bugs / quirks pinned by the suite (Phase 2 backlog)

These were discovered while writing characterization tests. Each is currently
pinned as "expected" behavior with a `# NOTE: possible bug` marker in the tests;
fixing them in Phase 2 means updating the corresponding test to assert the
corrected behavior.

### High priority — code that cannot run

1. **`TensorProcessor/coreT1D.py` is unimportable.** Uses `Tuple[...]`
   annotations but never imports `Tuple` (only `Optional, Union, List, Any,
   Dict`). `import` raises `NameError` at class-definition time, so `Tensor1D`
   can never be instantiated.
2. **`io/reader_mariner_3d.py` fails to import** — missing `DataG3D`.
3. **`io/reader_mariner_gauge1d.py` fails to import** — missing `Data1DGauge`
   from `fiberis.analyzer`.
4. **`io/reader_mariner_rfs_abandoned.py` (`Mariner2DRFS2D`)** does not implement
   the abstract `to_analyzer`, so it cannot be instantiated (`TypeError`).

### Medium priority — incorrect results

5. **`Tensor2D.rotate_tensor` computes `R·T·R` instead of `R·T·Rᵀ`.** The einsum
   passes `R.T` under the wrong subscript; rotating `[[1,0],[0,0]]` by 90° yields
   `[[0,0],[0,-1]]` instead of the correct `[[0,0],[0,1]]`. (`CoreTensor.rotate_tensor`
   does it correctly — they should be unified.)
6. **`postprocessor.py` exclusion regex `r'.*?_\\d+\.csv$'`** has a doubled
   backslash, so `\d` is treated literally and never matches. Numbered
   vector-sampler files (`*_0000.csv`) are not excluded and get mis-loaded as
   point samplers.
7. **`Neumann` sign-convention mismatch** between `simulator/core/bcs.py`
   (`BoundaryCondition.apply`: diag +1 / neighbour −1) and
   `simulator/solver/matbuilder.py` (inline: diag −1 / neighbour +1). Only the
   matbuilder path is live; `BoundaryCondition` is currently unused by the solver.

### Lower priority — robustness / API consistency

8. **`Data1DPumpingCurve.get_start_time` / `get_end_time` ignore their
   `threshold` kwarg** (hard-coded indices), and `get_end_time` diverges from the
   base class signature (`usedatetime` vs `use_timestamp`, different return type).
9. **`get_value_by_time` on a single-point series** returns that point's value for
   any query time (no real interpolation/extrapolation), contrary to the docstring.
10. **`Data1DGauge.calculate_pressure_dropdown`** is an unimplemented stub that
    always returns `0`.
11. **`Data3D.load_npz` stringifies `name`/`variable_name`** with `str(...)`, so a
    `None` round-trips as the literal string `'None'`.
12. **`Tensor2D.set_data` / `Data2D.__copy__` share the input array by reference**
    (no defensive copy), unlike their `CoreTensor` / `.copy()` counterparts.
13. **`PostprocessorConfig` mutates the caller's `params` dict in place** (`.pop`),
    and `PostprocessorConfigBase` uses a mutable default argument.
14. **`build_mesh_for_casing_model` + `generate_input_file`** both call
    `_finalize_mesh_block_renaming()`, appending a duplicate `RenameBlockGenerator`.
15. `signal_utils` helpers with quirky edge-case conventions: `correlation_coefficient`
    returns numbers (not NaN) for degenerate inputs; `xcor_match` is annotated
    "TEST FAILED. DO NOT USE IT" in-source; `timeshift_xcor` sign convention is
    documented as uncertain.

## Phase 2 — Architecture modernization (proposed, not started)

Under the protection of the suite:

- Fix the high/medium-priority defects above (each has a pinned test ready to flip).
- Reduce duplication across the `core<Dim>` classes (shared base for history,
  npz I/O, naming, info/str, copy) — the philosophy doc already calls for this.
- Modernize type hints (`Optional`/`Union`/`List` → `X | Y` and builtin generics,
  `from __future__ import annotations`).
- Replace `black` + `flake8` with `ruff`; wire `ruff` + `mypy` into CI.

## Phase 3 — Dependency / Python upgrade (proposed, not started)

- Raise `requires-python` to `>=3.10`.
- Open up `numpy>=2.0` and `pandas` upper bounds, validating against the suite
  (CI matrix already spans 3.10–3.12).
