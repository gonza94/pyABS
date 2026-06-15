# Airbag Square-Well (ABS) Models

A Python library for accelerator beam-dynamics studies implementing Burov's
Airbag Square-Well (ABS) model equations and related tools for studying
convective instabilities.

## Goals

- Clean, tested CPU reference implementation of the ABS equations
- Numerical tools for mode spectra, wakes, and eigenvalue problems
- Diagnostics for characterizing convective instability behavior
- Compatibility with existing Mathematica notebooks used as physics references
- Clear separation between model assembly, solvers, and post-processing

## Quick Start

```bash
source /Users/gonza839/pyabs/bin/activate
/Users/gonza839/pyabs/bin/python -m pip install -e .
/Users/gonza839/pyabs/bin/python -m pytest tests/
```

## Project Structure

```text
src/pyabs/              # Core library
    core/               # ABS model assembly and physics kernels
    solvers/            # Eigenvalue and scan utilities
    diagnostics/        # Convective-instability post-processing
    plotting/           # Visualization helpers

tests/                  # Unit and regression tests
examples/               # Worked examples and parameter scans
mathematica_notebooks/  # Physics reference notebooks
literature/             # Reference papers
docs/                   # Theory and implementation notes
```

The exact structure may evolve as formulas are translated from the notebooks and
literature into tested Python code.

## Key Concepts

- **ABS model**: Airbag bunch dynamics in a square potential well, following
  Burov's formulation.
- **Wakefields**: Collective forces that couple beam slices and head-tail modes.
- **Space charge**: Tune shifts and mode structure changes caused by beam
  self-fields.
- **Convective instability**: Transient amplification along the bunch even when
  the absolute mode spectrum may appear stable.
- **Mode spectra**: Eigenvalue and eigenvector structure used to diagnose
  coherent beam motion.

## Design Principles

1. **Convention-explicit**: Document sign, tune-shift, wake, and normalization
   conventions near the formulas that depend on them.
2. **Reference-first**: Preserve the mathematical structure from Burov's papers
   and the Mathematica notebooks before optimizing.
3. **Test-driven translation**: Add focused tests whenever notebook or literature
   formulas are moved into Python.
4. **Separate physics from diagnostics**: Keep matrix assembly and model
   equations separate from growth-rate extraction, mode tracking, plotting, and
   stability classification.

## Dependencies

**Core:**

- numpy
- scipy
- matplotlib
- pytest

**Optional:**

- jax, jaxlib for differentiable or accelerated experiments
- cupy for GPU NumPy-style experiments

## Documentation

- `mathematica_notebooks/` - exploratory notebooks and symbolic references
- `literature/` - papers motivating the ABS model and convective-instability
  analysis
- `docs/` - future theory notes, conventions, and implementation details
