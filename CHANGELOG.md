# Changelog

All notable changes to this project are documented in this file.

## v0.1.1

### Added
- DGL integration wrappers in `rings.integrations` (`DGLOriginal`, `DGLEmptyFeatures`, `DGLRandomFeatures`, `DGLEmptyGraph`, `DGLCompleteGraph`, `DGLRandomGraph`).
- DGL/PyG conversion helpers in `rings.integrations.dgl` to round-trip homogeneous DGL graphs through existing RINGS perturbations.
- Test coverage for DGL integration behavior and seed/feature handling in `tests/test_dgl_integration.py`.
- Lightning integration tests that validate both PyG and DGL usage in `tests/test_lightning_framework_integration.py`.

### Changed
- Optional dependency model now includes explicit install targets for `lightning`, `dgl`, and combined `integrations` extras/groups in `pyproject.toml`.
- Integration documentation now includes DGL setup and usage examples in `README.md` and `docs/source/integrations.rst`.
- Package version updated to `0.1.1`.

## v0.1.0

### Added
- Initial `rings-evaluation` release on `main` with the core perturbation framework for graph-learning dataset evaluation.
- Support for integration into existing GNN workflows via `SeparabilityStudy`.
- PyTorch Lightning callback support via `SeparabilityCallback`.
- Documentation and runnable examples for integrating RINGS into model evaluation pipelines.
