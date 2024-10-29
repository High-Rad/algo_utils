# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.2] - 2024-10-29
### Fixed
- Fixing bug in the `matching_graphs.py` module that caused the `draw_matching_graph` function to crash if called when
`matplotlib`'s backend is not previously set.

## [1.1.1] - 2024-10-29
### Fixed
- Fixing README.md file to contain an explanation of new `matching_graphs.py` module.
- Fixing the installation dependencies in the `setup.py` file to include the `matplotlib` package.
- Fixing bug in the `matching_graphs.py` module that caused the `get_itk_colors` to crash when called with the default
argument.

## [1.1.0] - 2024-10-09
### Added
- adding a new module `matching_graphs.py` to the package, which contains functions that are used to save, load and
visualize the matching between longitudinal tumors.
- adding two new functions, `get_project_root` and `get_absolute_path`, to the `file_operations.py` module.
- adding two new classes, `CaseName` and `PairName`, to the `data_analysis.py` module.

## [1.0.0] - 2024-10-06
### Added
- Initial public release of the project.