# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-04-20

### Added
- Main application interface with PyQt5
- Radar parameters configuration (frequency, spectrum, FFT size, beam width, survey area, range)
- Support for 5 satellites: cloudSAT, calipso, ICESat2, LRO, solarB
- Two processing methods: standard and polar reformatting
- Real-time visualization of scattering field and RLI
- Batch processing of 16 frames
- Export results to PNG images
- Configuration saving to JSON
- Window icon application

### Changed
- Refactored file structure for better maintainability
- Optimized memory usage by saving results directly to disk
- Adjusted default FFT size from 4096 to 1024

### Removed
- "Совмещенная цель" (combined target) option (no data available)

## [0.0.1] - 2026-01-01

### Added
- Initial release
- Basic RLI processing
- Simple GUI