# Changelog

All notable changes to the AFS Acquisition project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- CHANGELOG.md to track future changes

### Changed
- Nothing yet

### Fixed
- Nothing yet

## [1.0.0] - 2025-12-05

### Added
- Initial release of AFS Acquisition system
- High-speed camera control (IDS uEye UI-3080CP)
- XY and Z stage control (Mad City Labs Nano-Drive)
- Function generator control (Siglent SDG1032X)
- Oscilloscope control (Siglent SDS804X HD)
- HDF5 video recording with GPU-accelerated downscaling
- Real-time camera visualization with LUT support
- Resonance finder tool
- Force path designer
- Lookup table generator for Z-stage calibration
- Comprehensive metadata storage
- Background compression for space efficiency
- State recovery and error handling
- Performance monitoring and logging
- Keyboard shortcuts for efficient operation

### Notes
- Optimized for Windows 11 with Intel i7-14700 (20 cores)
- Supports AMD Radeon Pro WX 3100 GPU acceleration via OpenCL
- Production-ready for scientific data acquisition

---

## Version History Format

- **[X.Y.Z]** - Release date in YYYY-MM-DD format
- **Added** for new features
- **Changed** for changes in existing functionality
- **Deprecated** for soon-to-be removed features
- **Removed** for now removed features
- **Fixed** for any bug fixes
- **Security** in case of vulnerabilities
