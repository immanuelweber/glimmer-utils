# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.2] - 2025-01-12

### Added
- CHANGELOG.md to track version history
- Just installation instructions in README
- Just setup target for simplified onboarding

### Changed
- Modernized README documentation
- Removed hardcoded version references (automated via setuptools_scm)
- Removed redundant Requirements section from README
- Updated Testing section to reference justfile commands

## [0.3.1] - 2024-11-03

### Added
- CI tests, coverage, and security scanning
- Secret scanning to CI pipeline
- GitHub issue and PR templates
- Just setup target for simplified onboarding
- Pre-commit hooks configuration

### Changed
- Remove PyPI references (package name conflict)
- Update CI workflow to use main branch only

## [0.3.0] - Earlier

Initial versioned release with core functionality:
- Data collators for padding and stacking tensors
- Lightning callbacks for progress bars, printing and plotting
- Patched LightningDataModule for quick dataset usage
