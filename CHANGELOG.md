# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial OpenTelemetry tracing integration
- `@spyglass_trace` decorator for function tracing
- OpenAI API call tracing with `spyglass_openai` wrapper
- Automatic capture of function arguments and return values
- Support for custom span names
- Environment-based configuration (SPYGLASS_API_KEY, SPYGLASS_DEPLOYMENT_ID)
- Comprehensive test suite with pytest
- Added langchain-aws and langchain StructuredTool for MCP support

### Changed
- Nothing yet

### Deprecated
- Nothing yet

### Removed
- Nothing yet

### Fixed
- Nothing yet

### Security
- Nothing yet

---

## Notes

This project is currently in beta (v0.1.x). The first stable release (v1.0.0) will mark the public API as stable.

For development versions and internal changes, see the git commit history.
