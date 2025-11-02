# XY File Specification

This document records the expected structure for PhaseID input files. XY files contain tabular spectra exported from diffractometers or spectrometers and must adhere to the following contract so the ingestion pipeline can operate deterministically.

## Structure
- Plain text using UTF-8 encoding.
- One observation per line, terminated with `\n`.
- Columns separated by whitespace (single space or tab); commas are not accepted.
- Header rows are **not** permitted. The first line must contain numeric values.
- Empty lines are ignored but discouraged.

## Columns
1. **X** — Sample coordinate (e.g., diffraction angle 2θ in degrees or wavelength in nanometres). Units must be documented in run metadata and remain consistent per file.
2. **Y** — Measured intensity (arbitrary units). Values should be non-negative real numbers.

Additional columns are currently unsupported. Future revisions may extend the schema with uncertainty estimates or auxiliary channels; clients must version their exports accordingly.

## Metadata Requirements
Each batch run must supply a sidecar JSON file (named `<stem>.meta.json`) or CLI overrides, providing:
- `instrument_id`: Identifier for the originating instrument.
- `sample_id`: Unique label for the physical sample.
- `x_unit`: Unit string used for the X column (e.g., `degrees`, `nm`).
- `acquisition_datetime`: ISO 8601 timestamp for the measurement.
- `operator`: Optional name or initials of the analyst.

When processing a single XY file via the CLI, these metadata fields can be supplied through `--metadata` flags or an overrides file. Batch folders can rely on the `<stem>.meta.json` sidecars; the CLI automatically discovers and merges them with global overrides, defaulting `sample_id` to the file stem when omitted.

Phase identification heuristics load reference peaks from `config/reference_phases.json` by default. Supply an alternative file via `--phase-library` to use project-specific phase catalogs.

## Validation Expectations
- Files must be readable without decompression or proprietary drivers.
- Column counts must match the two-column schema; mixed delimiters are rejected.
- X and Y entries must parse as finite floating-point numbers; `NaN` or infinite values trigger validation failures.
- Metadata is required for downstream MCP reporting; missing fields raise warnings that bubble up through the MCP server.

These requirements inform both the validation layer and the MCP interface contract described in the development plan.
