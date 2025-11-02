# PhaseID — Mineral Phase Identification Toolkit

PhaseID ingests XY diffraction spectra, extracts diagnostic features, and ranks likely mineral phases using a configurable reference library. It can optionally enrich results with metadata sourced from a PostgreSQL database populated from the Crystallography Open Database (COD). The project is designed to be deployed as an MCP-compatible container or operated locally via the CLI.

## Features

- Validates XY input files against the documented schema and captures machine-readable validation reports.
- Performs baseline correction, smoothing, and peak detection to derive descriptive features from spectra.
- Scores phase candidates using a reference peak library (`config/reference_phases.json`) that can be overridden at runtime.
- Integrates with PostgreSQL to surface matching COD entries for the best phase candidates.
- Ships docker-compose orchestration that builds the analysis container, starts PostgreSQL, and provides a one-shot loader for the COD dataset.

## Requirements

- Python 3.11 (for local execution).
- `pip install -r requirements.txt` (numpy, pandas, psycopg2-binary).
- Docker 20+ and Docker Compose v2 for container-based workflows.

## Repository Layout

```
.
├── main.py                # CLI entrypoint and core analysis logic
├── config/reference_phases.json
├── docs/
│   ├── xy_format.md       # XY file specification and metadata contract
│   └── database.md        # Database integration details
├── scripts/load_cod.py    # COD -> PostgreSQL loader utility
├── db_raw/                # (ignored) folder containing COD SQLite dump(s)
└── docker-compose.yml     # Orchestration for app + PostgreSQL + data loader
```

## Quick Start (Local CLI)

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. (Optional) Set database environment variables if you want COD enrichment:
   ```bash
   export DATABASE_HOST=localhost
   export DATABASE_PORT=5432
   export DATABASE_NAME=phaseid
   export DATABASE_USER=phaseid
   export DATABASE_PASSWORD=phaseid
   ```
3. Run the analyser on a single file:
   ```bash
   python main.py --input-file quartz_10003.xy --metadata instrument_id=demo --metadata sample_id=sample-001
   ```
   The command writes JSON artefacts to `outputs/` and prints a consolidated JSON payload to stdout.

## Container Workflow

The provided compose stack builds the PhaseID container, starts PostgreSQL, and runs a loader job for the COD dataset located in `db_raw/`.

```bash
# Build images and start database
docker compose up -d db

# Start the PhaseID API
docker compose up -d phaseid-api

# Load COD data (run once or when the dataset changes)
docker compose --profile loader run --rm cod-loader --truncate

# Run the analysis CLI inside the API image (override entrypoint)
docker compose run --rm phaseid-api python main.py --input-file /app/quartz_10003.xy --metadata instrument_id=demo
```

Container services:

- `db`: PostgreSQL 15 with persistent volume `postgres-data`.
- `phaseid-api`: FastAPI MCP server (default route `/analyze`) built from the analysis toolkit.
- `cod-loader`: Optional (profile `loader`) one-shot job invoking `scripts/load_cod.py` to populate `cod_entries`.

See `docs/database.md` for more details on the schema and environment variables.

## MCP Server / HTTP API

The PhaseID container starts a FastAPI service on port `8000` by default. Health and analysis endpoints:

- `GET /health` → `{ "status": "ok" }`
- `POST /analyze` → run phase identification.

Example request (inside the compose project):

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
        "input_files": ["/app/quartz_10003.xy"],
        "metadata": {
          "instrument_id": "demo",
          "sample_id": "sample-001",
          "x_unit": "degrees",
          "acquisition_datetime": "2024-01-01T00:00:00Z"
        },
        "save_plot": true
      }'
```

The response mirrors the CLI JSON payload (`files`, `phase_summary`, `database_connected`). Override the phase library or output directory using the request fields `phase_library` and `output_dir`. The service honours the same environment variables for database connectivity and will continue gracefully if PostgreSQL is unavailable.

### Open WebUI Tool Wrapper

An Open WebUI-compatible tool wrapper is included at `integrations/openwebui/phaseid_tool.py`. To enable it:

1. Copy the file into your Open WebUI instance under `data/tools/phaseid_tool.py` (or another path that is scanned for tools).
2. Restart Open WebUI so it discovers the new tool; it appears as `phaseid_analyze`.
3. Configure the following environment variables (or edit the wrapper) so the tool knows how to reach the API:
   - `PHASEID_API_URL` (e.g., `http://phaseid-api:8000`)
   - `PHASEID_API_TIMEOUT` (seconds, optional)
4. Within Open WebUI, grant the tool to the model/agent that should run analyses. The tool accepts the same payload as the HTTP API (`input_files`, `metadata`, `runtime_overrides`, etc.) and returns the JSON response as a formatted string for downstream reasoning.

The wrapper uses the standard Open WebUI `Tool` interface, so it works with both LLM function-calling providers and prompt-tooling workflows.

### Environment Variables

PhaseID honours several environment variables so the container can be customised without editing code:

| Variable | Purpose | Default |
|----------|---------|---------|
| `DATABASE_HOST`, `DATABASE_PORT`, `DATABASE_NAME`, `DATABASE_USER`, `DATABASE_PASSWORD` | PostgreSQL connection used for COD enrichment | `None` (disabled when unset) |
| `OUTPUT_DIR` | Root directory for generated artefacts | `outputs/` |
| `INPUT_ROOT` | Base directory for resolving relative `input_files` paths supplied to the API | Current working directory |
| `PHASE_LIBRARY_PATH` | Path to the phase reference JSON library | `config/reference_phases.json` |
| `APP_HOST`, `APP_PORT` | FastAPI bind host/port when running `server.py` | `0.0.0.0`, `8000` |

`docker-compose.yml` sets sane defaults (`/app` for `INPUT_ROOT`, `/outputs` for `OUTPUT_DIR`, and publishes `${PHASEID_PORT:-8000}`).

## XY File Specification

Refer to `docs/xy_format.md` for the full contract, including column requirements, metadata expectations, and sidecar naming conventions (`<stem>.meta.json`). The CLI automatically merges global overrides (`--metadata`, `--metadata-file`) with per-file sidecars and defaults `sample_id` to the file stem.

## Configuration

- **Phase Library**: Override the default reference peaks with `--phase-library /path/to/custom.json`.
- **Metadata Overrides**: Supply repeated `--metadata KEY=VALUE` flags or a JSON map via `--metadata-file`.
- **Runtime Config**: Pass additional MCP or pipeline settings via `--runtime-config`.
- **Plots**: Enable diagnostic plots with `--save-plot` (always saved) or `--show-plot` (interactive, requires `matplotlib` inside the environment).

## Database Loader Script

The loader accepts several flags (see `scripts/load_cod.py --help`):

```
--sqlite-path PATH      # Path to COD SQLite dump (default compose mount: /app/db_raw/cod2205.sq)
--chunk-size 1000       # Batch size for transfers
--table-name cod_entries
--truncate              # Clear the table before loading
```

Environment variables are shared with the main app (`DATABASE_*`). The script performs upserts so repeated runs keep the table in sync.

## Testing

Unit and integration tests use Python’s standard library (`unittest`). Run them with:

```bash
python -m unittest discover -s tests -v
```

CLI integration tests stub heavy dependencies, so they run quickly without requiring numpy/pandas or a live database.

## Outputs

Each processed file produces:

- `<stem>_validation.json`: Validation summary of the input data and metadata.
- `<stem>_analysis.json`: Feature extraction results, top phase candidates, and COD matches.
- Optional `<stem>_diagnostic.png` if `--save-plot` was used.
- Aggregate JSON printed to stdout summarising all files and roll-ups per detected phase (including referenced COD entries when available).

## Development Notes

- Compose automatically builds the project; re-run `docker compose build` after code changes.
- COD datasets are large and ignored by git (.gitignore); ensure `db_raw/` contains the required `.sq` dump before running the loader.
- The CLI is MCP-ready: command-line options expose runtime configurations that can be driven by an MCP server or orchestration layer.
