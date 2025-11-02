# Database Integration

PhaseID can enrich phase identification results with metadata sourced from a PostgreSQL database populated from the Crystallography Open Database (COD).

## Service Topology

The provided `docker-compose.yml` starts three services:
- `db` — PostgreSQL instance seeded via the `cod-loader` job.
- `phaseid` — Analysis container that runs `main.py`.
- `cod-loader` — One-shot job that transfers COD entries from `db_raw/cod2205.sq` into PostgreSQL (`cod_entries` table).

## Environment Variables

Both the application and the loader read connection properties from the following variables (defaults shown):

```
DATABASE_HOST=db
DATABASE_PORT=5432
DATABASE_NAME=phaseid
DATABASE_USER=phaseid
DATABASE_PASSWORD=phaseid
```

When running outside Docker, define these variables manually before invoking `python main.py`.

## Runtime Behaviour

During analysis runs, `main.py`:
1. Loads phase reference peaks from `config/reference_phases.json` (or a custom JSON supplied via `--phase-library`).
2. Attempts to connect to PostgreSQL using the environment variables. Connection failures are logged as warnings, and the run continues without database enrichment.
3. For the highest-confidence phase candidate, queries `cod_entries` for matching rows based on chemical formula (exact match) or mineral name (`ILIKE` search). The matches are embedded into the JSON response under `database_matches` and rolled up in the `phase_summary`.

This design keeps the MCP output self-contained while surfacing relational context to downstream tooling.
