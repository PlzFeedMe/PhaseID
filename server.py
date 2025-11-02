from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import main

INPUT_ROOT = Path(os.getenv("INPUT_ROOT", ".")).resolve()
OUTPUT_DIR_ENV = Path(os.getenv("OUTPUT_DIR", "outputs")).resolve()
PHASE_LIBRARY_ENV = os.getenv("PHASE_LIBRARY_PATH")

app = FastAPI(title="PhaseID MCP Server", version="1.0.0")


class AnalyzeRequest(BaseModel):
    input_files: List[str] = Field(..., description="Absolute or container-relative paths to XY files.")
    metadata: Dict[str, str] = Field(default_factory=dict, description="Global metadata overrides.")
    runtime_overrides: Dict[str, Any] = Field(default_factory=dict, description="Additional runtime configuration.")
    phase_library: Optional[str] = Field(default=None, description="Optional path to a phase library JSON file.")
    output_dir: Optional[str] = Field(default=None, description="Directory for generated outputs.")
    show_plot: bool = Field(default=False, description="Display plots (requires matplotlib).")
    save_plot: bool = Field(default=False, description="Persist diagnostic plots.")


class AnalyzeResponse(BaseModel):
    files: List[Dict[str, Any]]
    phase_summary: Dict[str, Any]
    database_connected: bool


@app.get("/health", summary="Service health check")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse, summary="Run phase identification")
def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    try:
        input_paths = []
        for raw in request.input_files:
            path = Path(raw)
            if not path.is_absolute():
                path = (INPUT_ROOT / path).resolve()
            input_paths.append(path)
    except Exception as exc:  # pragma: no cover - validation guard
        raise HTTPException(status_code=400, detail=f"Invalid input file path: {exc}") from exc

    for path in input_paths:
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"Input file not found: {path}")

    output_dir_raw = request.output_dir or os.getenv("OUTPUT_DIR")
    output_dir = Path(output_dir_raw).resolve() if output_dir_raw else OUTPUT_DIR_ENV
    output_dir.mkdir(parents=True, exist_ok=True)

    phase_library_path = None
    if request.phase_library:
        phase_library_path = Path(request.phase_library).resolve()
    elif PHASE_LIBRARY_ENV:
        phase_library_path = Path(PHASE_LIBRARY_ENV).resolve()

    try:
        result = main.run_analysis(
            input_files=input_paths,
            output_dir=output_dir,
            metadata_overrides=request.metadata,
            runtime_overrides=request.runtime_overrides,
            phase_library_path=phase_library_path,
            show_plot=request.show_plot,
            save_plot=request.save_plot,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive catch
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc

    return AnalyzeResponse(**result)


if __name__ == "__main__":  # pragma: no cover - manual launch
    import uvicorn

    host = os.getenv("APP_HOST", os.getenv("HOST", "0.0.0.0"))
    port = int(os.getenv("APP_PORT", os.getenv("PORT", "8010")))
    uvicorn.run("server:app", host=host, port=port, log_level="info")
