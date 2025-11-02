"""Open WebUI tool wrapper for invoking the PhaseID REST API.

This module can be dropped into an Open WebUI deployment (e.g.,
<data_dir>/tools/ directory) to expose the PhaseID analysis service as
a callable tool for LLMs. The implementation uses the standard Open
WebUI Tool API (pydantic models + callable object) so it works with
providers that support function/tool calling.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import requests
from pydantic import BaseModel, Field, root_validator

try:
    from open_webui.tools.tool import Tool  # type: ignore
except ImportError:  # pragma: no cover - fallback for local testing
    class Tool:  # type: ignore
        """Fallback Tool base class for environments without Open WebUI."""

        name: str
        description: str
        args_schema: BaseModel

        def __call__(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
            raise NotImplementedError


DEFAULT_API_URL = os.getenv("PHASEID_API_URL", "http://phaseid-api:8000")
DEFAULT_TIMEOUT = float(os.getenv("PHASEID_API_TIMEOUT", "120"))


class PhaseIDInput(BaseModel):
    """Expected request payload for PhaseID analysis."""

    input_files: List[str] = Field(
        ...,
        description=(
            "Absolute or container-relative paths to spectra files that the PhaseID API can access. "
            "If the API container runs with INPUT_ROOT=/app, relative paths such as "
            "'quartz_10003.xy' are resolved at /app/quartz_10003.xy."
        ),
    )
    metadata: Dict[str, str] = Field(
        default_factory=dict,
        description="Global metadata overrides (instrument ID, sample ID, etc.).",
    )
    runtime_overrides: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional runtime configuration passed straight to the API.",
    )
    phase_library: Optional[str] = Field(
        default=None,
        description="Optional path/URL to a JSON phase library. Overrides the default configured on the API side.",
    )
    output_dir: Optional[str] = Field(
        default=None,
        description="Override for the API output directory (defaults to server OUTPUT_DIR env).",
    )
    show_plot: bool = Field(
        default=False, description="Request the API to display interactive plots (requires GUI in container)."
    )
    save_plot: bool = Field(
        default=False, description="Request the API to persist diagnostic plots alongside JSON outputs."
    )

    @root_validator
    def check_files(cls, values: Dict[str, Any]) -> Dict[str, Any]:  # pylint: disable=no-self-argument
        if not values.get("input_files"):
            raise ValueError("input_files must contain at least one path")
        return values


class PhaseIDTool(Tool):
    name = "phaseid_analyze"
    description = (
        "Analyse X-Y diffraction spectra with the PhaseID service. "
        "Call with `input_files` (paths visible to the API container) and any known metadata such as "
        "`instrument_id`, `sample_id`, `x_unit`, `acquisition_datetime`. Optional arguments include "
        "`phase_library`, `runtime_overrides`, `output_dir`, `show_plot`, and `save_plot`. The tool returns "
        "structured JSON containing validation results, detected peaks, ranked phase candidates, and COD "
        "database matches plus a roll-up summary."
    )
    args_schema = PhaseIDInput

    def __init__(self, api_url: Optional[str] = None, timeout: Optional[float] = None) -> None:
        self.api_url = api_url or DEFAULT_API_URL
        self.timeout = timeout or DEFAULT_TIMEOUT

    def __call__(self, input_files: List[str], **kwargs: Any) -> str:
        payload = PhaseIDInput(input_files=input_files, **kwargs).dict()
        response = requests.post(
            f"{self.api_url.rstrip('/')}/analyze",
            json=payload,
            timeout=self.timeout,
        )
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:  # pragma: no cover - surfaced for visibility
            raise RuntimeError(
                f"PhaseID API returned {response.status_code}: {response.text}"
            ) from exc

        data = response.json()
        return json.dumps(data, indent=2)


# Convenience factory used by Open WebUI when auto-discovering tool modules.

def get_tools() -> List[Tool]:
    """Return instantiated tool(s) for Open WebUI auto-discovery."""

    return [PhaseIDTool()]
