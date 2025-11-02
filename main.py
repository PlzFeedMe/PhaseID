# Mineralogical Tool for XY data analysis

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from types import ModuleType

try:
    import numpy as np  # type: ignore
    import pandas as pd  # type: ignore
except ModuleNotFoundError as exc:  # pragma: no cover - handled at runtime
    _DEPENDENCY_ERROR = exc
    np = None  # type: ignore
    pd = None  # type: ignore
else:
    _DEPENDENCY_ERROR = None

try:
    import psycopg2  # type: ignore
    from psycopg2.extras import RealDictCursor  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    psycopg2 = None  # type: ignore
    RealDictCursor = None  # type: ignore

DEFAULT_PHASE_LIBRARY_PATH = Path("config/reference_phases.json")

REQUIRED_METADATA_FIELDS = [
    "instrument_id",
    "sample_id",
    "x_unit",
    "acquisition_datetime",
]

@dataclass(frozen=True)
class PhasePattern:
    name: str
    formula: str
    primary_peaks: List[float]
    notes: Optional[str] = None


_DEFAULT_PHASE_LIBRARY: List[PhasePattern] = [
    PhasePattern(name="Quartz", formula="SiO2", primary_peaks=[20.85, 26.64, 36.54, 39.45]),
    PhasePattern(name="Calcite", formula="CaCO3", primary_peaks=[23.04, 29.42, 35.97, 39.32]),
    PhasePattern(name="Hematite", formula="Fe2O3", primary_peaks=[24.14, 33.18, 35.65, 54.10]),
    PhasePattern(name="Magnetite", formula="Fe3O4", primary_peaks=[18.30, 30.10, 35.45, 57.05]),
    PhasePattern(name="Chalcopyrite", formula="CuFeS2", primary_peaks=[29.36, 34.34, 47.92, 56.00]),
]

logger = logging.getLogger(__name__)


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s - %(message)s")


def ensure_analysis_dependencies() -> Tuple[ModuleType, ModuleType]:
    if _DEPENDENCY_ERROR is not None:
        raise RuntimeError(
            "PhaseID analysis requires numpy and pandas. Install dependencies with `pip install numpy pandas`."
        ) from _DEPENDENCY_ERROR
    assert np is not None and pd is not None  # for type checkers
    return np, pd


def connect_database() -> Optional["psycopg2.extensions.connection"]:
    if psycopg2 is None:
        logger.info("psycopg2 not installed; database integration disabled.")
        return None

    host = os.getenv("DATABASE_HOST")
    if not host:
        logger.info("DATABASE_HOST not set; skipping database connection.")
        return None

    port = int(os.getenv("DATABASE_PORT", "5432"))
    name = os.getenv("DATABASE_NAME", "phaseid")
    user = os.getenv("DATABASE_USER", "phaseid")
    password = os.getenv("DATABASE_PASSWORD", "phaseid")

    try:
        connection = psycopg2.connect(
            host=host,
            port=port,
            dbname=name,
            user=user,
            password=password,
            connect_timeout=5,
        )
        connection.autocommit = True
        logger.info("Connected to PostgreSQL at %s:%d/%s.", host, port, name)
        return connection
    except Exception as exc:
        logger.warning("Failed to connect to PostgreSQL (%s:%d/%s): %s", host, port, name, exc)
        return None


def load_phase_library(path: Optional[Path]) -> List[PhasePattern]:
    if path is None:
        candidate_paths = [DEFAULT_PHASE_LIBRARY_PATH]
        allow_missing = True
    else:
        candidate_paths = [path]
        allow_missing = False

    for candidate in candidate_paths:
        resolved = candidate.expanduser().resolve()
        if not resolved.exists():
            if allow_missing:
                continue
            raise FileNotFoundError(f"Phase library file not found: {resolved}")

        with resolved.open("r", encoding="utf-8") as handle:
            try:
                data = json.load(handle)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Failed to parse phase library JSON from {resolved}: {exc}") from exc

        if not isinstance(data, list):
            raise ValueError(f"Phase library must be a list of phase objects: {resolved}")

        library: List[PhasePattern] = []
        for entry in data:
            if not isinstance(entry, dict):
                raise ValueError(f"Invalid phase entry {entry!r} in {resolved}; expected an object.")
            try:
                name = entry["name"]
                formula = entry.get("formula", "")
                primary_peaks = entry["primary_peaks"]
            except KeyError as exc:
                raise ValueError(f"Missing required key {exc} in {resolved}") from exc
            if not isinstance(primary_peaks, list) or not all(isinstance(p, (int, float)) for p in primary_peaks):
                raise ValueError(f"Primary peaks must be numeric array in {resolved} for phase {name!r}.")
            library.append(
                PhasePattern(
                    name=str(name),
                    formula=str(formula),
                    primary_peaks=[float(p) for p in primary_peaks],
                    notes=str(entry["notes"]) if "notes" in entry else None,
                )
            )
        if library:
            return library

    return list(_DEFAULT_PHASE_LIBRARY)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PhaseID XY data analysis pipeline with MCP-ready configuration overrides."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input-file", type=Path, help="Path to a single XY file.")
    input_group.add_argument("--batch-dir", type=Path, help="Directory containing one or more XY files.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory where figures and machine-readable outputs will be written.",
    )
    parser.add_argument(
        "--metadata",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Inline metadata overrides for the run. Can be supplied multiple times.",
    )
    parser.add_argument(
        "--metadata-file",
        type=Path,
        help="Optional JSON file containing metadata overrides.",
    )
    parser.add_argument(
        "--runtime-config",
        type=Path,
        help="Optional JSON file containing MCP runtime overrides.",
    )
    parser.add_argument(
        "--phase-library",
        type=Path,
        help="Path to a JSON file describing reference phase patterns.",
    )
    parser.add_argument(
        "--show-plot",
        action="store_true",
        help="Display interactive diagnostic plots (requires matplotlib).",
    )
    parser.add_argument(
        "--save-plot",
        action="store_true",
        help="Persist diagnostic plots alongside JSON outputs.",
    )
    return parser.parse_args()


def load_kv_overrides(pairs: List[str]) -> Dict[str, str]:
    overrides: Dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Invalid metadata override '{pair}'. Expected KEY=VALUE.")
        key, value = pair.split("=", 1)
        overrides[key.strip()] = value.strip()
    return overrides


def load_json_file(path: Optional[Path]) -> Dict[str, object]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Override file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        try:
            return json.load(handle)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to parse JSON overrides from {path}: {exc}") from exc


def load_metadata_for_file(file_path: Path, base_metadata: Dict[str, str]) -> Dict[str, str]:
    metadata: Dict[str, str] = dict(base_metadata)
    sidecar_path = file_path.with_suffix(".meta.json")
    if sidecar_path.exists():
        with sidecar_path.open("r", encoding="utf-8") as handle:
            try:
                sidecar = json.load(handle)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Failed to parse metadata sidecar {sidecar_path}: {exc}") from exc
        if not isinstance(sidecar, dict):
            raise ValueError(f"Metadata sidecar {sidecar_path} must contain a JSON object.")
        metadata.update({str(key): str(value) for key, value in sidecar.items()})
    metadata.setdefault("sample_id", file_path.stem)
    return metadata


def collect_input_files(input_file: Optional[Path], batch_dir: Optional[Path]) -> List[Path]:
    if input_file:
        resolved = input_file.expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"XY file not found: {resolved}")
        return [resolved]
    if not batch_dir:
        return []
    resolved_dir = batch_dir.expanduser().resolve()
    if not resolved_dir.is_dir():
        raise NotADirectoryError(f"Batch directory not found: {resolved_dir}")
    xy_files = sorted([path for path in resolved_dir.iterdir() if path.suffix.lower() == ".xy"])
    return xy_files


@dataclass
class ValidationResult:
    file: Path
    rows: List[List[float]]
    errors: List[str]
    warnings: List[str]
    metadata: Dict[str, str]

    @property
    def passed(self) -> bool:
        return not self.errors

    @property
    def row_count(self) -> int:
        return len(self.rows)

    def to_dataframe(self) -> pd.DataFrame:
        _, pd_module = ensure_analysis_dependencies()
        return pd_module.DataFrame(self.rows, columns=["X", "Y"])

    def summary(self) -> Dict[str, object]:
        return {
            "file": str(self.file),
            "passed": self.passed,
            "row_count": self.row_count,
            "errors": self.errors,
            "warnings": self.warnings,
            "metadata": self.metadata,
        }


def validate_xy_file(file_path: Path, metadata: Dict[str, str]) -> ValidationResult:
    errors: List[str] = []
    warnings: List[str] = []
    rows: List[List[float]] = []

    try:
        with file_path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                stripped = raw_line.strip()
                if not stripped:
                    warnings.append(f"Line {line_number} is empty; skipped.")
                    continue
                parts = stripped.split()
                if len(parts) != 2:
                    errors.append(f"Line {line_number} expected 2 columns, found {len(parts)}.")
                    continue
                try:
                    x_val = float(parts[0])
                    y_val = float(parts[1])
                except ValueError:
                    errors.append(f"Line {line_number} contains non-numeric values: {parts}.")
                    continue
                if not math.isfinite(x_val) or not math.isfinite(y_val):
                    errors.append(f"Line {line_number} contains non-finite values: {parts}.")
                    continue
                rows.append([x_val, y_val])
    except OSError as exc:
        errors.append(f"Unreadable file: {exc}")

    if not rows and not errors:
        errors.append("No valid data rows found.")

    missing_fields = [field for field in REQUIRED_METADATA_FIELDS if field not in metadata]
    if missing_fields:
        warnings.append(f"Missing metadata fields: {', '.join(missing_fields)}.")

    return ValidationResult(file=file_path, rows=rows, errors=errors, warnings=warnings, metadata=metadata)


def baseline_correct(values: np.ndarray, window: int = 75) -> np.ndarray:
    _, pd_module = ensure_analysis_dependencies()
    series = pd_module.Series(values)
    baseline = series.rolling(window=window, min_periods=1, center=True).median()
    corrected = series - baseline
    corrected = corrected.clip(lower=0.0)
    return corrected.to_numpy()


def smooth_signal(values: np.ndarray, window: int = 7) -> np.ndarray:
    _, pd_module = ensure_analysis_dependencies()
    series = pd_module.Series(values)
    smoothed = series.rolling(window=window, min_periods=1, center=True).mean()
    return smoothed.to_numpy()


def detect_peaks(
    x: np.ndarray,
    y: np.ndarray,
    min_height: float = 0.05,
    min_distance: int = 5,
    min_prominence: float = 0.02,
) -> List[Dict[str, float]]:
    np_module, _ = ensure_analysis_dependencies()
    if len(y) < 3:
        return []

    peaks: List[Dict[str, float]] = []
    last_index = -min_distance
    max_intensity = float(np_module.max(y)) if len(y) else 0.0
    height_threshold = min_height if max_intensity == 0 else min_height * max_intensity

    for index in range(1, len(y) - 1):
        if y[index] < height_threshold:
            continue
        if index - last_index < min_distance:
            continue
        left = y[index - 1]
        right = y[index + 1]
        current = y[index]
        if current <= left or current < right:
            continue
        prominence = min(current - left, current - right)
        if max_intensity > 0 and prominence < min_prominence * max_intensity:
            continue
        peaks.append(
            {
                "index": float(index),
                "position": float(x[index]),
                "relative_intensity": float(current / max_intensity) if max_intensity else 0.0,
                "intensity": float(current),
            }
        )
        last_index = index

    return peaks


def extract_features(xy_data: pd.DataFrame) -> Dict[str, object]:
    np_module, _ = ensure_analysis_dependencies()
    x = xy_data["X"].to_numpy()
    y = xy_data["Y"].to_numpy()

    corrected = baseline_correct(y)
    if corrected.size == 0:
        normalized = corrected
    else:
        max_value = np_module.max(corrected)
        normalized = corrected / max_value if max_value > 0 else corrected
    smoothed = smooth_signal(normalized)
    peaks = detect_peaks(x, smoothed)

    return {
        "baseline_corrected": corrected,
        "normalized": normalized,
        "smoothed": smoothed,
        "peaks": peaks,
        "peak_positions": [peak["position"] for peak in peaks],
        "peak_count": len(peaks),
    }


def score_phase_matches(peak_positions: List[float], references: List[PhasePattern]) -> List[Dict[str, object]]:
    np_module, _ = ensure_analysis_dependencies()
    results: List[Dict[str, object]] = []

    for pattern in references:
        if not peak_positions:
            results.append(
                {
                    "name": pattern.name,
                    "formula": pattern.formula,
                    "score": None,
                    "confidence": 0.0,
                    "matched_peaks": 0,
                    "peak_coverage": 0.0,
                    "notes": pattern.notes,
                }
            )
            continue

        distances: List[float] = []
        matches = 0
        for reference_peak in pattern.primary_peaks:
            closest_distance = min(abs(reference_peak - sample_peak) for sample_peak in peak_positions)
            distances.append(closest_distance)
            if closest_distance <= 0.5:
                matches += 1

        average_distance = float(np_module.mean(distances)) if distances else None
        coverage = matches / max(len(pattern.primary_peaks), 1)
        if average_distance is None:
            confidence = 0.0
        else:
            confidence = math.exp(-average_distance) * coverage
            confidence = max(0.0, min(1.0, confidence))

        results.append(
            {
                "name": pattern.name,
                "formula": pattern.formula,
                "score": average_distance,
                "confidence": confidence,
                "matched_peaks": matches,
                "peak_coverage": coverage,
                "notes": pattern.notes,
            }
        )

    results.sort(key=lambda item: float("inf") if item["score"] is None else item["score"])
    return results


def fetch_database_matches(
    connection: Optional["psycopg2.extensions.connection"],
    phase: Optional[Dict[str, object]],
    limit: int = 5,
) -> List[Dict[str, object]]:
    if connection is None or phase is None or RealDictCursor is None:
        return []

    query_formula = """
        SELECT entry_id, mineral_name, chemical_formula, space_group, quality
        FROM cod_entries
        WHERE chemical_formula = %s
        ORDER BY entry_id
        LIMIT %s;
    """
    query_name = """
        SELECT entry_id, mineral_name, chemical_formula, space_group, quality
        FROM cod_entries
        WHERE mineral_name ILIKE %s
        ORDER BY entry_id
        LIMIT %s;
    """

    matches: List[Dict[str, object]] = []
    try:
        with connection.cursor(cursor_factory=RealDictCursor) as cursor:
            formula = phase.get("formula")
            if formula:
                cursor.execute(query_formula, (formula, limit))
                matches = cursor.fetchall()
            if not matches:
                name = phase.get("name")
                if name:
                    cursor.execute(query_name, (f"%{name}%", limit))
                    matches = cursor.fetchall()
    except Exception as exc:
        logger.warning("Database lookup failed for phase %s: %s", phase.get("name"), exc)
        return []

    return [
        {
            "entry_id": row["entry_id"],
            "mineral_name": row["mineral_name"],
            "chemical_formula": row["chemical_formula"],
            "space_group": row["space_group"],
            "quality": row["quality"],
        }
        for row in matches
    ]


def analyze_signal(
    xy_data: pd.DataFrame,
    phase_library: List[PhasePattern],
) -> Tuple[Dict[str, object], Dict[str, object]]:
    features = extract_features(xy_data)
    peak_positions = features["peak_positions"]
    phase_scores = score_phase_matches(peak_positions, phase_library)
    top_phase = phase_scores[0] if phase_scores else None

    analysis = {
        "peak_count": features["peak_count"],
        "peaks": [
            {
                "position": float(peak["position"]),
                "intensity": float(peak["intensity"]),
                "relative_intensity": float(peak["relative_intensity"]),
            }
            for peak in features["peaks"]
        ],
        "phase_candidates": phase_scores[:3],
        "phase_scores": phase_scores,
        "phase_library_size": len(phase_library),
        "top_phase": top_phase,
    }
    return analysis, features


def render_diagnostic_plot(
    file_path: Path,
    xy_data: pd.DataFrame,
    features: Dict[str, object],
    analysis: Dict[str, object],
    output_dir: Path,
    show_plot: bool,
    save_plot: bool,
) -> Optional[Path]:
    if not (show_plot or save_plot):
        return None

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed; skipping plot generation for %s", file_path.name)
        return None

    x_values = xy_data["X"].to_numpy()
    raw_values = xy_data["Y"].to_numpy()
    smoothed = features["smoothed"]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x_values, raw_values, label="Raw intensity", color="#4C72B0", alpha=0.4)
    ax.plot(x_values, smoothed, label="Smoothed (normalized)", color="#55A868")
    for peak in analysis["peaks"]:
        ax.axvline(peak["position"], color="#C44E52", linestyle="--", alpha=0.6)

    ax.set_xlabel("X")
    ax.set_ylabel("Intensity")
    ax.set_title(f"Diagnostic plot for {file_path.name}")
    ax.legend(loc="best")

    figure_path: Optional[Path] = None
    if save_plot:
        output_dir.mkdir(parents=True, exist_ok=True)
        figure_path = output_dir / f"{file_path.stem}_diagnostic.png"
        fig.savefig(figure_path, dpi=200, bbox_inches="tight")

    if show_plot:
        plt.show(block=False)
    plt.close(fig)
    return figure_path

def process_file(
    file_path: Path,
    output_dir: Path,
    show_plot: bool,
    save_plot: bool,
    metadata: Dict[str, str],
    runtime_overrides: Dict[str, object],
    phase_library: List[PhasePattern],
    db_connection: Optional["psycopg2.extensions.connection"],
) -> Dict[str, object]:
    logger.info("Processing %s", file_path)
    validation = validate_xy_file(file_path, metadata)
    for warning in validation.warnings:
        logger.warning("[%s] %s", file_path.name, warning)
    if not validation.passed:
        for error in validation.errors:
            logger.error("[%s] %s", file_path.name, error)
        raise ValueError(f"Validation failed for {file_path}")

    xy_data = validation.to_dataframe()
    analysis, features = analyze_signal(xy_data, phase_library)
    db_matches = fetch_database_matches(db_connection, analysis["top_phase"])

    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = render_diagnostic_plot(
        file_path=file_path,
        xy_data=xy_data,
        features=features,
        analysis=analysis,
        output_dir=output_dir,
        show_plot=show_plot,
        save_plot=save_plot,
    )

    validation_path = output_dir / f"{file_path.stem}_validation.json"
    with validation_path.open("w", encoding="utf-8") as handle:
        json.dump(validation.summary(), handle, indent=2)

    analysis_path = output_dir / f"{file_path.stem}_analysis.json"
    with analysis_path.open("w", encoding="utf-8") as handle:
        json.dump(analysis, handle, indent=2)

    return {
        "file": str(file_path),
        "figure_path": str(figure_path) if figure_path else None,
        "validation_path": str(validation_path),
        "analysis_path": str(analysis_path),
        "phase_candidates": analysis["phase_candidates"],
        "top_phase": analysis["top_phase"],
        "database_matches": db_matches,
        "metadata": metadata,
        "runtime_overrides": runtime_overrides,
        "row_count": validation.row_count,
    }


def main():
    configure_logging()
    args = parse_args()

    metadata_overrides = load_json_file(args.metadata_file)
    metadata_overrides.update(load_kv_overrides(args.metadata))
    runtime_overrides = load_json_file(args.runtime_config)

    if metadata_overrides:
        logger.info("Metadata overrides: %s", json.dumps(metadata_overrides, indent=2))
    if runtime_overrides:
        logger.info("Runtime overrides: %s", json.dumps(runtime_overrides, indent=2))

    input_files = collect_input_files(args.input_file, args.batch_dir)
    if not input_files:
        raise FileNotFoundError("No XY files found to process.")

    ensure_analysis_dependencies()
    phase_library = load_phase_library(args.phase_library)
    logger.info("Loaded %d reference phases.", len(phase_library))
    db_connection = connect_database()

    run_results = []
    for file_path in input_files:
        file_metadata = load_metadata_for_file(file_path, metadata_overrides)
        result = process_file(
            file_path,
            args.output_dir,
            args.show_plot,
            args.save_plot,
            file_metadata,
            runtime_overrides,
            phase_library,
            db_connection,
        )
        run_results.append(result)

    logger.info("Processed %d file(s).", len(input_files))
    for result in run_results:
        top_candidate = result["top_phase"]
        top_label = top_candidate["name"] if top_candidate else "Unknown"
        confidence = top_candidate["confidence"] if top_candidate else 0.0
        logger.info(
            "Outputs for %s -> top_phase=%s (confidence=%.2f), analysis=%s, validation=%s, figure=%s",
            result["file"],
            top_label,
            confidence,
            result["analysis_path"],
            result["validation_path"],
            result["figure_path"],
        )

    phase_rollup: Dict[str, Dict[str, object]] = {}
    for result in run_results:
        top_candidate = result["top_phase"]
        if not top_candidate:
            continue
        name = top_candidate["name"]
        rollup_entry = phase_rollup.setdefault(
            name,
            {
                "formula": top_candidate.get("formula"),
                "detections": 0,
                "files": [],
                "database_entries": [],
            },
        )
        rollup_entry["detections"] = int(rollup_entry["detections"]) + 1
        rollup_entry["files"].append(
            {
                "file": result["file"],
                "confidence": top_candidate.get("confidence"),
                "matched_peaks": top_candidate.get("matched_peaks"),
            }
        )
        if result["database_matches"]:
            rollup_entry["database_entries"].extend(
                {
                    "entry_id": match.get("entry_id"),
                    "mineral_name": match.get("mineral_name"),
                    "chemical_formula": match.get("chemical_formula"),
                    "quality": match.get("quality"),
                }
                for match in result["database_matches"]
            )

    summary = {
        "files": run_results,
        "phase_summary": {
            "total_files": len(run_results),
            "detected_phases": phase_rollup,
        },
    }
    print(json.dumps(summary, indent=2))

    if db_connection is not None:
        try:
            db_connection.close()
        except Exception:
            logger.debug("Ignoring error while closing database connection.", exc_info=True)


if __name__ == "__main__":
    main()
