import json
import tempfile
import unittest
from pathlib import Path

import main


class ValidationTests(unittest.TestCase):
    def test_validate_xy_file_passes_with_clean_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "sample.xy"
            file_path.write_text("1 2\n3 4\n", encoding="utf-8")

            metadata = {field: f"value_{idx}" for idx, field in enumerate(main.REQUIRED_METADATA_FIELDS)}
            result = main.validate_xy_file(file_path, metadata)

            self.assertTrue(result.passed)
            self.assertEqual(result.row_count, 2)
            self.assertEqual(result.errors, [])
            self.assertEqual(result.warnings, [])

    def test_validate_xy_file_warns_when_metadata_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "sample.xy"
            file_path.write_text("1 2\n", encoding="utf-8")

            metadata = {}
            result = main.validate_xy_file(file_path, metadata)

            self.assertTrue(result.passed)
            self.assertEqual(result.row_count, 1)
            self.assertIn("Missing metadata fields", " ".join(result.warnings))


class MetadataTests(unittest.TestCase):
    def test_load_metadata_for_file_merges_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            xy_path = base / "sample.xy"
            xy_path.write_text("1 2\n", encoding="utf-8")
            sidecar_path = base / "sample.meta.json"
            sidecar_path.write_text(json.dumps({"instrument_id": "sidecar", "operator": "op"}), encoding="utf-8")

            base_metadata = {"instrument_id": "base", "sample_id": "base-sample"}
            merged = main.load_metadata_for_file(xy_path, base_metadata)

            self.assertEqual(merged["instrument_id"], "sidecar")
            self.assertEqual(merged["operator"], "op")
            self.assertEqual(merged["sample_id"], "base-sample")

    def test_load_metadata_for_file_defaults_sample_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            xy_path = base / "dataset.xy"
            xy_path.write_text("1 2\n", encoding="utf-8")

            merged = main.load_metadata_for_file(xy_path, {})
            self.assertEqual(merged["sample_id"], "dataset")


class PhaseLibraryTests(unittest.TestCase):
    def test_load_phase_library_from_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            library_path = Path(tmpdir) / "phases.json"
            payload = [
                {"name": "Testite", "formula": "T1", "primary_peaks": [10.0, 20.0]},
                {"name": "Demoite", "formula": "D2", "primary_peaks": [15.5], "notes": "example"},
            ]
            library_path.write_text(json.dumps(payload), encoding="utf-8")

            phases = main.load_phase_library(library_path)
            self.assertEqual(len(phases), 2)
            self.assertEqual(phases[0].name, "Testite")
            self.assertAlmostEqual(phases[0].primary_peaks[1], 20.0)
            self.assertEqual(phases[1].notes, "example")


if __name__ == "__main__":
    unittest.main()
