import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import main


class RunAnalysisIntegrationTests(unittest.TestCase):
    @patch("main.connect_database", return_value=None)
    @patch("main.fetch_database_matches")
    @patch("main.render_diagnostic_plot", return_value=None)
    @patch("main.analyze_signal")
    @patch("main.load_phase_library")
    @patch("main.ensure_analysis_dependencies")
    @patch("main.ValidationResult.to_dataframe")
    def test_run_analysis_happy_path(
        self,
        mock_to_dataframe,
        mock_dependencies,
        mock_load_library,
        mock_analyze_signal,
        mock_render_plot,
        mock_fetch_matches,
        mock_connect_db,
    ) -> None:
        mock_to_dataframe.return_value = object()
        mock_load_library.return_value = [main.PhasePattern(name="Testite", formula="T1", primary_peaks=[10.0])]

        candidate = {
            "name": "Testite",
            "formula": "T1",
            "score": 0.1,
            "confidence": 0.6,
            "matched_peaks": 1,
            "peak_coverage": 1.0,
        }
        analysis_payload = {
            "peak_count": 1,
            "peaks": [{"position": 10.0, "intensity": 100.0, "relative_intensity": 1.0}],
            "phase_candidates": [candidate.copy()],
            "phase_scores": [candidate.copy()],
            "phase_library_size": 1,
            "top_phase": candidate.copy(),
        }
        features_payload = {"smoothed": [1.0], "peaks": [], "peak_positions": [10.0]}
        mock_analyze_signal.return_value = (analysis_payload, features_payload)

        mock_fetch_matches.return_value = [
            {
                "entry_id": 42,
                "mineral_name": "Testite",
                "chemical_formula": "T1",
                "space_group": "P1",
                "quality": "A",
            }
        ]

        metadata_overrides = {
            "instrument_id": "inst",
            "sample_id": "sample-over",
            "x_unit": "degrees",
            "acquisition_datetime": "2024-01-01T00:00:00Z",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            xy_path = tmp_path / "sample.xy"
            xy_path.write_text("1 2\n3 4\n", encoding="utf-8")
            output_dir = tmp_path / "outputs"

            result = main.run_analysis(
                input_files=[xy_path],
                output_dir=output_dir,
                metadata_overrides=metadata_overrides,
                runtime_overrides={},
                phase_library_path=None,
                show_plot=False,
                save_plot=False,
            )

            self.assertTrue(result["database_connected"] is False)
            self.assertEqual(result["phase_summary"]["total_files"], 1)
            self.assertEqual(result["phase_summary"]["database_hits"], 1)
            detected = result["phase_summary"]["detected_phases"]
            self.assertIn("Testite", detected)
            self.assertGreaterEqual(detected["Testite"]["max_quality_boost"], 0.2)

            file_result = result["files"][0]
            self.assertEqual(file_result["quality_boost"], 0.2)
            self.assertEqual(file_result["top_phase"]["confidence_adjusted"], 0.8)
            self.assertTrue((output_dir / "sample_validation.json").exists())
            self.assertTrue((output_dir / "sample_analysis.json").exists())

            analysis_doc = json.loads((output_dir / "sample_analysis.json").read_text(encoding="utf-8"))
            self.assertEqual(analysis_doc["quality_boost"], 0.2)
            self.assertEqual(analysis_doc["top_phase"]["confidence_adjusted"], 0.8)


if __name__ == "__main__":
    unittest.main()
