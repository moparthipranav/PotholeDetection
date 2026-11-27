import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dashboard import (
    discover_runs,
    display_percent,
    find_media,
    load_records,
    parse_records,
    select_frame,
)


class DashboardTests(unittest.TestCase):
    def test_discovers_runs_with_telemetry_only(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            run = root / "run-a"
            run.mkdir()
            (run / "situational_awareness.jsonl").write_text(
                '{"features": {}, "decision": {}}\n',
                encoding="utf-8",
            )
            (root / "unrelated").mkdir()

            self.assertEqual(discover_runs(root), [run])

    def test_load_records_reports_malformed_line_number(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            telemetry = Path(temporary_directory) / "results.jsonl"
            telemetry.write_text(
                '{"features": {}, "decision": {}}\n{bad json}\n',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, r"results\.jsonl:2"):
                load_records(telemetry)

    def test_uploaded_telemetry_parses_jsonl_text(self):
        records = parse_records(
            '{"source":"road.jpg","frame":0,"features":{},"decision":{}}\n',
            "uploaded.jsonl",
        )

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["frame"], 0)

    def test_media_lookup_uses_source_basename(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            run = Path(temporary_directory)
            image = run / "road.jpg"
            image.touch()

            self.assertEqual(
                find_media(run, r"D:\input\road.jpg"),
                image,
            )
            self.assertIsNone(find_media(run, "0"))

    def test_media_lookup_prefers_annotated_media_name(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            run = Path(temporary_directory)
            video = run / "road-preview.mp4"
            video.touch()

            self.assertEqual(
                find_media(run, r"D:\input\road.avi", "road-preview.mp4"),
                video,
            )

    def test_fractional_advisory_values_are_rendered_as_percent(self):
        self.assertEqual(display_percent(0.587), "59%")
        self.assertEqual(display_percent(None), "—")

    def test_single_frame_does_not_create_invalid_slider(self):
        with patch("dashboard.st.sidebar.caption") as caption, patch(
            "dashboard.st.sidebar.slider"
        ) as slider:
            self.assertEqual(select_frame([0]), 0)

        caption.assert_called_once_with("Only frame 0 is available.")
        slider.assert_not_called()

    def test_multiple_frames_use_a_slider(self):
        with patch("dashboard.st.sidebar.slider", return_value=2) as slider:
            self.assertEqual(select_frame([1, 2]), 2)

        slider.assert_called_once_with(
            "Frame",
            min_value=1,
            max_value=2,
            value=2,
        )


if __name__ == "__main__":
    unittest.main()
