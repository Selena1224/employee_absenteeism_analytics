"""Calculation and export consistency tests for the five-tab dashboard.

Run: python -m unittest discover -s tests -v
"""
import io
import unittest
from zipfile import ZipFile

import app as dashboard


class DashboardTests(unittest.TestCase):
    def test_five_tab_navigation(self):
        # Switching pages must keep all components mounted.
        names = ("overview", "patterns", "factors", "simulator", "recommendations")
        for active in names:
            styles = dashboard.show_tab(active)
            self.assertEqual(len(styles), 5)
            self.assertEqual(
                [style["display"] for style in styles],
                ["block" if name == active else "none" for name in names],
            )

    def test_source_totals(self):
        self.assertEqual(len(dashboard.df), 740)
        self.assertEqual(dashboard.df["ID"].nunique(), 36)
        self.assertEqual(int(dashboard.df["Absenteeism time in hours"].sum()), 5124)
        self.assertEqual(len(dashboard.absence_data), 696)

    def test_march(self):
        records = dashboard.apply_exploration_filters([3], [], [], "all")
        self.assertEqual(len(records), 87)
        self.assertEqual(int(records["Absenteeism time in hours"].sum()), 765)
        self.assertEqual(int((records["Absenteeism time in hours"] > 0).sum()), 83)

    def test_extended(self):
        records = dashboard.apply_exploration_filters([], [], [], "extended")
        self.assertEqual(len(records), 63)
        self.assertEqual(int(records["Absenteeism time in hours"].sum()), 2440)

    def test_zero_hours(self):
        records = dashboard.apply_exploration_filters([], [], [], "zero")
        self.assertEqual(len(records), 44)
        self.assertEqual(int(records["Absenteeism time in hours"].sum()), 0)

    def test_sanitized_detail_has_all_records_without_id(self):
        records = dashboard.apply_exploration_filters([], [], [], "all")
        detail = dashboard.filtered_detail_export(records)
        self.assertEqual(len(detail), 740)
        self.assertNotIn("ID", detail.columns)
        self.assertNotIn("Age", detail.columns)
        self.assertIn("Reason label", detail.columns)
        self.assertEqual(dashboard.reason_labels[27], "Physiotherapy")

    def test_export_tables_reconcile(self):
        for months, seasons, reasons, duration in (
            ([], [], [], "all"),
            ([3], [], [], "all"),
            ([], [], [], "extended"),
            ([], [], [], "zero"),
            ([3], [1], [28], "extended"),
        ):
            with self.subTest(months=months, seasons=seasons,
                              reasons=reasons, duration=duration):
                rows = dashboard.apply_exploration_filters(
                    months, seasons, reasons, duration
                )
                frames = dashboard.analysis_export_frames(
                    rows, months, seasons, reasons, duration
                )
                total = int(rows["Absenteeism time in hours"].sum())
                self.assertEqual(len(frames), 9)
                self.assertEqual(len(frames["01_filtered_records_no_ids.csv"]), len(rows))
                for name in ("02_monthly_summary.csv", "03_reason_summary.csv",
                             "04_weekday_summary.csv", "05_seasonal_summary.csv",
                             "06_duration_summary.csv"):
                    self.assertEqual(int(frames[name]["total_hours"].sum()), total)

    def test_scenario_default(self):
        result = dashboard.compute_coverage_scenario(12, 8, 2, 96, 2, 8, 100)
        self.assertEqual(result["Coverage gap without backup"], 16)
        self.assertEqual(result["Coverage gap with backup"], 0)
        result = dashboard.compute_coverage_scenario(12, 8, 2, 96, 2, 8, 50)
        self.assertEqual(result["Coverage gap with backup"], 8)

    def test_reset_scenario(self):
        self.assertEqual(
            dashboard.reset_coverage_scenario(1),
            (12, 8, 2, 96, 1, 8, 100),
        )
        defaults = dashboard.reset_coverage_scenario(1)
        scenario = dashboard.compute_coverage_scenario(*defaults)
        self.assertEqual(scenario["Coverage gap without backup"], 16)
        self.assertEqual(scenario["Coverage gap with backup"], 8)

    def test_scenario_validation(self):
        with self.assertRaises(ValueError):
            dashboard.compute_coverage_scenario(12, 8, 15, 96, 2, 8, 100)
        with self.assertRaises(ValueError):
            dashboard.compute_coverage_scenario(12, 8, 2, 96, 2, 8, 150)

    def test_zip_contents(self):
        payload = dashboard.download_analysis_pack(1, [3], [], [], "all")
        self.assertEqual(payload["filename"], "absenteeism_analysis_pack.zip")
        # Dash dcc.Download returns a base64-encoded payload for send_bytes.
        import base64
        content = base64.b64decode(payload["content"])
        with ZipFile(io.BytesIO(content)) as archive:
            names = archive.namelist()
            self.assertEqual(len(names), 9)
            self.assertIn("01_filtered_records_no_ids.csv", names)
            self.assertTrue(archive.read("00_selected_kpis.csv").startswith(b"\xef\xbb\xbf"))


if __name__ == "__main__":
    unittest.main()
