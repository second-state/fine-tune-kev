import json
from pathlib import Path
import tempfile
import unittest

from difficulty_latency import latency_stats, summarize_run
from jev_difficulty import write_json


class DifficultyLatencyTests(unittest.TestCase):
    def test_p95_uses_nearest_rank_and_retains_first_request(self):
        result = latency_stats([1000] + list(range(1, 20)))
        self.assertEqual(result["count"], 20)
        self.assertEqual(result["median_ms"], 10.5)
        self.assertEqual(result["p95_ms"], 19)
        self.assertEqual(result["max_ms"], 1000)

    def test_missing_server_measurements_are_not_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(root / "run_meta.json", {"model": "test", "endpoint": "https://example.test"})
            (root / "responses.jsonl").write_text(json.dumps({
                "dataset": "one", "task_id": "one", "attempts": [
                    {"http_status": 200, "latency_ms": 10, "response": {}},
                ],
            }) + "\n")
            summary, rows = summarize_run(root)
            self.assertEqual(summary["client"]["mean_ms"], 10)
            self.assertIsNone(summary["server"])
            self.assertIsNone(rows[0]["server_latency_ms"])
            self.assertEqual(summary["server_latency_missing_count"], 1)

    def test_invalid_measurements_fail_instead_of_distorting_averages(self):
        for value in [-1, float("nan"), float("inf"), True]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                latency_stats([value])
        self.assertIsNone(latency_stats([]))

    def test_header_timer_keeps_its_source_and_body_timer_takes_precedence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(root / "run_meta.json", {"model": "test", "endpoint": "https://example.test"})
            attempts = [
                {"http_status": 200, "latency_ms": 20, "response": {},
                 "timing_headers": {"x-inference-time-ms": "12.5"}},
                {"http_status": 200, "latency_ms": 20, "response": {"latency_ms": 8},
                 "timing_headers": {"x-inference-time-ms": "12.5"}},
            ]
            (root / "responses.jsonl").write_text(json.dumps({
                "dataset": "one", "task_id": "one", "attempts": attempts,
            }) + "\n")
            summary, rows = summarize_run(root)
            self.assertEqual(rows[0]["server_latency_ms"], 12.5)
            self.assertEqual(rows[0]["server_latency_source"], "header:x-inference-time-ms")
            self.assertEqual(rows[1]["server_latency_ms"], 8)
            self.assertNotIn("server_latency_source", rows[1])
            self.assertEqual(summary["server"]["mean_ms"], 10.25)

    def test_nested_usage_timer_precedes_header_and_rejects_invalid_values(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(root / "run_meta.json", {"model": "clef", "endpoint": "http://127.0.0.1"})
            record = {"dataset": "one", "task_id": "one", "attempts": [
                {"http_status": 200, "latency_ms": 20,
                 "response": {"usage": {"latency_ms": 12.5}},
                 "timing_headers": {"x-inference-time-ms": "15"}},
                {"http_status": 200, "latency_ms": 20,
                 "response": {"latency_ms": 8, "usage": {"latency_ms": 12.5}}},
            ]}
            path = root / "responses.jsonl"
            path.write_text(json.dumps(record) + "\n")
            summary, rows = summarize_run(root)
            self.assertEqual(rows[0]["server_latency_ms"], 12.5)
            self.assertEqual(rows[0]["server_latency_source"], "body:usage.latency_ms")
            self.assertEqual(rows[1]["server_latency_ms"], 8)
            self.assertNotIn("server_latency_source", rows[1])
            self.assertEqual(summary["server"]["mean_ms"], 10.25)
            for value in [-1, float("nan"), float("inf"), True, "12.5"]:
                with self.subTest(value=value):
                    record["attempts"][0]["response"]["usage"]["latency_ms"] = value
                    path.write_text(json.dumps(record) + "\n")
                    with self.assertRaises(ValueError):
                        summarize_run(root)


if __name__ == "__main__":
    unittest.main()
