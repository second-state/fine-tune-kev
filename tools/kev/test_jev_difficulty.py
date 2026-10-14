import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import jev_difficulty as jev


class JevDifficultyTests(unittest.TestCase):
    def test_captures_server_timing_without_unrelated_headers(self):
        response = io.BytesIO(b'{"answers": {}}')
        response.status = 200
        response.headers = {
            "server-timing": "inference;dur=12.34", "x-inference-time-ms": "12.34",
            "x-request-id": "request-one", "set-cookie": "do-not-save",
        }
        with patch("urllib.request.urlopen", return_value=response):
            result = jev.call_api("https://example.test", "test-key")
        self.assertEqual(result["timing_headers"], {
            "server-timing": "inference;dur=12.34", "x-inference-time-ms": "12.34",
        })
        self.assertEqual(result["request_ids"], {"x-request-id": "request-one"})
        self.assertNotIn("do-not-save", json.dumps(result))

    def test_request_cannot_leak_reference_metadata(self):
        task = {
            "prompt": "What is 2 + 2?", "task_id": "secret_low_id", "expected": "low",
            "pair_id": "secret_pair", "dataset": "secret_dataset", "answer": "4",
        }
        body = jev.request_body(task, "model", jev.QUESTIONS)
        self.assertEqual(set(body), {"model", "state", "questions"})
        self.assertEqual(body["state"], task["prompt"])
        self.assertNotIn("secret", jev.canonical(body))
        self.assertNotIn('"answer"', jev.canonical(body))

    def test_error_rows_stay_in_accuracy_denominator(self):
        rows = [
            {"expected": "low", "predicted": "low", "correct": True},
            {"expected": "medium", "predicted": "high", "correct": False},
            {"expected": "high", "predicted": "low", "correct": False},
            {"expected": "high", "predicted": None, "correct": False},
        ]
        result = jev.metrics(rows)
        self.assertEqual(result["accuracy"], 0.25)
        self.assertEqual(result["valid_accuracy"], 1 / 3)
        self.assertEqual(result["errors"], 1)
        self.assertEqual(result["underestimated"], 1)
        self.assertEqual(result["overestimated"], 1)
        self.assertEqual(result["two_level_errors"], 1)
        self.assertEqual(result["per_class"]["low"]["precision"], 0.5)
        self.assertEqual(result["per_class"]["high"]["support"], 2)
        self.assertEqual(result["confusion"]["high"]["error"], 1)

    def test_invalid_choices_and_confidence_are_errors(self):
        for answer in [{"choice": "unknown"}, {"choice": "low", "confidence": 1.2},
                       {"choice": "low", "confidence": True}, {"choice": "low", "confidence": float("nan")}]:
            with self.subTest(answer=answer), self.assertRaises(ValueError):
                jev.parse_decision({"http_status": 200, "response": {"answers": {"difficulty": answer}}})
        with self.assertRaises(ValueError):
            jev.parse_decision({"http_status": 200, "response": {}})

    def test_pair_consistency_does_not_imply_accuracy(self):
        rows = [
            {"pair_id": "one", "lang": "en", "predicted": "high", "correct": False},
            {"pair_id": "one", "lang": "zh-Hant", "predicted": "high", "correct": False},
        ]
        result = jev.paired_metrics(rows)
        self.assertEqual(result["consistency"], 1)
        self.assertEqual(result["both_correct"], 0)
        self.assertEqual(result["accuracy_ci95_pair_bootstrap"], [0, 0])

    def test_resume_does_not_repeat_paid_calls_and_rejects_changed_inputs(self):
        tasks = [{"dataset": "dataset", "task_id": "one", "prompt": "Test"}]
        response = {"http_status": 200, "response": {"answers": {"difficulty": {"choice": "low"}}}}
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(
                run=Path(directory), datasets_root=Path("unused"), model="model",
                base_url="https://example.test/typesafe", workers=1, limit=None,
            )
            with patch.object(jev, "load_inputs", return_value=(tasks, {})), \
                 patch.object(jev, "api_key", return_value="test-key"), \
                 patch.object(jev, "call_api", return_value=response) as api, \
                 contextlib.redirect_stdout(io.StringIO()):
                jev.collect(args)
                calls = api.call_count
                jev.collect(args)
                self.assertEqual(api.call_count, calls)
                self.assertEqual(calls, 2)  # One model listing and one decision.
                tasks[0]["prompt"] = "Changed"
                with self.assertRaisesRegex(ValueError, "inputs changed"):
                    jev.collect(args)

    def test_report_rejects_a_response_for_a_different_request(self):
        tasks = [{"dataset": "dataset", "task_id": "one", "prompt": "Test", "expected": "low"}]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jev.write_json(root / "run_meta.json", {
                "inputs_sha256": jev.digest(jev.canonical(tasks).encode()),
                "model": "model", "questions": jev.QUESTIONS,
            })
            (root / "inputs.jsonl").write_text(jev.canonical(tasks[0]) + "\n")
            (root / "responses.jsonl").write_text(json.dumps({
                "dataset": "dataset", "task_id": "one", "request_sha256": "wrong",
            }) + "\n")
            args = argparse.Namespace(run=root, out=root / "report")
            with patch.object(jev, "call_api", side_effect=AssertionError("Report must stay offline")), \
                 self.assertRaisesRegex(ValueError, "Request hash"):
                jev.report(args)


if __name__ == "__main__":
    unittest.main()
