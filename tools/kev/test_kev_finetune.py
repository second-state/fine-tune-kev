import json
import hashlib
import tempfile
import unittest
from pathlib import Path

import kev_finetune_eval as evaluation
import prepare_kev_finetune as preparation
import report_kev_finetune as reporting
from jev_difficulty import QUESTIONS, canonical, load_inputs

ROOT = Path(__file__).resolve().parents[2]


class FinetuneDataTests(unittest.TestCase):
    def test_split_keeps_pairs_and_families_together_and_preserves_inputs(self):
        tasks, _ = load_inputs(ROOT / "datasets/routing_calibration")
        originals = {t["task_id"]: t for t in tasks}
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            manifest = preparation.prepare(ROOT / "datasets/routing_calibration", out)
            pairs, families, ids = {}, {}, set()
            for split in preparation.SPLITS:
                rows = [json.loads(s) for s in (out / f"{split}.jsonl").read_text().splitlines()]
                for row in rows:
                    meta = row["_meta"]
                    self.assertNotIn(meta["id"], ids)
                    ids.add(meta["id"])
                    self.assertEqual(pairs.setdefault(meta["pair_id"], split), split)
                    self.assertEqual(families.setdefault(meta["family_id"], split), split)
                    original = originals[meta["id"]]
                    self.assertEqual(row["state"], original["prompt"])
                    self.assertEqual(row["questions"]["difficulty"]["label"], original["expected"])
                    self.assertEqual(evaluation.inference_body(row), {"state": original["prompt"], "questions": QUESTIONS})
                self.assertEqual(set(manifest["counts"][split]["by_domain_level"]),
                                 {f"{domain}/{level}" for domain in ("architecture", "marketing", "reasoning")
                                  for level in preparation.LEVELS})
            self.assertEqual(ids, set(originals))
            self.assertEqual(len(pairs), 96)
            first = {p.name: p.read_bytes() for p in out.iterdir()}
            preparation.prepare(ROOT / "datasets/routing_calibration", out)
            self.assertEqual(first, {p.name: p.read_bytes() for p in out.iterdir()})

    def test_inference_strips_training_targets_and_metadata(self):
        row = {"state": "Prompt", "_meta": {"id": "gold_high"}, "questions": {
            "difficulty": {"type": "choice", "criteria": {"low": "Easy", "high": "Hard"},
                           "label": "high", "target": {"high": 1}, "src": "private"}}}
        body = evaluation.inference_body(row)
        self.assertEqual(set(body), {"state", "questions"})
        self.assertEqual(set(body["questions"]["difficulty"]), {"type", "criteria"})

    def test_bias_changes_argmax_and_handles_zero_probabilities(self):
        rows = [{"expected": "high", "probabilities": {"high": 0.4, "medium": 0.6, "low": 0.0}}]
        corrected = evaluation.predictions_with_bias(rows, {"high": 0.5})
        self.assertEqual(corrected[0]["predicted"], "high")
        self.assertTrue(corrected[0]["correct"])
        self.assertNotIn("predicted", rows[0])

    def test_selection_refuses_existing_test_predictions(self):
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            (work / "predictions").mkdir()
            (work / "predictions/parent-test.jsonl").write_text("{}\n")
            with self.assertRaisesRegex(ValueError, "test predictions exist"):
                reporting.select(work)

    def test_family_bootstrap_keeps_identical_predictions_equal(self):
        rows = [{"id": str(i), "family_id": str(i // 2), "expected": level,
                 "predicted": level, "correct": True}
                for i, level in enumerate(["low", "low", "medium", "medium", "high", "high"])]
        result = reporting.bootstrap(rows, rows, replicates=100)
        self.assertEqual(result["families"], 3)
        self.assertEqual(result["accuracy_delta_ci95"], [0, 0])
        self.assertEqual(result["macro_f1_delta_ci95"], [0, 0])

    def test_dev_selection_and_request_integrity_without_model_calls(self):
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            preparation.prepare(ROOT / "datasets/routing_calibration", work / "data")
            (work / "predictions").mkdir()
            for name, split in (("parent", "train"), ("parent", "dev"), ("lora-e1", "dev"), ("lora-e2", "dev")):
                data = work / f"data/{split}.jsonl"
                rows = []
                for i, record in enumerate(json.loads(s) for s in data.read_text().splitlines()):
                    expected = record["questions"]["difficulty"]["label"]
                    predicted = expected if name != "lora-e2" or i else ("high" if expected != "high" else "low")
                    probs = {level: 0.8 if level == predicted else 0.1 for level in preparation.LEVELS}
                    rows.append({**record["_meta"], "expected": expected, "predicted": predicted,
                                 "correct": expected == predicted, "probabilities": probs,
                                 "request_sha256": hashlib.sha256(canonical(evaluation.inference_body(record)).encode()).hexdigest()})
                output = work / f"predictions/{name}-{split}.jsonl"
                output.write_text("".join(json.dumps(r) + "\n" for r in rows))
                output.with_suffix(".meta.json").write_text(json.dumps({"data_sha256": reporting.file_hash(data)}))
            import contextlib
            import io
            with contextlib.redirect_stdout(io.StringIO()):
                reporting.select(work)
            selected = json.loads((work / "selection.json").read_text())
            self.assertEqual(selected["selected_checkpoint"], "lora-e1")
            self.assertFalse(selected["test_labels_used_for_selection"])
            target = work / "predictions/parent-dev.jsonl"
            rows = [json.loads(s) for s in target.read_text().splitlines()]
            rows[0]["request_sha256"] = "corrupted"
            target.write_text("".join(json.dumps(r) + "\n" for r in rows))
            with self.assertRaisesRegex(ValueError, "original prompt and rubric"):
                reporting.checked_rows(work, "parent", "dev")


if __name__ == "__main__":
    unittest.main()
