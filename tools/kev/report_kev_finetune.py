#!/usr/bin/env python3
"""Select Kev checkpoints on dev, then summarize saved held-out predictions."""
import argparse
import collections
import hashlib
import json
import random
import shutil
from pathlib import Path

from difficulty_latency import latency_stats
from jev_difficulty import canonical, metrics, read_jsonl, write_json
from kev_finetune_eval import fit_bias, inference_body, predictions_with_bias


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_rows(work, name, split):
    rows = read_jsonl(work / "predictions" / f"{name}-{split}.jsonl")
    manifest = json.loads((work / "data/split_manifest.json").read_text())
    data = work / f"data/{split}.jsonl"
    prediction_meta = json.loads((work / "predictions" / f"{name}-{split}.meta.json").read_text())
    if file_hash(data) != manifest["files_sha256"][data.name] or prediction_meta["data_sha256"] != file_hash(data):
        raise ValueError("Prediction data differs from the frozen split")
    requests = {r["_meta"]["id"]: hashlib.sha256(canonical(inference_body(r)).encode()).hexdigest() for r in read_jsonl(data)}
    expected = {r["id"]: r for r in manifest["records"] if r["split"] == split}
    if len(rows) != len(expected) or len({r["id"] for r in rows}) != len(rows):
        raise ValueError(f"Missing or duplicate rows: {name}/{split}")
    for row in rows:
        target = expected[row["id"]]
        if any(row[k] != target[k] for k in ("expected", "family_id", "pair_id", "lang", "domain", "split")):
            raise ValueError(f"Prediction identity mismatch: {row['id']}")
        if row["correct"] != (row["predicted"] == row["expected"]):
            raise ValueError("Incorrect correctness field")
        if row["request_sha256"] != requests[row["id"]]:
            raise ValueError("Prediction request differs from the original prompt and rubric")
    return rows


def select(work):
    path = work / "selection.json"
    if path.exists() or list((work / "predictions").glob("*-test.jsonl")):
        raise ValueError("Selection is already frozen or test predictions exist")
    bias = fit_bias(checked_rows(work, "parent", "train"), checked_rows(work, "parent", "dev"))
    candidates = {name: metrics(checked_rows(work, name, "dev")) for name in ("lora-e1", "lora-e2")}
    selected = max(candidates, key=lambda name: (candidates[name]["macro_f1"], candidates[name]["accuracy"], -int(name[-1])))
    inputs = {str(p.relative_to(work)): file_hash(p) for p in sorted((work / "predictions").glob("*.jsonl"))}
    result = {"selected_checkpoint": selected, "selection_inputs_sha256": inputs,
              "split_manifest_sha256": file_hash(work / "data/split_manifest.json"),
              "development": {"parent": metrics(checked_rows(work, "parent", "dev")), **candidates},
              "class_bias": bias, "test_labels_used_for_selection": False}
    write_json(path, result)
    print(json.dumps({"selected_checkpoint": selected, "dev": candidates,
                      "bias": bias["selected"]["bias"]}, indent=2))


def summarize(rows):
    pairs = collections.defaultdict(list)
    for row in rows:
        pairs[row["pair_id"]].append(row)
    if any(len(pair) != 2 or {r["lang"] for r in pair} != {"en", "zh-Hant"} for pair in pairs.values()):
        raise ValueError("Incomplete language pair")
    return {**metrics(rows),
            "by_domain": {domain: metrics([r for r in rows if r["domain"] == domain]) for domain in sorted({r["domain"] for r in rows})},
            "by_language": {lang: metrics([r for r in rows if r["lang"] == lang]) for lang in ("en", "zh-Hant")},
            "pairs": {"total": len(pairs), "consistent": sum(p[0]["predicted"] == p[1]["predicted"] for p in pairs.values()),
                      "both_correct": sum(all(r["correct"] for r in p) for p in pairs.values())}}


def bootstrap(reference, candidate, replicates=5000):
    ref = {r["id"]: r for r in reference}
    cand = {r["id"]: r for r in candidate}
    if set(ref) != set(cand):
        raise ValueError("Paired comparison requires identical task IDs")
    families = collections.defaultdict(list)
    for key, row in sorted(ref.items()):
        if row["family_id"] != cand[key]["family_id"] or row["expected"] != cand[key]["expected"]:
            raise ValueError("Paired rows disagree on family or label")
        families[row["family_id"]].append(key)
    rng = random.Random(42)
    names = sorted(families)
    draws = {"accuracy_delta": [], "macro_f1_delta": []}
    for _ in range(replicates):
        keys = [key for group in rng.choices(names, k=len(names)) for key in families[group]]
        left, right = (metrics([lookup[key] for key in keys]) for lookup in (ref, cand))
        for metric in ("accuracy", "macro_f1"):
            draws[f"{metric}_delta"].append(right[metric] - left[metric])
    return {"unit": "family", "families": len(names), "replicates": replicates, "seed": 42,
            "note": "Percentile intervals; fixed three-class macro-F1, including zero for absent classes. Few families limit inference.",
            **{key + "_ci95": [sorted(values)[int(0.025 * replicates)], sorted(values)[int(0.975 * replicates) - 1]] for key, values in draws.items()}}


def report(work, out):
    selection = json.loads((work / "selection.json").read_text())
    for filename, expected_hash in selection["selection_inputs_sha256"].items():
        if file_hash(work / filename) != expected_hash:
            raise ValueError("Model-selection inputs changed")
    if file_hash(work / "data/split_manifest.json") != selection["split_manifest_sha256"]:
        raise ValueError("Frozen split changed")
    chosen = selection["selected_checkpoint"]
    parent = checked_rows(work, "parent", "test")
    candidate = checked_rows(work, chosen, "test")
    reference_requests = {r["id"]: r["request_sha256"] for r in parent}
    if any(reference_requests[r["id"]] != r["request_sha256"] for r in candidate):
        raise ValueError("Parent and candidate received different requests")
    biased = predictions_with_bias(parent, selection["class_bias"]["selected"]["bias"])
    versions = {"parent": parent, "parent_class_bias": biased, chosen: candidate}
    result = {"selection": selection,
              "test": {name: summarize(rows) for name, rows in versions.items()},
              "test_latency": {name: {timer: latency_stats([r[timer] for r in rows]) for timer in ("forward_ms", "predictor_ms")}
                               for name, rows in (("parent", parent), (chosen, candidate))},
              "paired_family_bootstrap": {"lora_minus_parent": bootstrap(parent, candidate),
                                          "bias_minus_parent": bootstrap(parent, biased),
                                          "lora_minus_bias": bootstrap(biased, candidate)},
              "training": {name: {"process": json.loads((work / f"{name}.process.json").read_text()),
                                  "loop": json.loads((work / name / "training_metrics.json").read_text())}
                           for name in ("preflight", "lora-e1", "lora-e2") if (work / name / "training_metrics.json").exists()}}
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "summary.json", result)
    saved = out / "predictions"
    saved.mkdir(exist_ok=True)
    for path in sorted((work / "predictions").glob("*.json*")):
        shutil.copyfile(path, saved / path.name)
    for name in ("selection.json", "data_validation.json", "download.json", "runtime.json"):
        if (work / name).exists():
            shutil.copyfile(work / name, out / name)
    if (work / "data/preflight.jsonl").exists():
        shutil.copyfile(work / "data/preflight.jsonl", out / "preflight.jsonl")
    with (out / "test_predictions.jsonl").open("w", encoding="utf-8") as stream:
        for name, rows in versions.items():
            for row in rows:
                # Bias uses the parent's inference; it has no separately measured model latency.
                saved = {k: v for k, v in row.items() if name != "parent_class_bias" or k not in ("forward_ms", "predictor_ms")}
                stream.write(json.dumps({"method": name, **saved}, ensure_ascii=False) + "\n")
    evidence = {str(p.relative_to(work)): file_hash(p) for p in sorted(work.rglob("*.json*"))
                if "hf-cache" not in p.parts and p.is_file()}
    write_json(out / "evidence_sha256.json", evidence)
    print(json.dumps({name: {k: result["test"][name][k] for k in ("correct", "total", "accuracy", "macro_f1")}
                      for name in versions}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("select", "report"))
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.action == "select":
        select(args.work)
    elif args.out is None:
        parser.error("report requires --out")
    else:
        report(args.work, args.out)
