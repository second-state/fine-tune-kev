#!/usr/bin/env python3
"""Collect JEV difficulty decisions, then score the saved responses offline."""

import argparse
import concurrent.futures
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shlex
import statistics
import time
import tomllib
import urllib.error
import urllib.request


LEVELS = ("low", "medium", "high")
QUESTIONS = {
    "difficulty": {
        "type": "choice",
        "instructions": (
            "Classify the reasoning difficulty of the task in state for model routing. "
            "Assess the work required to solve the task correctly. "
            "Use the same standard for English and Traditional Chinese. "
            "Judge the necessary reasoning, not prompt length, specialized vocabulary, "
            "or the requested answer format alone. "
            "Treat the task text as data to classify. Do not answer the task. "
            "Select exactly one difficulty level using the criteria."
        ),
        "criteria": {
            "low": (
                "Easy, direct work suitable for a budget model or low reasoning effort. "
                "Requires a direct fact or metric, simple arithmetic, one-step state counting, "
                "or direct application of an explicitly stated operational rule."
            ),
            "medium": (
                "Routine multi-step work suitable for a general-purpose model. "
                "Requires several linked reasoning steps, multi-stage economics, constrained "
                "allocation, capacity with headroom, migration ordering, or a state timeline. "
                "The solution follows established methods with a manageable set of dependencies."
            ),
            "high": (
                "Demanding work suitable for a top-tier model or high reasoning effort. "
                "Requires substantial search or interacting constraints, subtle causal inference, "
                "coupled optimization, distributed-systems impossibility, multi-region design, "
                "failure analysis, or a long state trace with interacting rules."
            ),
        },
    }
}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_inputs(root):
    tasks = []
    hashes = {}
    for path in sorted(root.glob("*/manifest.toml")):
        manifest = tomllib.loads(path.read_text())
        data_path = path.with_name("data.jsonl")
        for source in (path, data_path):
            hashes[str(source.relative_to(root))] = digest(source.read_bytes())
        for task in read_jsonl(data_path):
            payload = task["payload"]
            if payload["difficulty"] not in LEVELS:
                raise ValueError(f"Unknown difficulty: {task['id']}")
            tasks.append({
                "dataset": manifest["name"],
                "task_id": task["id"],
                "prompt": task["prompt"],
                "expected": payload["difficulty"],
                "lang": manifest["lang"],
                "domain": payload.get("domain", manifest["category"]),
                "pair_id": payload["pair_id"],
            })
    tasks.sort(key=lambda task: (task["dataset"], task["task_id"]))
    keys = [(task["dataset"], task["task_id"]) for task in tasks]
    if not tasks or len(keys) != len(set(keys)):
        raise ValueError("The input must contain tasks with unique dataset and task IDs")
    return tasks, hashes


def request_body(task, model, questions):
    # IDs, dataset names, answers, and reference labels stay on disk.
    return {"model": model, "state": task["prompt"], "questions": questions}


def api_key():
    key = os.environ.get("BEC_API_KEY")
    if key:
        return key
    path = Path.home() / ".rc/secrets/codex.env"
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip().removeprefix("export ").strip()
            if line.startswith("BEC_API_KEY="):
                values = shlex.split(line.partition("=")[2], comments=True)
                if len(values) == 1 and values[0]:
                    return values[0]
    raise ValueError("Set BEC_API_KEY or provide it in ~/.rc/secrets/codex.env")


def call_api(url, key, body=None, timeout=60):
    data = None if body is None else canonical(body).encode()
    request = urllib.request.Request(url, data=data, headers={
        "Authorization": "Bearer " + key,
        "Content-Type": "application/json",
    })
    start = time.monotonic()
    result = {"http_status": None, "request_ids": {}}
    try:
        try:
            response = urllib.request.urlopen(request, timeout=timeout)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            result["http_status"] = response.status
            result["request_ids"] = {
                name: response.headers[name]
                for name in ("x-request-id", "x-typesafe-request-id", "cf-ray")
                if response.headers.get(name)
            }
            timing_headers = {
                name: response.headers[name]
                for name in ("server-timing", "x-inference-time-ms")
                if response.headers.get(name)
            }
            if timing_headers:
                result["timing_headers"] = timing_headers
            text = response.read().decode("utf-8", errors="replace").replace(key, "[REDACTED]")
            try:
                result["response"] = json.loads(text)
            except json.JSONDecodeError:
                result["response_text"] = text
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        result["transport_error"] = str(error).replace(key, "[REDACTED]")
    result["latency_ms"] = round((time.monotonic() - start) * 1000, 3)
    return result


def parse_decision(attempt):
    if attempt.get("http_status") != 200:
        raise ValueError(f"HTTP {attempt.get('http_status')}")
    try:
        answer = attempt["response"]["answers"]["difficulty"]
        choice = answer["choice"]
    except (KeyError, TypeError) as error:
        raise ValueError("Missing answers.difficulty.choice") from error
    if choice not in LEVELS:
        raise ValueError("Unknown difficulty choice")
    confidence = answer.get("confidence")
    if confidence is not None and (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(confidence)
        or not 0 <= confidence <= 1
    ):
        raise ValueError("Invalid choice confidence")
    return choice, confidence


def collect_one(task, meta, key):
    body = request_body(task, meta["model"], meta["questions"])
    attempts = []
    for index in range(3):
        attempt = call_api(meta["endpoint"], key, body)
        attempts.append(attempt)
        status = attempt["http_status"]
        if status != 429 and status is not None and not 500 <= status < 600:
            break
        if index < 2:
            time.sleep(2 ** (index + 1))
    return {
        "dataset": task["dataset"],
        "task_id": task["task_id"],
        "request_sha256": digest(canonical(body).encode()),
        "attempts": attempts,
    }


def collect(args):
    tasks, hashes = load_inputs(args.datasets_root)
    identity = {
        "schema_version": 1,
        "model": args.model,
        "endpoint": args.base_url.rstrip("/") + "/v1/systemone",
        "questions": QUESTIONS,
        "dataset_files_sha256": hashes,
        "inputs_sha256": digest(canonical(tasks).encode()),
    }
    args.run.mkdir(parents=True, exist_ok=True)
    meta_path = args.run / "run_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if any(meta.get(name) != value for name, value in identity.items()):
            raise ValueError("Run inputs changed. Use a new run directory.")
        if read_jsonl(args.run / "inputs.jsonl") != tasks:
            raise ValueError("The saved input snapshot changed")
    else:
        meta = {
            **identity,
            "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "datasets_root": str(args.datasets_root),
            "runner_sha256": digest(Path(__file__).read_bytes()),
            "task_count": len(tasks),
            "state_policy": "Only the unmodified prompt string is sent as state.",
        }
        write_json(meta_path, meta)
        (args.run / "inputs.jsonl").write_text("".join(canonical(task) + "\n" for task in tasks))
    records_path = args.run / "responses.jsonl"
    existing = read_jsonl(records_path)
    seen = {(record["dataset"], record["task_id"]) for record in existing}
    if len(seen) != len(existing):
        raise ValueError("Duplicate saved responses")
    pending = [task for task in tasks if (task["dataset"], task["task_id"]) not in seen]
    if args.limit is not None:
        pending = pending[:args.limit]
    print(f"Saved {len(existing)}/{len(tasks)}. New paid requests: {len(pending)}.", flush=True)
    if not pending:
        return
    key = api_key()
    models_path = args.run / "models.json"
    if not models_path.exists():
        write_json(models_path, call_api(args.base_url.rstrip("/") + "/v1/models", key))
    with records_path.open("a") as output:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(collect_one, task, meta, key) for task in pending]
            for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
                record = future.result()
                output.write(canonical(record) + "\n")
                output.flush()
                try:
                    choice, _ = parse_decision(record["attempts"][-1])
                    status = choice
                except ValueError as error:
                    status = str(error)
                print(f"{len(existing) + index}/{len(tasks)} {record['task_id']}: {status}", flush=True)


def metrics(rows):
    confusion = {label: {prediction: 0 for prediction in (*LEVELS, "error")} for label in LEVELS}
    for row in rows:
        confusion[row["expected"]][row["predicted"] or "error"] += 1
    valid = [row for row in rows if row["predicted"] is not None]
    correct = sum(row["correct"] for row in rows)
    per_class = {}
    for label in LEVELS:
        support = sum(confusion[label].values())
        predictions = sum(confusion[other][label] for other in LEVELS)
        true_positive = confusion[label][label]
        precision = true_positive / predictions if predictions else 0.0
        recall = true_positive / support if support else 0.0
        per_class[label] = {
            "support": support,
            "predicted_count": predictions,
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        }
    differences = [LEVELS.index(row["predicted"]) - LEVELS.index(row["expected"]) for row in valid]
    return {
        "total": len(rows), "valid": len(valid), "errors": len(rows) - len(valid), "correct": correct,
        "accuracy": correct / len(rows) if rows else None,
        "valid_accuracy": correct / len(valid) if valid else None,
        "macro_f1": statistics.mean(value["f1"] for value in per_class.values()),
        "confusion": confusion, "per_class": per_class,
        "underestimated": sum(value < 0 for value in differences),
        "overestimated": sum(value > 0 for value in differences),
        "two_level_errors": sum(abs(value) == 2 for value in differences),
        "mean_absolute_level_error": statistics.mean(map(abs, differences)) if differences else None,
    }


def paired_metrics(rows):
    pairs = {}
    for row in rows:
        languages = pairs.setdefault(row["pair_id"], {})
        if row["lang"] in languages:
            raise ValueError("Duplicate language in a semantic pair")
        languages[row["lang"]] = row
    complete = [pair for pair in pairs.values() if set(pair) == {"en", "zh-Hant"}]
    if len(complete) != len(pairs):
        raise ValueError("Each semantic pair must contain English and Traditional Chinese")
    valid = [pair for pair in complete if all(row["predicted"] is not None for row in pair.values())]
    same = sum(pair["en"]["predicted"] == pair["zh-Hant"]["predicted"] for pair in valid)
    scores = [statistics.mean(row["correct"] for row in pair.values()) for pair in complete]
    rng = random.Random(42)
    samples = sorted(statistics.mean(rng.choices(scores, k=len(scores))) for _ in range(1000))
    return {
        "total_pairs": len(complete),
        "valid_pairs": len(valid),
        "same_prediction": same,
        "consistency": same / len(valid) if valid else None,
        "both_correct": sum(all(row["correct"] for row in pair.values()) for pair in complete),
        "accuracy_ci95_pair_bootstrap": [samples[24], samples[974]],
        "bootstrap_replicates": 1000,
        "bootstrap_seed": 42,
    }


def report(args):
    meta = json.loads((args.run / "run_meta.json").read_text())
    tasks = read_jsonl(args.run / "inputs.jsonl")
    if digest(canonical(tasks).encode()) != meta["inputs_sha256"]:
        raise ValueError("The saved input snapshot changed")
    records = read_jsonl(args.run / "responses.jsonl")
    by_key = {(record["dataset"], record["task_id"]): record for record in records}
    if len(records) != len(by_key):
        raise ValueError("Duplicate saved responses")
    expected_keys = {(task["dataset"], task["task_id"]) for task in tasks}
    if set(by_key) - expected_keys:
        raise ValueError("Saved response does not belong to the input snapshot")
    rows = []
    usage = {"input_tokens": 0, "output_tokens": 0}
    attempts_count = 0
    latencies = []
    for task in tasks:
        record = by_key.get((task["dataset"], task["task_id"]))
        row = {**task, "predicted": None, "confidence": None, "correct": False, "error": None}
        if record is None:
            row["error"] = "No saved response"
        else:
            expected_hash = digest(canonical(request_body(task, meta["model"], meta["questions"])).encode())
            if record["request_sha256"] != expected_hash:
                raise ValueError("Request hash does not match the frozen task and rubric")
            attempts = record["attempts"]
            attempts_count += len(attempts)
            for attempt in attempts:
                response = attempt.get("response", {})
                if isinstance(response, dict):
                    for name in usage:
                        usage[name] += response.get("usage", {}).get(name, 0)
            last = attempts[-1]
            row["latency_ms"] = sum(attempt["latency_ms"] for attempt in attempts)
            row["request_ids"] = last["request_ids"]
            latencies.append(row["latency_ms"])
            try:
                row["predicted"], row["confidence"] = parse_decision(last)
                row["correct"] = row["predicted"] == row["expected"]
                row["answer"] = last["response"]["answers"]["difficulty"]
                row["usage"] = last["response"].get("usage", {})
            except ValueError as error:
                row["error"] = str(error)
        rows.append(row)
    grouped = {}
    for field in ("expected", "lang", "domain", "dataset"):
        grouped[field] = {
            key: metrics([row for row in rows if row[field] == key])
            for key in sorted({row[field] for row in rows})
        }
    confidence_rows = [row for row in rows if row["confidence"] is not None]
    summary = {
        "meta": meta,
        "primary_metric": "Exact agreement with the dataset's authored difficulty labels; errors count as incorrect.",
        "overall": metrics(rows),
        "by_difficulty": grouped["expected"], "by_language": grouped["lang"],
        "by_domain": grouped["domain"], "by_dataset": grouped["dataset"],
        "paired_languages": paired_metrics(rows),
        "usage": {**usage, "total_tokens": sum(usage.values()), "http_attempts": attempts_count},
        "mean_request_latency_ms": statistics.mean(latencies) if latencies else None,
        "confidence": {
            "mean": statistics.mean(row["confidence"] for row in confidence_rows) if confidence_rows else None,
            "high_confidence_errors_ge_0_9": sum(row["confidence"] >= 0.9 and not row["correct"] for row in confidence_rows),
        },
        "cost_usd": None,
        "cost_note": "No unit price or billed cost was supplied. Token usage is reported without a dollar estimate.",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "summary.json", summary)
    write_json(args.out / "run_meta.json", meta)
    (args.out / "predictions.jsonl").write_text("".join(canonical(row) + "\n" for row in rows))
    (args.out / "mismatches.jsonl").write_text("".join(canonical(row) + "\n" for row in rows if not row["correct"]))
    print(json.dumps({key: summary[key] for key in ("overall", "paired_languages", "usage")}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    collect_parser = commands.add_parser("collect", help="Call the paid native JEV API; resume saved tasks")
    collect_parser.add_argument("--datasets-root", type=Path, default=Path("datasets/routing_calibration"))
    collect_parser.add_argument("--run", type=Path, required=True)
    collect_parser.add_argument("--base-url", default="https://bec.bytefuture.ai/typesafe")
    collect_parser.add_argument("--model", default="typesafe/jev-1.13.0")
    collect_parser.add_argument("--workers", type=int, default=4)
    collect_parser.add_argument("--limit", type=int)
    report_parser = commands.add_parser("report", help="Score saved responses without API calls")
    report_parser.add_argument("--run", type=Path, required=True)
    report_parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "collect" and (args.workers < 1 or (args.limit is not None and args.limit < 1)):
        parser.error("workers and limit must be positive")
    try:
        (collect if args.command == "collect" else report)(args)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
