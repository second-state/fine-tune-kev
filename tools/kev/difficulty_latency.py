#!/usr/bin/env python3
"""Summarize the HTTP and server timings saved by difficulty evaluations."""

import argparse
import json
import math
from pathlib import Path
import statistics

from jev_difficulty import canonical, read_jsonl, write_json


def latency_stats(values):
    if not values:
        return None
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("Latency must be a finite, nonnegative number")
    values = sorted(values)
    return {
        "count": len(values),
        "mean_ms": statistics.mean(values),
        "median_ms": statistics.median(values),
        "p95_ms": values[math.ceil(0.95 * len(values)) - 1],
        "min_ms": values[0],
        "max_ms": values[-1],
    }


def summarize_run(run):
    meta = json.loads((run / "run_meta.json").read_text())
    records = read_jsonl(run / "responses.jsonl")
    keys = [(record["dataset"], record["task_id"]) for record in records]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate task response records")
    runtime = meta.get("runtime", {})
    label = runtime.get("checkpoint", meta["model"])
    rows = []
    for record in sorted(records, key=lambda row: (row["dataset"], row["task_id"])):
        for index, attempt in enumerate(record["attempts"], 1):
            response = attempt.get("response")
            server_ms = response.get("latency_ms") if isinstance(response, dict) else None
            header_ms = attempt.get("timing_headers", {}).get("x-inference-time-ms")
            source = None
            usage = response.get("usage") if isinstance(response, dict) else None
            if server_ms is None and isinstance(usage, dict) and usage.get("latency_ms") is not None:
                server_ms = usage["latency_ms"]
                source = "body:usage.latency_ms"
            if server_ms is None and header_ms is not None:
                server_ms = float(header_ms)
                source = "header:x-inference-time-ms"
            row = {
                "run": str(run), "model": label,
                "dataset": record["dataset"], "task_id": record["task_id"], "attempt": index,
                "http_status": attempt.get("http_status"),
                "client_latency_ms": attempt["latency_ms"],
                "server_latency_ms": server_ms,
                "request_ids": attempt.get("request_ids", {}),
            }
            if source:
                row["server_latency_source"] = source
            rows.append(row)
    summary = {
        "run": str(run), "model": label, "endpoint": meta["endpoint"],
        "task_count": len(records), "http_attempts": len(rows),
        "successful_http_attempts": sum(row["http_status"] == 200 for row in rows),
        "client": latency_stats([row["client_latency_ms"] for row in rows]),
        "server": latency_stats([row["server_latency_ms"] for row in rows if row["server_latency_ms"] is not None]),
        "server_latency_missing_count": sum(row["server_latency_ms"] is None for row in rows),
    }
    return summary, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if len(set(args.runs)) != len(args.runs):
        parser.error("Specify each run once")
    try:
        results = [summarize_run(run) for run in args.runs]
        output = {
            "units": "milliseconds",
            "p95_method": "Nearest rank: sorted values at ceil(0.95 * n) - 1.",
            "client_definition": "Timed HTTP attempt including network, service wait, response reading and JSON parsing. Excludes retry backoff and executor queue wait.",
            "server_definition": "Optional latency_ms reported in the native response. Kev measures model batch time, excluding queue wait, tokenization and HTTP handling. Missing measurements remain null.",
            "population": "All recorded HTTP attempts, including first requests and errors. Model download and startup are outside these timers.",
            "runs": [summary for summary, _ in results],
        }
        timer_sources = {row.get("server_latency_source") for _, rows in results for row in rows}
        if "header:x-inference-time-ms" in timer_sources:
            output["server_header_definition"] = (
                "Laya uses X-Inference-Time-Ms when the response has no latency_ms. "
                "It includes worker dispatch, routing, tokenization, inference and answer formatting. "
                "It excludes HTTP body reading and inference-gate wait. "
                "This scope differs from Kev model batch time."
            )
        if "body:usage.latency_ms" in timer_sources:
            output["server_usage_definition"] = (
                "Clef MLX reports usage.latency_ms. It includes inference-lock wait, "
                "tokenization, backbone and decision-head execution, and answer formatting. "
                "It excludes HTTP body reading, JSON decoding, and response serialization. "
                "This scope differs from Kev model batch time."
            )
        args.out.mkdir(parents=True, exist_ok=True)
        write_json(args.out / "latency_summary.json", output)
        (args.out / "request_latencies.jsonl").write_text(
            "".join(canonical(row) + "\n" for _, rows in results for row in rows)
        )
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f"Error: {error}\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
