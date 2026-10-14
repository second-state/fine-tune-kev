#!/usr/bin/env python3
"""Collect local Kev decisions. Saved predictions support offline scoring."""
import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

from jev_difficulty import LEVELS, canonical, metrics, read_jsonl, write_json


def inference_body(record):
    return {"state": record["state"], "questions": {
        name: {key: value for key, value in q.items() if key in ("type", "instructions", "criteria")}
        for name, q in record["questions"].items()}}


def predictions_with_bias(rows, bias):
    result = []
    for row in rows:
        scores = {key: math.log(max(prob, 1e-12)) + bias.get(key, 0.0)
                  for key, prob in row["probabilities"].items()}
        predicted = max(scores, key=scores.get)
        result.append({**row, "predicted": predicted, "correct": predicted == row["expected"],
                       "class_bias": dict(bias), "biased_log_scores": scores})
    return result


def fit_bias(train, dev):
    """Fit on train and select regularization on dev. No test input is accepted."""
    candidates = []
    grid = []
    for high in range(-12, 13):
        for low in range(-12, 13):
            bias = {"high": high / 4, "low": low / 4, "medium": 0.0}
            summary = metrics(predictions_with_bias(train, bias))
            grid.append((bias, summary, bias["high"] ** 2 + bias["low"] ** 2))
    for regularization in (0.0, 0.001, 0.01, 0.1):
        bias, training, norm = max(grid, key=lambda item: (
            item[1]["macro_f1"] - regularization * item[2], item[1]["accuracy"], -item[2],
            -abs(item[0]["high"]), -abs(item[0]["low"])))
        development = metrics(predictions_with_bias(dev, bias))
        candidates.append({"regularization": regularization, "bias": bias, "squared_norm": norm,
                           "train": training, "dev": development})
    selected = max(candidates, key=lambda c: (c["dev"]["macro_f1"], c["dev"]["accuracy"], -c["squared_norm"]))
    return {"selected": selected, "candidates": candidates}


def collect(args):
    sys.path.insert(0, str(args.kev_source.resolve()))
    import torch
    from kev.api import SystemOneRequest, to_record
    from kev.checkpoint import Checkpoint, LoadOptions
    from kev.device import sync
    from kev.model import training_context

    if args.out.exists():
        raise ValueError(f"Refusing to overwrite {args.out}")
    inputs = read_jsonl(args.data)
    warmups = read_jsonl(args.warmup)
    if not inputs or not warmups:
        raise ValueError("Evaluation and warmup data must not be empty")
    if any(r["_meta"]["split"] != "train" for r in warmups):
        raise ValueError("Warmup must use training records")
    checkpoint = Checkpoint(args.run)
    initialized = time.perf_counter()
    tok, model = checkpoint.load("mps", LoadOptions(backend="mlx", temperature=1.0))
    initialization = time.perf_counter() - initialized
    context = training_context()

    @torch.no_grad()
    def predict(record):
        body = inference_body(record)
        started = time.perf_counter()
        internal, meta = to_record(SystemOneRequest.model_validate(body))
        enc = model.encode(tok, internal, max_state=context["max_state"], max_branch=context["max_branch"], strict=True)
        if len(enc["ids"]) > context["max_packed"]:
            raise ValueError("Context overflow")
        sync("mps")
        forward_started = time.perf_counter()
        logits = model.forward(enc)[0]
        probabilities = torch.softmax(logits.float(), -1).cpu().tolist()
        values = logits.float().cpu().tolist()
        sync("mps")
        forward_ms = (time.perf_counter() - forward_started) * 1000
        probs = dict(zip(meta[0]["keys"], probabilities))
        if set(probs) != set(LEVELS) or any(not math.isfinite(p) for p in probabilities):
            raise ValueError("Invalid difficulty distribution")
        return {"probabilities": probs, "logits": dict(zip(meta[0]["keys"], values)),
                "forward_ms": forward_ms, "predictor_ms": (time.perf_counter() - started) * 1000,
                "input_tokens": len(enc["ids"]), "request_sha256": hashlib.sha256(canonical(body).encode()).hexdigest()}

    warmup = predict(warmups[0])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    meta = {"checkpoint": args.run, "checkpoint_weights_sha256": checkpoint.weights_sha256(),
            "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
            "warmup_id": warmups[0]["_meta"]["id"], "warmup": warmup,
            "initialization_seconds": initialization, "backend": "mlx", "backbone_dtype": "bf16",
            "temperature": 1.0, "merge": True, "cross_request_prefix_cache": False,
            "timer": "forward_ms includes model forward, pointer head, softmax and synchronization; predictor_ms also includes parsing and encoding. Neither includes HTTP, initialization or warmup."}
    write_json(args.out.with_suffix(".meta.json"), meta)
    with args.out.open("x", encoding="utf-8") as stream:
        for index, record in enumerate(inputs):
            prediction = predict(record)
            predicted = max(prediction["probabilities"], key=prediction["probabilities"].get)
            expected = record["questions"]["difficulty"]["label"]
            row = {**record["_meta"], "task_id": record["_meta"]["id"], "expected": expected,
                   **prediction, "predicted": predicted, "correct": predicted == expected}
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            if (index + 1) % 20 == 0:
                print(f"Scored {index + 1}/{len(inputs)}", flush=True)
    print(json.dumps(metrics(read_jsonl(args.out)), indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kev-source", type=Path, required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--warmup", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    collect(parser.parse_args())
