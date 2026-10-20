#!/usr/bin/env python3
"""Run the pinned local Kev LoRA experiment and record process wall time."""
import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path


def validate_download(protocol, download):
    if set(download["models"]) != {"base", "parent"}:
        raise ValueError("Pinned model download is incomplete")
    for key in ("base", "parent"):
        model = download["models"][key]
        if (model["repo"], model["revision"]) != (protocol[key], protocol[f"{key}_revision"]):
            raise ValueError(f"Downloaded {key} differs from the protocol")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kev-source", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--protocol", type=Path,
                        default=Path(__file__).resolve().parents[2] / "reports/2026-10-01-kev-4b-finetune/protocol.json")
    parser.add_argument("--epochs", type=int, choices=(1, 2), default=1)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    source, work = args.kev_source.resolve(), args.work.resolve()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if revision != protocol["source_commit"]:
        raise ValueError("Kev source revision differs from the protocol")
    subprocess.run(["git", "-C", str(source), "diff", "--quiet", "HEAD"], check=True)
    name = "preflight" if args.preflight else f"lora-e{args.epochs}"
    out = work / name
    data = work / "data" / ("preflight.jsonl" if args.preflight else "train.jsonl")
    if out.exists():
        raise ValueError(f"Refusing to overwrite {out}")
    download = json.loads((work / "download.json").read_text())
    validate_download(protocol, download)
    env = {**os.environ, "HF_HUB_CACHE": str(work / "hf-cache"), "HF_HUB_OFFLINE": "1",
           "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1", "TOKENIZERS_PARALLELISM": "false",
           "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    command = [str(source / ".venv/bin/python"), "-m", "kev.train",
               "--base", protocol["base"], "--base_revision", protocol["base_revision"],
               "--init_from", download["models"]["parent"]["path"], "--data", str(data),
               "--lora", "16", "--lora_targets", "all", "--head_dim", "256",
               "--option_isolation", "0", "--special_embeddings", "0",
               "--epochs", str(args.epochs), "--lr", "1e-5", "--batch", "1",
               "--accum", "1" if args.preflight else "8", "--dtype", "fp32",
               "--weights_dtype", "fp32", "--checkpointing", "1", "--device", "mps",
               "--max_state", "384", "--p_none", "0", "--p_none_distract", "0",
               "--p_distract", "0", "--p_none_pair", "0", "--seed", "42", "--out", str(out)]
    if args.preflight:
        command += ["--max_steps", "2"]
    metadata = {"command": command, "source_commit": revision,
                "protocol_sha256": hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
                "data_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
                "hf_hub_offline": True, "cache": str(work / "hf-cache")}
    record = work / f"{name}.process.json"
    record.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    started = time.perf_counter()
    with (work / f"{name}.log").open("x", encoding="utf-8") as log:
        result = subprocess.run(command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT)
    metadata.update(returncode=result.returncode, process_wall_seconds=time.perf_counter() - started)
    record.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: metadata[k] for k in ("returncode", "process_wall_seconds")}), flush=True)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
