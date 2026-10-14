"""Download pinned public checkpoints into an experiment-only cache."""
import argparse
import fnmatch
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--work", type=Path, required=True)
args = parser.parse_args()
WORK = args.work.resolve()
WORK.mkdir(parents=True, exist_ok=True)
CACHE = WORK / "hf-cache"
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
from huggingface_hub import HfApi, snapshot_download

MODELS = {
    "base": ("Qwen/Qwen3.5-4B-Base", "1001bb4d826a52d1f399e183466143f4da7b741b"),
    "parent": ("jaredpalmer/kev-4b", "139fdd94f1b6a6ad80cc15e08fcb99cac885a101"),
}
PATTERNS = ["*.json", "*.safetensors", "*.pt", "*.txt", "*.jinja"]
infos = {key: HfApi(token=False).model_info(repo, revision=rev, files_metadata=True)
         for key, (repo, rev) in MODELS.items()}
size = sum(f.size or 0 for info in infos.values() for f in info.siblings
           if any(fnmatch.fnmatch(f.rfilename, p) for p in PATTERNS))
free = shutil.disk_usage(WORK).free
if not CACHE.exists() and free < size + 2 * 1024**3:
    raise SystemExit(f"Insufficient headroom: {free=}, {size=}")
print(f"Download {size / 1024**3:.2f} GiB; free {free / 1024**3:.2f} GiB", flush=True)
result = {"expected_bytes": size, "free_bytes_before": free, "models": {}}
start = time.monotonic()
for key, (repo, rev) in MODELS.items():
    directory = Path(snapshot_download(repo, revision=rev, cache_dir=CACHE,
                                      allow_patterns=PATTERNS, token=False, max_workers=1))
    files = {}
    for file in sorted(directory.rglob("*")):
        if file.is_file():
            with file.open("rb") as stream:
                sha = hashlib.file_digest(stream, "sha256").hexdigest()
            files[str(file.relative_to(directory))] = {"bytes": file.stat().st_size, "sha256": sha}
    result["models"][key] = {"repo": repo, "revision": rev, "path": str(directory), "files": files}
    (WORK / "download.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Ready: {key}", flush=True)
result.update(wall_seconds=time.monotonic() - start, free_bytes_after=shutil.disk_usage(WORK).free)
(WORK / "download.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print("Download and hashes complete", flush=True)
