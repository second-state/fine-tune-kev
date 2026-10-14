#!/usr/bin/env python3
"""Freeze paired, domain-family-disjoint Kev training data. No model calls."""
import argparse
import collections
import copy
import hashlib
import json
import random
from pathlib import Path

from jev_difficulty import QUESTIONS, canonical, load_inputs

LEVELS = ("low", "medium", "high")
SPLITS = ("train", "dev", "test")

# Manual review of all English originals. Chinese translations inherit the group.
# Families group related scenarios and close task variants, not every shared skill.
FAMILIES = {
    "reasoning.arithmetic": "low_001 low_002 low_003 low_006 low_015",
    "reasoning.proportions": "low_004 low_005 medium_009 medium_015",
    "reasoning.units_rates": "low_007 low_012 medium_010",
    "reasoning.sequences_traces": "low_009 medium_006 medium_016 high_016",
    "reasoning.logical_constraints": "low_010 medium_002 medium_005 medium_007 high_011",
    "reasoning.string_counts": "low_011 high_002 high_007",
    "reasoning.list_statistics": "low_008 low_013 low_016",
    "reasoning.geometry": "low_014 medium_012",
    "reasoning.arrangements_schedules": "medium_001 medium_014 high_001 high_003 high_008",
    "reasoning.set_counts": "medium_011",
    "reasoning.congruences": "medium_004",
    "reasoning.shortest_paths": "medium_008 high_004 high_005",
    "reasoning.probability": "medium_003 medium_013 high_006 high_010 high_013",
    "reasoning.project_selection": "high_009 high_015",
    "reasoning.games": "high_012 high_014",
    "architecture.idempotent_effects": "low_001 low_002 low_006 medium_005 medium_006 high_001 high_005",
    "architecture.capacity_slo": "low_008 medium_001 medium_008 high_002 high_008",
    "architecture.caches": "low_003 medium_002 high_003",
    "architecture.migrations": "low_004 medium_003 high_004",
    "architecture.quorums": "low_005 medium_004 high_006",
    "architecture.tenants": "low_007 medium_007 high_007",
    "marketing.funnels": "low_001 low_002 medium_001 high_001",
    "marketing.causal_rate_comparisons": "low_003 low_004 low_007 medium_002 medium_003 high_002 high_003",
    "marketing.attribution": "low_005 medium_004 high_004",
    "marketing.pricing": "low_006 medium_005 high_005",
    "marketing.positioning": "medium_006 high_006",
    "marketing.channel_allocation": "low_008 medium_007 high_007",
    "marketing.localization": "medium_008 high_008",
}


def family_map():
    result = {}
    for family, names in FAMILIES.items():
        domain = family.split(".")[0]
        for name in names.split():
            pair = name if domain == "reasoning" else f"{domain}_{name}"
            if pair in result:
                raise ValueError(f"Duplicate pair {pair}")
            result[pair] = family
    return result


def assign_families(tasks, seed=42, trials=30000):
    """Choose balance using labels and domains only, without predictions or text."""
    families = family_map()
    groups = collections.defaultdict(list)
    pairs = {}
    for task in tasks:
        pairs.setdefault(task["pair_id"], task)
    if set(pairs) != set(families):
        raise ValueError("Family map must cover every pair exactly once")
    for pair, task in pairs.items():
        groups[families[pair]].append((task["domain"], task["expected"]))
    domains = sorted({t["domain"] for t in tasks})
    totals = collections.Counter((t["domain"], t["expected"]) for t in pairs.values())
    fractions = (0.6, 0.2, 0.2)
    names = sorted(groups)
    rng = random.Random(seed)
    best = None
    for _ in range(trials):
        assignment = dict(zip(names, rng.choices(SPLITS, weights=fractions, k=len(names))))
        counts = collections.Counter((assignment[f], d, level) for f in names for d, level in groups[f])
        if any(counts[s, d, level] == 0 for s in SPLITS for d in domains for level in LEVELS):
            continue
        score = sum((counts[s, d, level] - totals[d, level] * frac) ** 2 / (totals[d, level] * frac)
                    for s, frac in zip(SPLITS, fractions) for d in domains for level in LEVELS)
        candidate = (score, tuple(assignment[n] for n in names))
        if best is None or candidate < best:
            best = candidate
    if best is None:
        raise ValueError("No valid split with every domain and level represented")
    return dict(zip(names, best[1])), best[0]


def prepare(root, out):
    tasks, hashes = load_inputs(root)
    families = family_map()
    assignment, objective = assign_families(tasks)
    rows = {s: [] for s in SPLITS}
    index = []
    for task in tasks:
        family = families[task["pair_id"]]
        split = assignment[family]
        questions = copy.deepcopy(QUESTIONS)
        questions["difficulty"]["label"] = task["expected"]
        meta = {"id": task["task_id"], "pair_id": task["pair_id"], "group_id": family,
                "family_id": family, "source": "routing_calibration", "split": split,
                "lang": task["lang"], "domain": task["domain"], "dataset": task["dataset"]}
        rows[split].append({"state": task["prompt"], "questions": questions, "_meta": meta})
        index.append({**meta, "expected": task["expected"],
                      "prompt_sha256": hashlib.sha256(task["prompt"].encode()).hexdigest()})
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    for split, records in rows.items():
        data = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in records)
        target = out / f"{split}.jsonl"
        if target.exists() and target.read_text(encoding="utf-8") != data:
            raise ValueError(f"Refusing to replace a different frozen split: {target}")
        target.write_text(data, encoding="utf-8")
        files[target.name] = hashlib.sha256(data.encode()).hexdigest()
    manifest = {
        "schema_version": 1, "seed": 42, "search_trials": 30000,
        "target_fractions": dict(zip(SPLITS, (0.6, 0.2, 0.2))), "balance_objective": objective,
        "split_method": "Manual domain-family grouping, label/domain-only random search for balance.",
        "limitations": ["All prompts were previously inspected in the baseline study; this is an exploratory holdout.",
                        "Families are manual domain-specific groups, not proof of disjoint reasoning skills.",
                        "Difficulty labels describe dataset design, not measured routing quality."],
        "source_files_sha256": hashes, "rubric_sha256": hashlib.sha256(canonical(QUESTIONS).encode()).hexdigest(),
        "files_sha256": files, "family_assignment": assignment,
        "counts": {s: {"prompts": len(records), "pairs": len({r['_meta']['pair_id'] for r in records}),
                       "families": len({r['_meta']['family_id'] for r in records}),
                       "by_level": dict(sorted(collections.Counter(r['questions']['difficulty']['label'] for r in records).items())),
                       "by_domain_level": dict(sorted(collections.Counter(r['_meta']['domain'] + '/' + r['questions']['difficulty']['label'] for r in records).items()))}
                   for s, records in rows.items()},
        "records": index,
    }
    (out / "split_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets-root", type=Path, default=Path("datasets/routing_calibration"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.datasets_root, args.out)["counts"], indent=2))
