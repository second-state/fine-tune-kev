# Routing calibration dataset

This directory contains the original data files from `smart-route-bench`.
The 18 dataset directories contain 192 prompts and 96 translation pairs.
Each directory contains a `manifest.toml` file and a `data.jsonl` file.

The dataset covers general reasoning, marketing, and software architecture.
Each domain contains English and Traditional Chinese prompts at three difficulty levels.
General reasoning contains 96 prompts.
Each other domain contains 48 prompts.
The levels `low`, `medium`, and `high` each contain 64 prompts.

Each data row contains a prompt and a numeric answer.
The payload contains the difficulty label and translation pair ID.
Marketing and architecture rows also contain a domain and task shape.
The manifests declare the CC0-1.0 license.

The difficulty experiment classifies the prompt instead of solving the numeric problem.
The prepared [Kev records](../kev_difficulty/) keep the original prompt and add the shared class rules.
The evaluator removes the difficulty label and metadata before inference.
The numeric answer stays outside the model input.

Difficulty labels express the dataset design.
They do not prove that a specific model can solve a prompt.
The marketing and architecture tasks use fixed constraints and numeric results.
They do not measure open-ended writing or system design quality.

The [project README](../../README.md) describes the fixed split, training method, and results.
The [provenance file](../../provenance.json) records the original file paths and hashes.
