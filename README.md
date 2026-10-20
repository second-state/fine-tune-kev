# Fine-tune Kev for prompt difficulty

We want to route each LLM API call to a model based on the prompt content.
Kev-4B and Kev-0.8B classify the prompt as `low`, `medium`, or `high` difficulty.
A routing rule can then select a budget, general-purpose, or premium model.
The goal is to reduce API cost while keeping the required answer quality.

This repository records two experiments from `smart-route-bench`.
The Kev-4B experiment ran on October 1, 2026. The Kev-0.8B experiment ran on October 16, 2026.
It includes the dataset, fixed data split, training tools, and saved results.
Both experiments measure agreement with difficulty labels.
They do not measure production routing savings or the quality of model answers.

## Environment

We ran both experiments locally on a MacBook Pro with an Apple M2 Max.

| Component | Specification |
|---|---|
| CPU | 12 cores: 8 performance and 4 efficiency |
| GPU | 30 cores |
| Unified memory | 64 GiB |
| macOS | 26.6.2 |
| MLX | 0.32.2 |
| MLX-LM | 0.31.3 |

Training used PyTorch 2.8.0 with the MPS backend.
Evaluation used MLX with bfloat16 precision.
The [4B runtime record](reports/2026-10-01-kev-4b-finetune/runtime.json) and [0.8B runtime record](reports/2026-10-16-kev-0.8b-finetune/runtime.json) list identical Python and package versions.

## Dataset

We used the project-authored [routing_calibration dataset](datasets/routing_calibration/README.md).
It contains 192 synthetic prompts in 18 directories.
The prompts form 96 English and Traditional Chinese translation pairs.
Each difficulty level contains 64 prompts.

| Domain | English prompts | Chinese prompts | Total | Examples |
|---|---:|---:|---:|---|
| General reasoning | 48 | 48 | 96 | Arithmetic, probability, scheduling, graph problems |
| Marketing | 24 | 24 | 48 | Funnels, pricing, attribution, budget allocation |
| Software architecture | 24 | 24 | 48 | Caches, migrations, capacity, consistency |
| Total | 96 | 96 | 192 | |

The raw files retain the original prompts, numeric answers, and difficulty labels.
Every prompt asks for a numeric answer, but Kev classifies the required reasoning instead of solving the task.
The difficulty labels express the intended task difficulty.
They do not identify which deployed model can solve each prompt.
Each dataset manifest declares the CC0-1.0 license.

The class rules use the same English text for both languages:

| Class | Meaning |
|---|---|
| `low` | Direct facts, simple arithmetic, or one explicit rule |
| `medium` | Several connected steps with manageable constraints |
| `high` | Interacting constraints, substantial search, causal reasoning, or complex system behavior |

The prepared records in [datasets/kev_difficulty](datasets/kev_difficulty/) preserve the full rules and original prompts.
Each record uses the following fields:

| Field | Purpose |
|---|---|
| `state` | Original prompt text |
| `questions.difficulty` | A native `choice` question with instructions and three class descriptions |
| `questions.difficulty.label` | Training target from the original difficulty label |
| `_meta` | Task ID, language, domain, family, and split for audit purposes |

The evaluator removes labels and metadata before model inference.
The numeric answer never enters the prepared model input.

## Original Kev-4B results

The September 30 baseline used the original `jaredpalmer/kev-4b` checkpoint without dataset-specific training.
It ran locally through MLX with bfloat16 precision on an Apple M2 Max with 64 GiB of memory.

| Metric on all 192 prompts | Result |
|---|---:|
| Accuracy | 136 / 192 = 70.83% |
| Macro F1 | 0.6760 |
| Low recall | 60 / 64 = 93.75% |
| Medium recall | 59 / 64 = 92.19% |
| High recall | 17 / 64 = 26.56% |
| High prompts classified as medium | 47 / 64 |
| Mean HTTP request time | 729.94 ms |
| P95 HTTP request time | 990.53 ms |

Accuracy gives each prompt equal weight.
Macro F1 averages the F1 scores for the three classes.
Recall measures how many prompts of a class receive the correct label.
The main baseline error was the assignment of medium difficulty to high-difficulty prompts.

The [baseline summary](reports/2026-09-30-kev-4b-difficulty/summary.json) and [192 predictions](reports/2026-09-30-kev-4b-difficulty/predictions.jsonl) preserve these results.

## Fine-tuning method

Both Kev models combine a Qwen3.5 base model, a LoRA adapter, and a pointer head that scores the class options.

| Kev checkpoint | Base model |
|---|---|
| `jaredpalmer/kev-4b` | `Qwen/Qwen3.5-4B-Base` |
| `jaredpalmer/kev-0.8b` | `Qwen/Qwen3.5-0.8B-Base` |

The trainer kept the base weights fixed.
It continued training the existing adapter and pointer head from the original Kev checkpoint.
A checkpoint stores the resulting adapter, head, and configuration.

We froze each protocol before training and candidate evaluation.
The [4B protocol](reports/2026-10-01-kev-4b-finetune/protocol.json) and [0.8B protocol](reports/2026-10-16-kev-0.8b-finetune/protocol.json) use the same training configuration and selection rules.
We grouped related scenarios into 28 families.
Each family and each translation pair stayed in one split.
The split algorithm used domain and difficulty labels to balance the groups.

| Split | Prompts | Translation pairs | Families | Low / medium / high |
|---|---:|---:|---:|---|
| Train | 122 | 61 | 17 | 42 / 40 / 40 |
| Development | 36 | 18 | 6 | 12 / 12 / 12 |
| Test | 34 | 17 | 5 | 10 / 12 / 12 |

The [split manifest](datasets/kev_difficulty/split_manifest.json) fixes every task ID, family, split, and prompt hash.
Both experiments use identical train, development, and test files.
Training used only the 122 training prompts.
Training and model selection used no test labels.

| Training configuration | Value |
|---|---|
| Candidate runs | 1 epoch and 2 epochs, each from the same original checkpoint |
| LoRA rank and target modules | 16, all parent target modules |
| Pointer head dimension | 256 |
| Learning rate | `1e-5` |
| Batch size | 1 |
| Gradient accumulation | 8 |
| Seed | 42 |
| Training backend | PyTorch MPS, fp32 |
| Gradient checkpointing | Enabled |
| Loss | Cross-entropy |
| Option permutation | Enabled |
| None-option and distractor augmentation | Disabled |

Development macro F1 selected one of the two candidates.
Accuracy and then fewer epochs resolved ties.
The rule selected the 1-epoch checkpoint for both model sizes.
The test evaluation then compared that checkpoint with the original model.

## Fine-tuned results

All methods in this table use the same 34 test prompts:

| Method | Correct | Accuracy | Macro F1 | High recall |
|---|---:|---:|---:|---:|
| Original Kev-4B | 22 / 34 | 64.71% | 0.6066 | 2 / 12 = 16.67% |
| Fine-tuned Kev-4B, 1 epoch | 29 / 34 | 85.29% | 0.8538 | 10 / 12 = 83.33% |
| Original Kev-0.8B | 16 / 34 | 47.06% | 0.3897 | 0 / 12 = 0.00% |
| Fine-tuned Kev-0.8B, 1 epoch | 23 / 34 | 67.65% | 0.6684 | 8 / 12 = 66.67% |

The fine-tuned Kev-4B checkpoint gained seven correct predictions, or 20.59 percentage points, over the original Kev-4B model.
It corrected eight high-to-medium errors and introduced one medium-to-high error.
Low recall stayed at 100%.
Medium recall decreased from 83.33% to 75.00%.

The fine-tuned Kev-0.8B checkpoint also gained seven correct predictions, or 20.59 percentage points, over its original model.
High recall increased from 0 / 12 to 8 / 12.
Medium recall decreased from 6 / 12 to 5 / 12. Low recall stayed at 100%.
The 0.8B fine-tune scored six fewer correct predictions than the 4B fine-tune.

The 0.8B experiment used the fixed 4B recipe. It does not establish the best achievable 0.8B performance.

The 70.83% Kev-4B baseline uses all 192 prompts.
All four test results use only the 34 test prompts.
The full dataset contains training and development prompts, so it cannot provide an independent score for either fine-tuned checkpoint.

The development results also matter:

| Method | Development accuracy | Development macro F1 |
|---|---:|---:|
| Original Kev-4B | 28 / 36 = 77.78% | 0.7813 |
| Fine-tuned Kev-4B, 1 epoch | 28 / 36 = 77.78% | 0.7746 |
| Fine-tuned Kev-4B, 2 epochs | 24 / 36 = 66.67% | 0.6500 |
| Original Kev-0.8B | 21 / 36 = 58.33% | 0.4694 |
| Fine-tuned Kev-0.8B, 1 epoch | 29 / 36 = 80.56% | 0.8038 |
| Fine-tuned Kev-0.8B, 2 epochs | 27 / 36 = 75.00% | 0.7429 |

The selected 4B checkpoint did not beat the original 4B model on development macro F1.
The selected 0.8B checkpoint beat its original model on development macro F1.
We did not evaluate either rejected 2-epoch checkpoint on the test set.

All 192 prompts appeared in the earlier baseline study.
The test set is an exploratory holdout, even though training and selection kept its records separate.
New scenario families and real requests remain necessary to measure performance on new data.

All checkpoints used MLX bfloat16 and temperature 1 for the test comparison.
The earlier Kev-4B full-dataset HTTP baseline used the shipped temperature of about 2.40605.
Fresh original Kev-4B inference produced the same class on all 192 prompts, as recorded in [baseline_reproduction.json](reports/2026-10-01-kev-4b-finetune/baseline_reproduction.json).

| Model | 1-epoch training loop | Full training process | Original predictor mean | Fine-tuned predictor mean |
|---|---:|---:|---:|---:|
| Kev-4B | 688.94 s | 705.89 s | 778.60 ms | 727.60 ms |
| Kev-0.8B | 206.59 s | 211.92 s | 93.89 ms | 100.37 ms |

The full training process includes startup, model loading, training, and checkpoint saving.
The 4B process used a peak RSS of about 24.11 GiB.
Predictor times cover the same 34 test prompts and exclude HTTP, model loading, and warmup.
The model sizes ran on different dates. These observations do not establish a controlled speed comparison.

| Evidence | Kev-4B | Kev-0.8B |
|---|---|---|
| Metrics and timings | [Summary](reports/2026-10-01-kev-4b-finetune/summary.json) | [Summary](reports/2026-10-16-kev-0.8b-finetune/summary.json) |
| Checkpoint selection | [Selection](reports/2026-10-01-kev-4b-finetune/selection.json) | [Selection](reports/2026-10-16-kev-0.8b-finetune/selection.json) |
| Test predictions | [Predictions](reports/2026-10-01-kev-4b-finetune/test_predictions.jsonl) | [Predictions](reports/2026-10-16-kev-0.8b-finetune/test_predictions.jsonl) |

The [comparison record](reports/2026-10-16-kev-0.8b-finetune/comparison_kev4b.json) contains paired comparisons between the model sizes.

## Run the experiment

These commands reproduce the Apple Silicon workflow with the original model revisions.
The commands default to Kev-4B. For Kev-0.8B, use the alternative variables in step 1 before continuing to step 2.
Both sizes use the same environment, data, training configuration, and evaluation commands.

The recorded machine used macOS, an M2 Max, and 64 GiB of memory.
The model downloads require about 8.85 GiB for 4B or 1.71 GiB for 0.8B.
The Python environment and trained checkpoints require additional disk space.
New training runs can produce different weights and predictions.

The [October 13, 2026 reproduction check](reports/2026-10-13-readme-reproduction/verification.json) completed all five 4B steps on this hardware.
All 298 prediction distributions matched the October 1 run exactly.
Training and evaluation took about 39 minutes, excluding setup and downloads.

| Component | Pinned revision |
|---|---|
| Kev source | `0fe8fc97c2bcc247fa3efb6e5c32af4e99770e91` |
| `jaredpalmer/kev-4b` | `139fdd94f1b6a6ad80cc15e08fcb99cac885a101` |
| `Qwen/Qwen3.5-4B-Base` | `1001bb4d826a52d1f399e183466143f4da7b741b` |
| `jaredpalmer/kev-0.8b` | `9a45d25eb2ab761841196625383fa1dff0e56c1e` |
| `Qwen/Qwen3.5-0.8B-Base` | `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68` |

Run each step from this repository root in the same shell.
Use a new work directory for each experiment.
Run only one training or inference process at a time.
After the current step succeeds, continue to the next step.

1. Install `uv` and Python 3.13.5. Create the pinned Kev environment:

   If the pinned Kev checkout and environment already exist, skip the clone and environment setup commands.

   ```bash
   KEV_PROJECT="$PWD"
   KEV_SOURCE="$KEV_PROJECT/runs/kev-source"
   KEV_WORK="$KEV_PROJECT/runs/reproduction"
   KEV_PROTOCOL="$KEV_PROJECT/reports/2026-10-01-kev-4b-finetune/protocol.json"
   mkdir -p "$KEV_PROJECT/runs"
   git clone https://github.com/jaredpalmer/kev.git "$KEV_SOURCE"
   git -C "$KEV_SOURCE" checkout --detach 0fe8fc97c2bcc247fa3efb6e5c32af4e99770e91
   (
     cd "$KEV_SOURCE"
     uv sync --frozen --no-dev --extra serve --python 3.13.5
   )
   ```

   For Kev-0.8B, replace the work directory and protocol before step 2:

   ```bash
   KEV_WORK="$KEV_PROJECT/runs/reproduction-0.8b"
   KEV_PROTOCOL="$KEV_PROJECT/reports/2026-10-16-kev-0.8b-finetune/protocol.json"
   ```

   The protocol selects the base model, original checkpoint, and their pinned revisions.
   The remaining commands use these variables for either model size.

2. Copy the fixed split. Download the pinned base and original Kev checkpoint:

   ```bash
   mkdir -p "$KEV_WORK"
   cp -R datasets/kev_difficulty "$KEV_WORK/data"
   unset HF_HUB_OFFLINE
   "$KEV_SOURCE/.venv/bin/python" tools/kev/download_kev_finetune.py \
     --work "$KEV_WORK" --protocol "$KEV_PROTOCOL"
   ```

3. Run the two-record training probe. Train both candidates independently:

   ```bash
   python3 tools/kev/run_kev_finetune.py \
     --kev-source "$KEV_SOURCE" --work "$KEV_WORK" \
     --protocol "$KEV_PROTOCOL" --preflight
   python3 tools/kev/run_kev_finetune.py \
     --kev-source "$KEV_SOURCE" --work "$KEV_WORK" \
     --protocol "$KEV_PROTOCOL" --epochs 1
   python3 tools/kev/run_kev_finetune.py \
     --kev-source "$KEV_SOURCE" --work "$KEV_WORK" \
     --protocol "$KEV_PROTOCOL" --epochs 2
   ```

   Each command prints its return code and elapsed time after training ends.
   A successful command reports `"returncode": 0`.
   Training output goes to `$KEV_WORK/preflight.log`, `$KEV_WORK/lora-e1.log`, and `$KEV_WORK/lora-e2.log`.
   The logs can remain quiet for several minutes between updates.
   After the 1-epoch command starts, open a second terminal in the repository root.
   View its log:

   ```bash
   tail -f runs/reproduction/lora-e1.log
   ```

   Use `preflight.log` or `lora-e2.log` for the other commands.
   The 0.8B log path is `runs/reproduction-0.8b/lora-e1.log`.
   If you changed `KEV_WORK`, use that directory in the log path.
   Press Ctrl-C to stop the log viewer.

4. Collect the training and development predictions. Freeze the selected checkpoint:

   Evaluation reads the checkpoint paths, so it requires no `--protocol` argument.
   The report tools require the original model predictions on the training and development sets.

   ```bash
   export HF_HUB_CACHE="$KEV_WORK/hf-cache"
   export HF_HUB_OFFLINE=1
   KEV_PARENT=$(python3 -c \
     'import json,sys; print(json.load(open(sys.argv[1]))["models"]["parent"]["path"])' \
     "$KEV_WORK/download.json")

   kev_predict() {
     "$KEV_SOURCE/.venv/bin/python" tools/kev/kev_finetune_eval.py \
       --kev-source "$KEV_SOURCE" --run "$1" \
       --data "$KEV_WORK/data/$2.jsonl" \
       --warmup "$KEV_WORK/data/train.jsonl" \
       --out "$KEV_WORK/predictions/$3-$2.jsonl"
   }

   kev_predict "$KEV_PARENT" train parent
   kev_predict "$KEV_PARENT" dev parent
   kev_predict "$KEV_WORK/lora-e1" dev lora-e1
   kev_predict "$KEV_WORK/lora-e2" dev lora-e2
   python3 tools/kev/report_kev_finetune.py select --work "$KEV_WORK"
   ```

5. Evaluate the original and selected checkpoints on the test set. Generate the report from saved predictions:

   ```bash
   KEV_SELECTED=$(python3 -c \
     'import json,sys; print(json.load(open(sys.argv[1]))["selected_checkpoint"])' \
     "$KEV_WORK/selection.json")
   kev_predict "$KEV_PARENT" test parent
   kev_predict "$KEV_WORK/$KEV_SELECTED" test "$KEV_SELECTED"
   python3 tools/kev/report_kev_finetune.py report \
     --work "$KEV_WORK" --out "$KEV_WORK/report"
   ```

The report command computes test metrics from saved predictions without new model calls.
The training probe uses two training records and does not enter model selection.
The tools refuse to overwrite existing checkpoints, prediction files, or a frozen selection.
Generated weights and run files stay under the ignored `runs/` directory.
The selected checkpoint name is in `$KEV_WORK/selection.json`.
The final metrics are in `$KEV_WORK/report/summary.json`.
The `test` object contains metrics under `parent` and the selected checkpoint name.
The combined test predictions are in `$KEV_WORK/report/test_predictions.jsonl`.

## Check the data and tools offline

These checks require Python 3.11 or later and use no model weights or inference calls.

Run the tests:

```bash
python3 -B -m unittest discover -s tools/kev -p 'test_*.py'
```

Rebuild the train, development, and test files from the raw dataset:

```bash
python3 -B tools/kev/prepare_kev_finetune.py --out runs/data-check
```

The frozen manifest records the SHA-256 of each generated file.
[provenance.json](provenance.json) records the source commits and hashes of all copied data, evidence, and tools.
Historical JSON files retain paths from the original experiment.
The commands in this README use paths inside this repository.

## Source reports

The source reports explain the original baseline, research plan, and completed experiments:

- [Original Kev-4B benchmark](https://github.com/second-state/smart-route-bench/blob/64ccb3ad6d112f87ae17a3d37d91cfa5716d3efc/reports/2026-09-30-kev-4b-difficulty/README.md)
- [Fine-tuning feasibility study](https://github.com/second-state/smart-route-bench/blob/64ccb3ad6d112f87ae17a3d37d91cfa5716d3efc/reports/2026-10-01-kev-4b-finetune-research/README.md)
- [Completed Kev-4B fine-tuning experiment](https://github.com/second-state/smart-route-bench/blob/64ccb3ad6d112f87ae17a3d37d91cfa5716d3efc/reports/2026-10-01-kev-4b-finetune/README.md)
- [Completed Kev-0.8B fine-tuning experiment](https://github.com/second-state/smart-route-bench/blob/6875e98690cb246082df54660f28c474a68ad6aa/reports/2026-10-16-kev-0.8b-finetune/README.md).
