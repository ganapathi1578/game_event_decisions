# Badminton Video Decision Model — SmolVLM2 + BEV/Jev-style Choice Ordering

A runnable research project for training a configurable video multiple-choice decision model on
FineBadminton-20K. Input format: **short video clip + question + user-supplied options**. Output: one
probability per supplied option. The model is trained with a choice pointer head rather than answer-token
generation.

## What is implemented

- FineBadminton-20K snapshot downloader (Hugging Face Hub).
- Flexible parser for the released per-rally JSON annotation structure (`hitting`, `Foundational Actions Level`,
  `Tactical Semantics Level`).
- Match-level train/validation/test split to prevent clips from one match crossing splits.
- Three-second video windows sampled into eight RGB frames.
- QA generation for hit type, single-label subtype, hitter side, and ball area.
- Separate training / validation / test question-template banks.
- Random option count and order during training; correct indices are recomputed after shuffling.
- SmolVLM2 visual encoder + connector to obtain visual tokens; features replace the matching native
  `<image>` placeholder embeddings in the prompt, preserving the model's fake/global image markers.
- The reference decision layout: `<option>` markers, shared option position starts, independent option attention blocks,
  final answer prompt, reference-style choice head, bilinear option scores, and cross-entropy over options.
- LoRA rank 8 / alpha 16 on the last 12 text decoder layers by default.
- Training, checkpoint save/load, validation, test evaluation, permutation consistency evaluation, and interactive inference.
- Unit tests for the mask, positional reset, choice-head permutation equivariance, data parsing, and split integrity.

## Important scope note

The public reference implementation (`https://github.com/avbiswas/bev-train`) uses Qwen3-0.6B for text state
inputs. This project preserves its option tokenization, position-ID, block attention, choice-reading, choice-head,
and classification-loss mechanics, but adapts the backbone to `HuggingFaceTB/SmolVLM2-256M-Video-Instruct`.
The visual frames are encoded using SmolVLM2's pretrained vision tower and connector. The prompt is built with
SmolVLM-native frame/image markers, and connector features replace the corresponding `<image>` placeholder
embeddings in-place before the custom text decoder mask and shared option position IDs are applied. This preserves
the native multimodal prompt structure; it is not a byte-for-byte fork of the Qwen code.

The custom 4D attention mask is run with eager attention. Do not switch to FlashAttention 2 for this first run.

## Requirements

- Linux recommended for training.
- Python 3.10–3.12 recommended.
- NVIDIA GPU. A 10–11 GB RTX 2080 Ti should start with per-device batch 1, eight frames, FP16, and gradient
  checkpointing. If it runs out of memory, reduce frames to 4 before reducing image size.
- Approximately 50 GB free for the 43.4 GB dataset snapshot plus model/cache/checkpoints; 70–100 GB is safer.
- The first run downloads SmolVLM2 weights (several GB) from Hugging Face.

## 1. Environment setup

Install a PyTorch build matching your NVIDIA driver/CUDA environment first. Example for CUDA 12.4 wheels:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[dev]"
```

For Windows PowerShell, activate with:

```powershell
.venv\Scripts\Activate.ps1
```

Then run:

```bash
python scripts/check_environment.py
pytest -q
```

If your CUDA driver requires a different PyTorch wheel, use the official PyTorch install selector rather than blindly
using the CUDA 12.4 example. Do not install a CPU-only PyTorch build on the GPU training server.

## 2. Download the dataset

The dataset repository is `iLearn-Lab/Finebadminton-20K` (about 43.4 GB). The command downloads only the video,
annotation, dataset-card and benchmark-list paths used here.

```bash
python scripts/download_data.py --output-dir data/raw/finebadminton-20k
```

If the repository requires authentication in your environment:

```bash
export HF_TOKEN=hf_...
python scripts/download_data.py --output-dir data/raw/finebadminton-20k
```

After installing the environment, the repository also includes a sequential helper. It checks the environment,
runs tests, downloads the dataset, inspects annotations, and prepares the manifests. Add `--train` to start
the full configured training run after preparation:

```bash
bash scripts/quickstart.sh
# or, including training:
bash scripts/quickstart.sh --train
```

Expected structure:

```text
data/raw/finebadminton-20k/
├── videos/
├── finebadminton-20K/
├── benchmark_source_videos.txt
└── README.md
```

## 3. Inspect the raw annotations

```bash
python scripts/inspect_data.py --data-dir data/raw/finebadminton-20k
```

This reports annotation files, video files, parsed event count, matches and label counts. The Hugging Face dataset
viewer currently reports a schema-casting failure on nested annotations; this project reads the raw JSON files
instead of relying on the viewer/Arrow cast.

If `parsed_events` is zero, inspect the JSON layout and generated report before starting training. The parser is
schema-tolerant, but it intentionally does not invent missing timestamps or labels.

## 4. Build splits and QA manifests

```bash
python scripts/prepare_data.py \
  --data-dir data/raw/finebadminton-20k \
  --output-dir data/processed \
  --config configs/default.yaml \
  --templates configs/question_templates.yaml \
  --seed 42
```

The command creates:

```text
data/processed/
├── events.jsonl
├── train.jsonl
├── validation.jsonl
├── test.jsonl
└── prepare_summary.json
```

Splits are assigned by full-match ID (80/10/10 by default). The QA builder creates tasks only when the annotation
contains an unambiguous single answer. `subtype` rows with multiple subtype tags are skipped in this first
single-choice setup; this is reported in `prepare_summary.json`.

The `train`, `validation`, and `test` question templates are disjoint. Training samples choose a template and a
random number of options at load time; the correct option is always included and options are shuffled. Validation
and test use separate templates not present in training.

## 5. Run unit tests and decision-layout smoke tests

```bash
pytest -q
```

The optional development dependencies (including pytest) are installed by `pip install -e ".[dev]"` above.
The tests verify:

- every option has the same logical starting position ID;
- choice representations are read at the final `</option>` marker token, matching the reference tokenizer/network;
- option blocks cannot attend to other option blocks;
- options can attend to context and their own causal block;
- the answer block can attend to all previous options;
- the choice head is permutation-equivariant when option embeddings are permuted.

## 6. Train

Single GPU:

```bash
accelerate launch --num_processes 1 scripts/train.py \
  --config configs/default.yaml \
  --name badminton-smolvlm2-r8-k12
```

Two GPUs (e.g. two RTX 2080 Ti devices):

```bash
accelerate launch --multi_gpu --num_processes 2 scripts/train.py \
  --config configs/default.yaml \
  --name badminton-smolvlm2-r8-k12
```

The default settings are deliberately conservative:

- SmolVLM2-256M-Video-Instruct.
- Vision encoder and connector frozen.
- LoRA rank 8, alpha 16, dropout 0.
- LoRA on q/k/v/o projections in the last 12 of 30 text decoder layers.
- Choice head hidden dimension 512, two attention blocks.
- Eight frames per three-second clip.
- Per-device batch 1, gradient accumulation 32.
- FP16 and gradient checkpointing.
- Three epochs, LoRA LR 1e-4, head LR 1e-3.

With one GPU, the effective batch is 32. With two GPUs under Accelerate/DDP, it is 64. If the 2080 Ti runs out of
memory, edit `configs/default.yaml` and set `video.num_frames: 4`. Do not start by changing the choice mask or
sequence layout.

Training writes the config, metrics and checkpoints to `runs/<name>/`. The `best`, `latest`, and `final`
checkpoints contain only LoRA adapter weights, choice-head weights and metadata; the base SmolVLM2 weights are
reloaded from Hugging Face during inference.

## 7. Evaluate

Validation:

```bash
python scripts/evaluate.py runs/badminton-smolvlm2-r8-k12/checkpoints/best \
  --processed-dir data/processed \
  --video-root data/raw/finebadminton-20k/videos \
  --split validation \
  --permutation-trials 5 \
  --output runs/eval_validation.json
```

Held-out test:

```bash
python scripts/evaluate.py runs/badminton-smolvlm2-r8-k12/checkpoints/best \
  --processed-dir data/processed \
  --video-root data/raw/finebadminton-20k/videos \
  --split test \
  --permutation-trials 5 \
  --output runs/eval_test.json
```

Metrics include accuracy, macro-F1, per-class precision/recall/F1, ECE, per-task results, prediction consistency
under option permutations, and the mean absolute change in probability assigned to the same semantic option.

## 8. Interactive inference on a new clip

```bash
python scripts/predict.py runs/badminton-smolvlm2-r8-k12/checkpoints/best \
  --video /path/to/match.mp4 \
  --start 42.0 --end 45.0 \
  --task-type subtype \
  --question "Which type of serve is performed?" \
  --options "High serve" "Low serve" "Flick serve" "Drive serve"
```

The same checkpoint accepts a different question and a different list/count of options at inference. Accuracy on
truly unseen concepts is not guaranteed: option-order invariance provides structural handling of candidate order,
not zero-shot visual knowledge of labels never represented during training.

## 9. Smoke-training subset

A small smoke configuration is included, so the full configuration does not need to be edited:

```bash
accelerate launch --num_processes 1 scripts/train.py \
  --config configs/smoke.yaml \
  --name smoke-smolvlm2
```

It uses up to 8 train examples, 4 validation examples, 4 frames and one epoch. It still downloads the base model
and runs the real SmolVLM2 vision/text forward path. Use it after dataset preparation and before a full run. It is
only a code-path check, not a meaningful accuracy experiment.

## 10. Data rights and research limitations

The dataset card declares Apache-2.0 for the dataset release, while its paper describes video sources from public
BWF broadcasts. Do not assume the dataset card alone grants all rights to redistribute or commercially deploy the
underlying broadcast footage. Check the dataset terms, source-video copyrights and any applicable platform terms.
For the investor-facing product, obtain rights-cleared match footage or customer-provided data.

The initial model has four single-answer tasks: hit type, one unambiguous subtype, hitter side, and ball area. It
does not initially train shot-quality scoring, multi-label tactical semantics, rally-level reasoning, or free-form
caption generation.

## Reference

- Method reference: https://github.com/avbiswas/bev-train
- Video walkthrough: https://www.youtube.com/watch?v=sF3CNPbWA8o
- Dataset: https://huggingface.co/datasets/iLearn-Lab/Finebadminton-20K
- Base model: https://huggingface.co/HuggingFaceTB/SmolVLM2-256M-Video-Instruct

See `docs/REFERENCE_METHOD.md` for the exact mapping between the reference method and this video adaptation.

## Troubleshooting: `ModuleNotFoundError: badminton_decision.models`

This repository uses a `src/` package layout. From the repository root, install
it in editable mode before running the training commands:

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

The test suite also includes `tests/conftest.py`, which adds the local `src/`
directory to the test import path. If an older installation shadows the local
checkout, verify the resolved package with:

```bash
python -c "import badminton_decision; print(badminton_decision.__file__)"
```

The expected path ends with `src/badminton_decision/__init__.py`.
