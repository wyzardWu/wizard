# Wizard — training and inference code

Wizard is a real-time playable world model built on [Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B). A lightweight engine keeps the world state and renders a *proxy video* (character skeletons + scene geometry); the video model in this repository turns the proxy, a first frame, a text prompt, up to two character reference images and a few retrieved memory frames into the picture.

This directory holds the model, the four training stages and offline inference. The code lives on GitHub at [wyzardWu/wizard](https://github.com/wyzardWu/wizard) (`code/`) and is mirrored in the `code/` folder of [`yuzhewu207/wizard`](https://huggingface.co/datasets/yuzhewu207/wizard), which also holds the raw data, the weights (`weights/`) and ready-made inference cases (`examples/`). The pre-encoded training data is in [`yuzhewu207/wizard-precoded`](https://huggingface.co/datasets/yuzhewu207/wizard-precoded).

## Method in one page

| component | what it does | where |
|---|---|---|
| Proxy video | R = height, B = depth, G = geometry tier (sky / unknown / coarse / exact) for the scene and identity + facing for characters; injected by channel concatenation through a zero-initialised patch embedding | `wizard/models/proxy.py` |
| Character-aware mask | each reference image is visible only to the video tokens inside its character's region, so two characters with the same skeleton stay distinct | `wizard/models/reference.py` |
| World-anchored spatial condition | every background cell of the proxy and of the memory frames carries a world-coordinate token; unknown cells, sky and characters carry none | `wizard/models/wizard_dit.py`, `wizard/memory/gbuffer.py` |
| Memory retrieval | per chunk of 4 latent frames, up to three earlier frames chosen greedily by how much of the currently visible scene each newly covers; one function for training and inference | `wizard/memory/retrieve.py` |

Training runs in four stages; from stage 1 on the model reads a sliding history of 20 latent frames plus the first frame as a permanent anchor (sink).

| stage | entry point | script | released run |
|---|---|---|---|
| 0. bidirectional | `wizard.train.bidirectional` | `scripts/train_bidirectional.sh` | 35K steps |
| 1. causal adaptation (teacher forcing, history corruption) | `wizard.train.causal_selfaug` | `scripts/train_causal.sh` | 11K steps |
| 2. consistency distillation | `wizard.distill.consistency_selfaug_future` | `scripts/distill_consistency.sh` | 10K steps |
| 3. distribution matching distillation (DMD) | `wizard.distill.dmd` | `scripts/distill_dmd.sh` | 480 student updates |

The deployed model is the EMA student of stage 3, sampled in 2 steps with a KV cache (`wizard/streaming/kv_stream.py`).

## Layout

```text
wizard/
  paths.py              data root and base-model directory (environment variables)
  models/
    dit.py              Wan2.2-TI2V-5B DiT (bidirectional and causal)
    flow_match.py       flow-matching scheduler
    proxy.py            proxy-video injection
    reference.py        reference images with the character-aware mask
    wizard_dit.py       the Wizard forward: references + memory frames + coordinate tokens
    wan/                Wan2.2 VAE and UMT5 encoder (unchanged upstream files)
  memory/               retrieval and per-cell scene geometry
  data/                 depth completion behind characters, video reading
  train/                bidirectional.py, causal.py, causal_selfaug.py (base.py: classes they extend)
  distill/              consistency*.py, dmd.py, cd_utils.py
  streaming/            KV-cache streaming of the causal model
  inference/            bidirectional.py, causal.py
  utils/                attention diagnostic
scripts/                one launch script per stage, with the released settings
configs/                accelerate (FSDP) configurations
```

## Installation

```bash
conda create -n wizard python=3.12 -y && conda activate wizard
pip install -r requirements.txt          # install the torch build that matches your CUDA first if needed
pip install -e .
```

Download the base model and point the code at it and at the data:

```bash
huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --local-dir checkpoints/Wan2.2-TI2V-5B
export WIZARD_WAN=$PWD/checkpoints/Wan2.2-TI2V-5B
export WIZARD_DATA=/path/to/encoded/data
```

Tested with Python 3.12, PyTorch 2.11 (CUDA 12.8) on 8 × H200. One 5B checkpoint is about 10 GB.

## Inference

A *case* is a directory produced by the engine: `first_frame.png`, `proxy.mkv` (proxy video), `cond.mp4` (the same proxy in the earlier colour coding; only the character layer is read from it), `tier.npy` (geometry tier per cell), `gbuf.npz` (depth and cameras), `slots.npz` (character regions), `prompt.txt`. Ready-made cases with our outputs are in the `examples/` folder of [`yuzhewu207/wizard`](https://huggingface.co/datasets/yuzhewu207/wizard).

```bash
# bidirectional model: 30 steps, guidance 3
huggingface-cli download yuzhewu207/wizard --repo-type dataset --include "weights/bidirectional/*" "weights/causal_dmd/*" "examples/*" --local-dir hf_wizard
C=hf_wizard/examples/fight_character_strikes_player
INF_UNK_NOCOORD=1 python -m wizard.inference.bidirectional \
    --dit hf_wizard/weights/bidirectional/dit_step35000.safetensors --case $C --frames 121 \
    --ref1 $C/ref_a.png --ref2 $C/ref_b.png --ref-slots 2,3 --out out.mp4

# causal 2-step student
INF_UNK_NOCOORD=1 python -m wizard.inference.causal \
    --dit hf_wizard/weights/causal_dmd/ema_step480.safetensors --case $C --frames 121 \
    --ref1 $C/ref_a.png --ref2 $C/ref_b.png --ref-slots 2,3 \
    --cm_steps 1000,500 --cfg 1 --window 20 --sink 1 --out out_causal.mp4
```

`INF_UNK_NOCOORD=1` is required: training gives no coordinate token to unknown cells, and inference does the same only when this is set.

## Training

Every stage is launched with accelerate under FSDP, one window per GPU. The scripts hold the settings of the released models; each takes its inputs from environment variables and passes extra arguments through.

```bash
OUT=runs/bidirectional                                                    bash scripts/train_bidirectional.sh
INIT=runs/bidirectional/dits/dit_step35000.safetensors OUT=runs/causal    bash scripts/train_causal.sh
TEACHER=runs/causal/dits/dit_step11000.safetensors OUT=runs/consistency   bash scripts/distill_consistency.sh
STUDENT=runs/consistency/ema/ema_step10000.safetensors \
REAL=runs/bidirectional/dits/dit_step35000.safetensors OUT=runs/dmd       bash scripts/distill_dmd.sh
```

Use `GPUS=0,1,2,3` to choose cards (the bidirectional script keeps the total batch at 8 with gradient accumulation). Re-running a stage-0 or stage-1 script resumes from the latest checkpoint.

### Switches

Most options that change *what* is trained are environment variables rather than arguments; the scripts set them.

| variable | meaning |
|---|---|
| `UNK_NOCOORD=1` | unknown (dark green) cells get no coordinate token. On in every released run; the inference counterpart is `INF_UNK_NOCOORD=1` |
| `WEDIT=1` | include the world-edit windows |
| `MEMW_COVER`, `MEMW_HISTW`, `MEMW_SINK` | how cells that only memory can explain are found (they get loss weight `--mem_loss_w`) |
| `SA_P`, `SA_MAX`, `SA_GROUPS` | history corruption: probability, maximum noise level, data groups it applies to (stage 1: 0.5 / 0.25; stage 2: 0.7 / 0.5) |
| `MEM_FUTURE=1` | training only: camera-flight windows may also retrieve frames after the window |
| `ABL_NOGATE=1`, `ABL_NOMEM=1`, `ABL_NOCOORD=1` | ablations: no character-aware mask; no memory frames and no coordinate tokens; memory frames without coordinate tokens (training and inference) |
| `CD_FREE_GB` | stage 2 skips intermediate EMA saves when the output disk has less free space than this (default 15) |

`--group_rep` sets how often each data group is repeated per epoch, `--rep_mem` repeats the windows that need memory, `--p_unk2` is the probability of the variant in which everything outside the first frame's view is marked unknown.

### Data format

The training code reads **pre-encoded** windows (Wan2.2-VAE latents of the target and of the proxy video, text embeddings, per-cell geometry, memory banks and retrieval indices) under `$WIZARD_DATA`: `windows/` (row file, proxy latents; `windows/geometry` links to `geometry/`, the per-cell depth and cameras), `samples/` (target latents and text table; its `clips/` link into `latents/` and the other sample folders), `roam_memory/`, `scenefly/`, `scenefly_replay/`, `synth_lookback_geo/`, `worldedit/`.

**The pre-encoded data is released in [`yuzhewu207/wizard-precoded`](https://huggingface.co/datasets/yuzhewu207/wizard-precoded)** (804 GiB, the exact files the models were trained on, minus the WildWorld windows). Restore it with

```bash
python scripts/restore_precoded.py --root /path/to/wizard_data     # download + extract + rewrite the absolute paths
export WIZARD_DATA=/path/to/wizard_data
```

and every script above runs unchanged (group `ww` simply has no rows). The dataset in this repository is the model-independent form of the same windows (target videos, decoded proxy videos, depth, cameras, memory metadata), for re-encoding with another backbone.

## Notes on the code

This is research code, released as it was used, after a mechanical clean-up: files were moved into a package, imports and paths were unified, and development notes were removed. Behaviour was not changed. Some traces of its history remain:

- The DiT class keeps its development name (`GameMasterDiT`).
- Later stages extend earlier modules by importing and patching them (for example `wizard.distill.consistency_selfaug` installs its loss into `wizard.distill.consistency`, and `wizard.train.causal_selfaug` adds gradient accumulation to the loop of `wizard.train.causal`). Launch the entry points listed above rather than the modules they extend.
- With FSDP every rank must make the same number of collective calls per step; this is why random per-step decisions (such as whether to corrupt the history) are shared across ranks.
- The `--maintain` options of the inference scripts need the online world-maintenance modules (MapAnything reconstruction and a point cache), which are not part of this release, as is the engine that produces cases.

## Licence

Non-commercial research use only; see `LICENSE.md`. Third-party files keep their own licences.

## Acknowledgements and third-party code

See `NOTICE.md`. The DiT and the scheduler follow DiffSynth-Studio's Wan implementation; `wizard/models/wan/` contains unchanged files of Wan2.2; the distillation stages follow Self Forcing, Causal Forcing and DMD.
