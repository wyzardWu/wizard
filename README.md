# Wizard: Manipulate Any World in Real Time via Agentic Control

[Project page](https://wyzardwu.github.io/wizard) · [Paper (arXiv 2610.40339)](https://arxiv.org/abs/2610.40339) · [Data, weights, examples](https://huggingface.co/datasets/yuzhewu207/wizard) · [Pre-encoded training data](https://huggingface.co/datasets/yuzhewu207/wizard-precoded)

Wizard is a real-time, playable world model. From any first frame it rigs the characters in the picture, lets an engine drive them (fights, movement, spells that reshape the world), renders the engine state as a *proxy video* of skeletons and scene geometry, and generates the real picture chunk by chunk with a 2-step causal video model built on [Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B), at about 20 frames per second.

## What is where

| | where |
|---|---|
| training and inference code | [`code/`](code/) in this repository (package `wizard`) |
| model weights | [`yuzhewu207/wizard`](https://huggingface.co/datasets/yuzhewu207/wizard) → `weights/` |
| raw training data (videos, proxy videos, geometry, memory) | [`yuzhewu207/wizard`](https://huggingface.co/datasets/yuzhewu207/wizard) |
| pre-encoded training data (the exact training files) | [`yuzhewu207/wizard-precoded`](https://huggingface.co/datasets/yuzhewu207/wizard-precoded) |
| ready-made inference cases | [`yuzhewu207/wizard`](https://huggingface.co/datasets/yuzhewu207/wizard) → `examples/` |
| project page | this repository's root (`index.html`), served at [wyzardwu.github.io/wizard](https://wyzardwu.github.io/wizard) |

## Quick start

```bash
git clone https://github.com/wyzardWu/wizard && cd wizard/code
conda create -n wizard python=3.12 -y && conda activate wizard
pip install -r requirements.txt && pip install -e .

huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --local-dir checkpoints/Wan2.2-TI2V-5B
export WIZARD_WAN=$PWD/checkpoints/Wan2.2-TI2V-5B

# inference on a ready-made case with the 2-step causal model
huggingface-cli download yuzhewu207/wizard --repo-type dataset \
    --include "weights/causal_dmd/*" "examples/*" --local-dir hf_wizard
C=hf_wizard/examples/fight_character_strikes_player
INF_UNK_NOCOORD=1 python -m wizard.inference.causal \
    --dit hf_wizard/weights/causal_dmd/ema_step480.safetensors --case $C --frames 121 \
    --ref1 $C/ref_a.png --ref2 $C/ref_b.png --ref-slots 2,3 \
    --cm_steps 1000,500 --cfg 1 --window 20 --sink 1 --out out_causal.mp4
```

## Training

```bash
python scripts/restore_precoded.py --root /path/to/wizard_data     # pre-encoded data, ~0.85 TB
export WIZARD_DATA=/path/to/wizard_data
bash scripts/train_bidirectional.sh      # stage 0: bidirectional model
bash scripts/train_causal.sh             # stage 1: causal adaptation
bash scripts/distill_consistency.sh      # stage 2: consistency distillation (2 steps)
bash scripts/distill_dmd.sh              # stage 3: DMD
```

Each stage runs on 8 × H200 with FSDP; the scripts carry the released settings and the weights each stage starts from are listed in [`code/README.md`](code/README.md). The WildWorld part of the training data (about half of the windows) cannot be redistributed; [`wizard-precoded/ww_index`](https://huggingface.co/datasets/yuzhewu207/wizard-precoded/tree/main/ww_index) explains how to rebuild it from the upstream dataset.

## Citation

```bibtex
@article{wu2026wizard,
  title   = {Wizard: Manipulate Any World in Real Time via Agentic Control},
  author  = {Wu, Yuzhe and Lin, Zijun and Jin, Yeying},
  journal = {arXiv preprint arXiv:2610.40339},
  year    = {2026}
}
```

## Licence

Research use only; see [`LICENSE.md`](LICENSE.md).
