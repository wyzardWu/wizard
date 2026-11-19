# Wizard: Manipulate Any World in Real Time via Agentic Control

[Code](https://github.com/wyzardWu/wizard27-clean) · [Project page](https://wyzardwu.github.io/wizard) · [Paper (arXiv 2610.40339)](https://arxiv.org/abs/2610.40339) · [Data, weights, examples](https://huggingface.co/datasets/yuzhewu207/wizard) · [Pre-encoded training data](https://huggingface.co/datasets/yuzhewu207/wizard-precoded)

Wizard is a real-time, playable world model. From any first frame it rigs the characters in the picture, lets an engine drive them (fights, movement, spells that reshape the world), renders the engine state as a *proxy video* of skeletons and scene geometry, and generates the real picture chunk by chunk with a 2-step causal video model built on [Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B), at about 20 frames per second.

## What is where

| | where |
|---|---|
| training and inference code | [wyzardWu/wizard27-clean](https://github.com/wyzardWu/wizard27-clean) |
| model weights | [`yuzhewu207/wizard`](https://huggingface.co/datasets/yuzhewu207/wizard) → `weights/` |
| raw training data (videos, proxy videos, geometry, memory) | [`yuzhewu207/wizard`](https://huggingface.co/datasets/yuzhewu207/wizard) |
| pre-encoded training data (the exact training files) | [`yuzhewu207/wizard-precoded`](https://huggingface.co/datasets/yuzhewu207/wizard-precoded) |
| ready-made inference cases | [`yuzhewu207/wizard`](https://huggingface.co/datasets/yuzhewu207/wizard) → `examples/` |
| project page | this repository (`index.html`), served at [wyzardwu.github.io/wizard](https://wyzardwu.github.io/wizard) |

Installation, quick start and training are described in the [code repository](https://github.com/wyzardWu/wizard27-clean).

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
