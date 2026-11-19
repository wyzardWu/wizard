# Third-party code

- `wizard/models/wan/vae2_2.py`, `t5.py`, `tokenizers.py`: unchanged from [Wan2.2](https://github.com/Wan-Video/Wan2.2) (`wan/modules/`), Copyright 2024-2025 The Alibaba Wan Team Authors, Apache License 2.0.
- `wizard/models/dit.py`, `wizard/models/flow_match.py`: a re-implementation whose structure, weight names and scheduler math follow [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) (`diffsynth/models/wan_video_dit.py`, `diffsynth/diffusion/flow_match.py`), Apache License 2.0.
- The base weights are [Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B), Apache License 2.0.

The training recipes build on Self Forcing, Causal Forcing and Distribution Matching Distillation (DMD / DMD2).
