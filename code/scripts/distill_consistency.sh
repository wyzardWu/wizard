#!/bin/bash
# Stage 2: consistency distillation of the causal teacher (10K steps for the released model). EMA students are saved to OUT/ema/.
#   TEACHER=runs/causal/dits/dit_step11000.safetensors OUT=runs/consistency bash scripts/distill_consistency.sh [extra arguments]
# History corruption uses the defaults SA_P=0.7, SA_MAX=0.5.
set -e
source "$(dirname "$0")/common.sh"
: "${OUT:?set OUT to the output directory}"; : "${TEACHER:?set TEACHER to the causal teacher checkpoint}"
mkdir -p "$OUT"; export UNK_NOCOORD=1 WEDIT=1
CUDA_VISIBLE_DEVICES=$GPUS accelerate launch --config_file configs/accelerate_fsdp_dmd.yaml --num_processes "$NPROC" --main_process_port "${PORT:-29853}" \
  -m wizard.distill.consistency_selfaug_future --data "$DS/windows" --ckpt "$TEACHER" --out "$OUT" --max_steps "${MAX:-10000}" --save_every 1000 --cfg 3 --window 20 --sink 1 \
  --group_rep "$GROUP_REP,lookback:4,lookback0:1" $MEM_ARGS \
  --extra_rows "$DS/scenefly/rows.jsonl,$DS/scenefly_replay/rows.jsonl,$DS/synth_lookback_geo/rows_full.jsonl,$DS/worldedit/rows_train.jsonl" \
  --p_trueunk 0 --p_unk2 0.5 "$@" 2>&1 | tee -a "$OUT/train_out.log"
