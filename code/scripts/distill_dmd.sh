#!/bin/bash
# Stage 3: distribution matching distillation of the 2-step causal student (480 student updates for the released model).
#   STUDENT=runs/consistency/ema/ema_step10000.safetensors REAL=runs/bidirectional/dits/dit_step35000.safetensors OUT=runs/dmd bash scripts/distill_dmd.sh
# 8 GPUs x --accum 4 = batch 32; critic learning rate 4e-7, 5 critic updates per student update. To resume, pass
#   --student OUT/student_last.safetensors --fake OUT/fake_last.safetensors --step0 <step>.
set -e
source "$(dirname "$0")/common.sh"
: "${OUT:?set OUT to the output directory}"; : "${STUDENT:?set STUDENT to the consistency-distilled EMA}"; : "${REAL:?set REAL to the bidirectional checkpoint}"
mkdir -p "$OUT"; export UNK_NOCOORD=1 MEMW_COVER=1
CUDA_VISIBLE_DEVICES=$GPUS accelerate launch --config_file configs/accelerate_fsdp_dmd.yaml --num_processes "$NPROC" --main_process_port "${PORT:-29861}" \
  -m wizard.distill.dmd --data "$DS/windows" --student "$STUDENT" --real "$REAL" --out "$OUT" \
  --accum "${ACCUM:-4}" --max_steps "${MAX:-480}" --save_every 30 --cfg 3 --levels 1000,500 --num_workers 4 \
  --group_rep e1o:8,e2o:8,e1n:8,e2n:8,e1x:8,turn:16,roam:4,sfly:8,sfly_replay:12,lookback:8,lookback0:2,ww:2,er:2,ue:2,bk:2 --rep_mem 16 --rep_keys "$DS/roam_memory/rep_keys_both30.json" --mem_index mem_index.pt \
  --extra_rows "$DS/scenefly/rows.jsonl,$DS/scenefly_replay/rows.jsonl,$DS/synth_lookback_geo/rows_full.jsonl" \
  --p_trueunk 0 --p_unk2 0.5 "$@" 2>&1 | tee -a "$OUT/train_out.log"
