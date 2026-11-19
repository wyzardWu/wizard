#!/bin/bash
# Stage 0: bidirectional training from the official Wan2.2-TI2V-5B weights (35K steps for the released model).
#   OUT=runs/bidirectional GPUS=0,1,2,3,4,5,6,7 bash scripts/train_bidirectional.sh [extra arguments]
# Total batch 8 (8 GPUs x 1, or 4 GPUs x gradient accumulation 2). Re-running resumes from the latest checkpoint in OUT.
set -e
source "$(dirname "$0")/common.sh"
: "${OUT:?set OUT to the output directory}"
MAX=${MAX:-35000}; ACC=${ACC:-$(( 8 / NPROC > 0 ? 8 / NPROC : 1 ))}; mkdir -p "$OUT"
st=$(latest "$OUT/dits"); init=""; [ -n "$st" ] && init="--init $OUT/dits/dit_step$st.safetensors --step0 $st"
export MEMW_COVER=1 UNK_NOCOORD=1 WEDIT=1
CUDA_VISIBLE_DEVICES=$GPUS accelerate launch --config_file configs/accelerate_fsdp.yaml --num_processes "$NPROC" --main_process_port "${PORT:-29951}" \
  -m wizard.train.bidirectional --data "$DS/windows" --out "$OUT" $init --max_steps "$MAX" --save_every 1000 --num_workers 6 --grad_accum "$ACC" \
  --group_rep "$GROUP_REP" $MEM_ARGS --mem_loss_w 3 \
  --extra_rows "$DS/scenefly/rows.jsonl,$DS/scenefly_replay/rows.jsonl,$DS/worldedit/rows_train.jsonl" \
  --p_memunk 0 --p_trueunk 0 --p_unk2 0.5 "$@" 2>&1 | tee -a "$OUT/train_out.log"
