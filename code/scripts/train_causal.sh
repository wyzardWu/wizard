#!/bin/bash
# Stage 1: causal adaptation (teacher forcing with history corruption) from a bidirectional checkpoint (11K steps for the released model).
#   INIT=runs/bidirectional/dits/dit_step35000.safetensors OUT=runs/causal bash scripts/train_causal.sh [extra arguments]
# History window 20 latent frames + the first frame as sink. Re-running resumes from the latest checkpoint in OUT.
set -e
source "$(dirname "$0")/common.sh"
: "${OUT:?set OUT to the output directory}"
MAX=${MAX:-11000}; mkdir -p "$OUT"
st=$(latest "$OUT/dits")
if [ -n "$st" ]; then init="--init $OUT/dits/dit_step$st.safetensors --step0 $st --seed $st"; else : "${INIT:?set INIT to the bidirectional checkpoint}"; init="--init $INIT --step0 0"; fi
export MEMW_COVER=1 UNK_NOCOORD=1 WEDIT=1 MEMW_HISTW=20 MEMW_SINK=1 SA_P=0.5 SA_MAX=0.25 MEM_FUTURE=1 SA_GROUPS=lookback,lookback0,sfly,sfly_replay
CUDA_VISIBLE_DEVICES=$GPUS accelerate launch --config_file configs/accelerate_fsdp.yaml --num_processes "$NPROC" --main_process_port "${PORT:-29845}" \
  -m wizard.train.causal_selfaug --data "$DS/windows" --out "$OUT" $init --max_steps "$MAX" --save_every 500 --num_workers 6 --window 20 --sink 1 \
  --group_rep "$GROUP_REP,lookback:4,lookback0:1" $MEM_ARGS --p_memunk 0 --mem_loss_w 3 \
  --extra_rows "$DS/scenefly/rows.jsonl,$DS/scenefly_replay/rows.jsonl,$DS/synth_lookback_geo/rows_full.jsonl,$DS/worldedit/rows_train.jsonl" \
  --p_trueunk 0 --p_unk2 0.5 --log_every 5 "$@" 2>&1 | tee -a "$OUT/train_out.log"
