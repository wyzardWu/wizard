# Shared settings of the launch scripts. Source it; override any variable from the environment.
: "${WIZARD_DATA:?set WIZARD_DATA to the root of the pre-encoded training data}"
: "${WIZARD_WAN:?set WIZARD_WAN to the Wan2.2-TI2V-5B directory}"
export WIZARD_DATA WIZARD_WAN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
NPROC=$(echo "$GPUS" | tr ',' '\n' | wc -l)
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DS=$WIZARD_DATA
# sampling weights of the data groups and the extra row files, as used for the released models
GROUP_REP=${GROUP_REP:-e1o:2,e2o:2,e1n:2,e2n:2,e1x:2,turn:2,roam:2,sfly:4,sfly_replay:6}
MEM_ARGS="--rep_mem 8 --rep_keys $DS/roam_memory/rep_keys_both30.json --mem_index mem_index_both.pt"
latest() { ls "$1"/dit_step*.safetensors 2>/dev/null | sed 's/.*step\([0-9]*\).*/\1/' | sort -n | tail -1; }
cd "$HERE"
