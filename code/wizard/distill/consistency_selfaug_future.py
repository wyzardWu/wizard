"""Stage 2 entry point used for the released run: wizard.distill.consistency_selfaug with past and future memory candidates
(as the causal teacher was trained) and a free-disk guard for the periodic EMA saves (``CD_FREE_GB``, default 15).

    accelerate launch --config_file configs/accelerate_fsdp_dmd.yaml --num_processes 8 -m wizard.distill.consistency_selfaug_future ..."""
import os, json
import wizard.train.causal as A
class ARDataFuture(A.WindowData):
    def __init__(self, root, rows, a):
        A.WindowData.__init__(self, root, rows, a)
        info = json.load(open(f"{A.SFLY}/index.json")) if os.path.exists(f"{A.SFLY}/index.json") else {}
        sf = A.lt(f"{A.SFLY}/mem_index.pt") if os.path.exists(f"{A.SFLY}/mem_index.pt") else {}
        n0 = n1 = 0
        for name in sf:
            if name not in info or name not in self.mem_index: continue
            f0 = int(info[name]["f0"]); e = dict(self.mem_index[name])
            ch = [[(f, g) for f, g in c if f <= f0 - A.GAP_F or f >= f0 + 121 + A.GAP_F] for c in e["chunks"]]
            n0 += sum(len(c) for c in e["chunks"]); n1 += sum(len(c) for c in ch); e["chunks"] = ch; self.mem_index[name] = e
        self.past_info = (n0, n1); print(f"MEM_FUTURE SceneFly picks kept (>= 3 s before or after the window): {n1} of {n0}", flush=True)
A.ARData = ARDataFuture
import wizard.distill.consistency_selfaug as _selfaug                      # installs its cd_loss into wizard.distill.consistency
import wizard.distill.consistency as CD
import sys, time, shutil
_save = CD.st.save_file; FREE_GB = float(os.environ.get("CD_FREE_GB", 15))
def _guarded_save(sd, path, *x, **k):
    """the disk is shared and often full: skip an intermediate EMA save rather than crash; the last one waits for space"""
    last = "--max_steps" in sys.argv and f"ema_step{sys.argv[sys.argv.index('--max_steps') + 1]}." in os.path.basename(path)
    while shutil.disk_usage(os.path.dirname(path)).free < FREE_GB * 2 ** 30:
        print(f"LOW_DISK {shutil.disk_usage(os.path.dirname(path)).free / 2 ** 30:.1f} GB free: {'waiting to save' if last else 'SKIPPED'} {os.path.basename(path)}", flush=True)
        if not last: return
        time.sleep(60)
    return _save(sd, path, *x, **k)
CD.st.save_file = _guarded_save
if __name__ == "__main__": CD.main()
