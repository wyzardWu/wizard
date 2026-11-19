"""Stage 1 with history corruption (the entry point used for the released causal teacher).

Under teacher forcing the clean history already contains almost the whole scene, so the model never learns to rely on memory when
its own history drifts. With probability ``SA_P`` the clean history is replaced by the model's one-step estimate from a noised
copy: x_t = (1 - s) x0 + s n, s ~ U(0, SA_MAX) per frame, x0_hat = x_t - s v(x_t) (no gradient). The sink frame, the memory frames
and the target stay clean. Only windows of ``SA_GROUPS`` are corrupted; the per-step decision is the same on every rank (FSDP needs
the same number of collective calls on every rank), and a rank whose window is not in SA_GROUPS discards the extra forward.

Environment: SA_P (0.7), SA_MAX (0.5), SA_GROUPS. Other arguments as wizard.train.causal. The training loop of
wizard.train.causal is reused with gradient accumulation patched in at import time."""
import os, sys, random, json
import torch
import wizard.train.causal as A
from wizard.train.causal import scene_gbuf, ident_aug, cat_cond, M_MAX
SA_P = float(os.environ.get("SA_P", "0.7")); SA_MAX = float(os.environ.get("SA_MAX", "0.5")); CALL = [0]; STAT = [0, 0]
SA_GROUPS = set(os.environ.get("SA_GROUPS", "lookback,lookback0,sfly,sfly_replay").split(","))
_get = A.WindowData.__getitem__
def _getitem(self, i):
    d = _get(self, i); d["sa_ok"] = torch.tensor(float(self.rows[i]["group"] in SA_GROUPS)); return d
A.WindowData.__getitem__ = _getitem


def loss_fn(model, core, sched, b, neg, a, dev):
    lat = b["latent"].to(dev).float(); act = b["actor"].to(dev).float(); mem = b["mem_lat"].to(dev).float()[:, :, :, 0].permute(0, 2, 1, 3, 4)
    if random.random() < a.p_memnoise: am = random.uniform(0, a.memnoise); mem = (1 - am) * mem + am * torch.randn_like(mem)
    B, C, Fl = lat.shape[:3]; assert B == 1
    idx = sched.sample_pf_timestep_ids(B, Fl, dev, ar_order_bias=0.0); noise = torch.randn_like(lat)
    noisy = sched.add_noise_pf(lat, noise, idx); ts = sched.timesteps.to(dev, torch.float32)[idx].clone()
    noisy[:, :, 0] = lat[:, :, 0]; ts[:, 0] = 0.0                                          # first frame always given
    gc = scene_gbuf(b["depth"][0].to(dev), b["pos"][0].to(dev), b["R"][0].to(dev), b["K"][0], b["origin"][0].to(dev), float(b["hp"][0]), None)
    tier_c = b["unknown"][0].to(dev).float()
    n = int(b["mem_n"][0]); gm = torch.zeros(M_MAX, 8, 60, 104, device=dev)
    for j in range(n):
        gm[j] = scene_gbuf(b["mem_depth"][0, j:j + 1].to(dev), b["mem_pos"][0, j:j + 1].to(dev), b["mem_R"][0, j:j + 1].to(dev), b["mem_K"][0, j], b["origin"][0].to(dev), float(b["hp"][0]))[:, 0]
    if random.random() < a.p_ident:
        aug = ident_aug([gc] + [gm[j:j + 1].permute(1, 0, 2, 3) for j in range(n)]); gc = aug[0]
        for j in range(n): gm[j] = aug[1 + j][:, 0]
    tier_m = b["mem_tier"][0].to(dev).float() / 3.0
    gc = torch.cat([gc[:3], (gc[3:4] > 0).to(gc.dtype), torch.zeros_like(gc[4:7]), tier_c[None].to(gc.dtype)], 0)
    gm = torch.cat([gm[:, :3], (gm[:, 3:4] > 0).to(gm.dtype), torch.zeros_like(gm[:, 4:7]), tier_m[:, None].to(gm.dtype)], 1)
    core._gb_in = (gc[None], gm[None]); core._mem_warp = None; core._mem_wrope = None
    ctx = b["context"].to(dev).to(torch.bfloat16)
    if random.random() < a.p_text_drop: ctx = neg.to(dev).to(torch.bfloat16)
    cond = torch.cat([torch.zeros_like(mem), act], 2)
    xn = cat_cond(torch.cat([mem, noisy], 2), cond); xc = cat_cond(torch.cat([mem, lat], 2), cond)
    ref = dict(lat=b["ref_lat"].to(dev), slots=b["ref_slots"].to(dev), cover=b["cover"].to(dev), mem_use=b["mem_use"], mem_n=b["mem_n"], mem_hide=b["mem_hide"])
    CALL[0] += 1
    if random.Random(1000003 * CALL[0] + 17).random() < SA_P:                             # same decision on every rank (FSDP: same number of forwards)
        with torch.no_grad():
            sig = torch.rand(B, Fl, device=dev) * SA_MAX; ia = sched.sigma_to_idx(sig.reshape(-1)).reshape(B, Fl)
            xa = sched.add_noise_pf(lat, torch.randn_like(lat), ia); tsa = sched.timesteps.to(dev, torch.float32)[ia].clone()
            xa[:, :, 0] = lat[:, :, 0]; tsa[:, 0] = 0.0
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                va = model(cat_cond(torch.cat([mem, xa], 2), cond), tsa, ctx, clean_x=xc, ref=ref)
            s_ = sched.sigmas.to(dev, torch.float32)[ia][:, None, :, None, None]
            hist = xa - s_ * va.float(); hist[:, :, 0] = lat[:, :, 0]                       # one-step x0 estimate; the sink stays clean
        if float(b["sa_ok"][0]) > 0.5: xc = cat_cond(torch.cat([mem, hist], 2), cond); STAT[0] += 1
        if os.environ.get("SA_DEBUG"): A._sa_dbg = dict(hist=hist.detach(), lat=lat.detach(), mem=mem.detach(), xc=xc.detach(), xn=xn.detach())
    STAT[1] += 1
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        v = model(xn, ts, ctx, clean_x=xc, ref=ref)
    core._gb_in = None
    se = ((v.float() - (noise - lat)).pow(2) * b["memw"].to(dev).float()[:, None]).mean(dim=(1, 3, 4)); keep = torch.ones_like(se); keep[:, 0] = 0.0
    if STAT[1] % 500 == 0: print(f"SELF_AUG {STAT[0]}/{STAT[1]} samples (p {SA_P} on {sorted(SA_GROUPS)}, sigma <= {SA_MAX})", flush=True)
    w = sched.training_weight(idx).float(); return (w * se * keep).sum() / keep.sum().clamp(min=1.0)


A.loss_fn = loss_fn
if os.environ.get("MEM_FUTURE") == "1":
    class ARDataFuture(A.WindowData):
        """SceneFly forward windows keep memory picks >= 3 s
        BEFORE the window (as ARData) or >= 3 s AFTER its end (f >= f0 + 121 + GAP_F); picks within 3 s of the window are dropped (near-copies).
        Memory slots carry no time stamp, so a future pick reads as any other memory frame; inference is unchanged (past memory only)."""
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
# --grad_accum N: N samples per GPU per optimizer step (as wizard/train/bidirectional.py). A.main is patched from its own source,
# so wizard/train/causal.py on disk is untouched; the patch is checked to apply exactly once.
import inspect, textwrap
_src = textwrap.dedent(inspect.getsource(A.main))
_old_arg = 'ap.add_argument("--limit", type=int, default=0)'
_old_loop = """        loss = loss_fn(model, core, sched, b, neg, a, accel.device)
        accel.backward(loss); gn = accel.clip_grad_norm_(model.parameters(), 1.0)"""
_new_loop = """        loss = loss_fn(model, core, sched, b, neg, a, accel.device)
        if a.grad_accum > 1:
            accel.backward(loss / a.grad_accum)
            for _ in range(a.grad_accum - 2):
                try: b = next(it)
                except StopIteration: it = iter(dl); b = next(it)
                accel.backward(loss_fn(model, core, sched, b, neg, a, accel.device) / a.grad_accum)
            try: b = next(it)
            except StopIteration: it = iter(dl); b = next(it)
            loss = loss_fn(model, core, sched, b, neg, a, accel.device); accel.backward(loss / a.grad_accum)
        else: accel.backward(loss)
        gn = accel.clip_grad_norm_(model.parameters(), 1.0)"""
assert _src.count(_old_arg) == 1 and _src.count(_old_loop) == 1, "wizard.train.causal.main changed: patch does not apply"
_src = _src.replace(_old_arg, _old_arg + '; ap.add_argument("--grad_accum", type=int, default=1)').replace(_old_loop, _new_loop)
exec(compile(_src, A.__file__ + " [main + grad_accum]", "exec"), A.__dict__)
if __name__ == "__main__":
    A.main()
