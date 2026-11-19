"""Base dataset over pre-encoded windows (video latent, proxy latent, references, text embedding, per-cell geometry, retrieved
memory frames), its collate function and flow-matching loss. Only ``BaseWindowData.pick`` (the per-chunk memory pool) is used, by the
inference scripts; ``wizard.train.bidirectional`` holds the dataset and loss the released models were trained with."""
import os, sys, time, argparse, json, random, collections, math
import numpy as np
import torch, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__))
from wizard.models.dit import GameMasterDiT, ti2v_5b_config, tiny_config
from wizard.models.flow_match import FlowMatchScheduler
from wizard.models.proxy import attach_cond, load_official, cat_cond
from wizard.models.wizard_dit import attach_wizard, scene_gbuf, ident_aug, M_MAX, CH, HW
from wizard.memory.retrieve import retrieve_chunks, chunk_pixels, MIN_GAIN
import wizard.memory.gbuffer as G
from wizard.data.env_fill import fill_env, smooth_env
from wizard.data.roam_actor import roam_actor
from wizard.paths import WAN_DIR, DATA_ROOT
CK = WAN_DIR; DS = DATA_ROOT; MEMROAM = f"{DS}/roam_memory"


def roam_depth(d):
    """the roaming window's depth with the player's hole filled by the environment behind (wizard.data.roam_actor + fill_env)"""
    z = d.float().numpy(); a = roam_actor(z); return torch.from_numpy(smooth_env(np.stack([fill_env(z[t], a[t]) for t in range(len(z))]), a))


def lt(p): return torch.load(p, map_location="cpu", weights_only=False)


class BaseWindowData(torch.utils.data.Dataset):
    def __init__(self, root, rows, a):
        self.O, self.rows, self.a = root, rows, a
        T = lt(f"{DS}/samples/text_table.pt"); self.emb, self.L = T["emb"], T["text_len"]
        self.roam_text = lt(f"{MEMROAM}/text.pt")["emb"]; self.mem_index = lt(f"{MEMROAM}/mem_index.pt"); self.meta = collections.OrderedDict()
        self.with_prev = [i for i, r in enumerate(rows) if r["prev"]]

    def __len__(self): return len(self.rows)

    def pad(self, e): return torch.cat([e, e.new_zeros(self.L - e.shape[0], e.shape[1])]) if e.shape[0] < self.L else e[:self.L]

    def bankmeta(self, s):
        if s not in self.meta:
            self.meta[s] = lt(f"{MEMROAM}/bank/{s}/meta.pt")
            if len(self.meta) > 16: self.meta.popitem(last=False)
        return self.meta[s]

    def first_frame(self, key):
        """a window's first frame as a memory candidate: latent [48,1,30,52], depth/pos/R/K of its frame 0, its character cover [15,26,3]"""
        c = lt(f"{DS}/samples/clips/clip_{key}.pt"); g = lt(f"{self.O}/geometry/clip_{key}.pt")
        return dict(lat=c["latent"][:, :1], depth=g["depth"][0].float() * (~g["actor"][0]), pos=g["pos"][0], R=g["R"][0], K=g["K"], cover=c["cover"][0].float(), slots=c["ref_slots"])

    def other_frame(self):
        r = self.rows[random.choice(self.with_prev)]; return self.first_frame(random.choice(r["prev"]))

    def pick(self, per):
        pool = []
        for rk in range(3):
            for p in per:
                if rk < len(p) and p[rk] not in pool and len(pool) < M_MAX: pool.append(p[rk])
        use = torch.zeros(len(CH), M_MAX, dtype=torch.bool)
        for c, p in enumerate(per):
            for f in p:
                if f in pool: use[c, pool.index(f)] = True
        return pool, use

    def __getitem__(self, i):
        r = self.rows[i]; a = self.a
        if r["group"] == "roam": return self.roam(r)
        c = lt(f"{DS}/samples/clips/clip_{r['key']}.pt"); g = lt(f"{self.O}/geometry/clip_{r['key']}.pt"); act = lt(r.get("actor") or f"{self.O}/actor/clip_{r['key']}.pt")["actor"]
        dep = torch.from_numpy(smooth_env(g["depth"].float().numpy(), g["actor"].numpy())); unk = g.get("unknown")     # rows -> 2D harmonic
        cands = {k: self.first_frame(k) for k in r["prev"]}; per = [[] for _ in CH]
        if cands:
            P = G.world_points(dep, g["pos"], g["R"], g["K"]); ok = dep > 0; pts, tcs = [], []
            for lats in CH:
                px = chunk_pixels(lats)[::4]; pts.append(P[px][:, ::4, ::4][ok[px][:, ::4, ::4]]); tcs.append(chunk_pixels(lats)[0] / 30.0)
            ks = list(cands); bank = dict(depth=torch.stack([cands[k]["depth"] for k in ks]), pos=torch.stack([cands[k]["pos"] for k in ks]),
                                          R=torch.stack([cands[k]["R"] for k in ks]), K=torch.stack([cands[k]["K"] for k in ks]),
                                          t=torch.tensor([-4.0 * (j + 1) for j in range(len(ks))], dtype=torch.float64))
            picks, _ = retrieve_chunks(pts, bank, tcs, k=3, min_gain=MIN_GAIN, context=dict(depth=dep[:1], pos=g["pos"][:1], R=g["R"][:1], K=g["K"]))
            per = [[ks[j] for j, _ in pk] for pk in picks]
        for c_ in range(len(per)):
            if random.random() < a.p_nomem: per[c_] = []
            elif per[c_] and random.random() < a.p_irrel:
                wk = f"other{c_}"; cands[wk] = self.other_frame(); per[c_][random.randrange(len(per[c_]))] = wk
        pool, use = self.pick(per); fr = [cands[k] for k in pool]
        return self.pack(c["latent"], act, self.pad(self.emb[c["prompt"]]), c["ref_latents"], c["ref_slots"], c["cover"], g, dep, unk, fr, use, c["ref_slots"])

    def roam(self, r):
        a = self.a; name = os.path.basename(r["video"]); d = lt(r["video"]); ix = self.mem_index[name]; s = ix["seg"]; m = self.bankmeta(s)
        per = []
        for ch in ix["chunks"]:
            p = [f for f, gn in ch if gn >= MIN_GAIN][:3]
            if random.random() < a.p_nomem: p = []
            elif p and random.random() < a.p_irrel: p[random.randrange(len(p))] = "other"
            per.append(p)
        pool, use = self.pick(per); fr = []
        for f in pool:
            if f == "other": fr.append(self.other_frame()); continue
            j = int((m["f"] == f).nonzero()[0]); x = lt(f"{MEMROAM}/bank/{s}/{f}.pt")
            fr.append(dict(lat=x["lat"], depth=m["depth"][j].float(), pos=m["pos"][j], R=m["R"][j], K=m["K"][j], cover=torch.zeros(15, 26, 3), slots=torch.zeros(2, dtype=torch.long)))
        g = dict(pos=d["pos"], R=d["R"], K=d["K"], origin=d["origin"], hp=d["hp"])
        return self.pack(d["latent"], d["skel"], self.pad(self.roam_text), torch.zeros(2, 48, 1, 30, 30, dtype=torch.bfloat16), torch.zeros(2, dtype=torch.long),
                         torch.zeros(31, 15, 26, 3, dtype=torch.float16), g, roam_depth(d["depth"]), None, fr, use, torch.zeros(2, dtype=torch.long))

    def pack(self, latent, actor, ctx, ref_lat, ref_slots, cover, g, dep, unk, fr, use, slots_now):
        n = len(fr); z = lambda shape, dt: torch.zeros((M_MAX,) + shape, dtype=dt)
        mem_lat, mem_dep, mem_pos, mem_R, mem_K, hide = z((48, 1, 30, 52), torch.bfloat16), z((60, 104), torch.float32), z((3,), torch.float32), z((3, 3), torch.float32), z((4,), torch.float32), z((HW,), torch.bool)
        live = [int(s) for s in slots_now.tolist() if int(s) > 0]                     # the slots that have a reference in THIS sample
        for j, f in enumerate(fr):
            mem_lat[j] = f["lat"].to(torch.bfloat16); mem_dep[j] = f["depth"]; mem_pos[j] = f["pos"]; mem_R[j] = f["R"]; mem_K[j] = f["K"].float()
            cov = f["cover"]; hm = torch.zeros(15, 26, dtype=torch.bool)
            for s in live: hm |= cov[..., s - 1] > 0.02
            hide[j] = (F.max_pool2d(hm.float()[None, None], 3, 1, 1)[0, 0] > 0).reshape(-1)            # grown one token
        return dict(latent=latent, actor=actor, context=ctx, ref_lat=ref_lat, ref_slots=ref_slots.long(), cover=cover.float(), depth=dep, pos=g["pos"], R=g["R"], K=g["K"].float(),
                    origin=g["origin"], hp=torch.tensor(float(g["hp"])), unknown=(unk if unk is not None else torch.zeros(dep.shape, dtype=torch.bool)),
                    mem_lat=mem_lat, mem_depth=mem_dep, mem_pos=mem_pos, mem_R=mem_R, mem_K=mem_K, mem_n=torch.tensor(n), mem_use=use, mem_hide=hide)


def collate(b): return {k: torch.stack([x[k] for x in b]) for k in b[0]}


def loss_fn(model, core, sched, b, neg, a, dev):
    lat = b["latent"].to(dev).float(); act = b["actor"].to(dev).float(); mem = b["mem_lat"].to(dev).float()[:, :, :, 0].permute(0, 2, 1, 3, 4)
    B, C, Fl = lat.shape[:3]; assert B == 1
    idx = sched.sample_timestep_ids(B, dev); noise = torch.randn_like(lat); noisy = sched.add_noise(lat, noise, idx)
    ts = sched.timesteps.to(dev, torch.float32)[idx][:, None].expand(B, Fl).clone(); noisy[:, :, 0] = lat[:, :, 0]; ts[:, 0] = 0.0   # first frame always given
    gc = scene_gbuf(b["depth"][0].to(dev), b["pos"][0].to(dev), b["R"][0].to(dev), b["K"][0], b["origin"][0].to(dev), float(b["hp"][0]), b["unknown"][0].to(dev))
    n = int(b["mem_n"][0]); gm = torch.zeros(M_MAX, 8, 60, 104, device=dev)
    for j in range(n):
        gm[j] = scene_gbuf(b["mem_depth"][0, j:j + 1].to(dev), b["mem_pos"][0, j:j + 1].to(dev), b["mem_R"][0, j:j + 1].to(dev), b["mem_K"][0, j], b["origin"][0].to(dev), float(b["hp"][0]))[:, 0]
    if random.random() < a.p_ident:                                                     # coordinates as identities: same piecewise rigid transform
        aug = ident_aug([gc] + [gm[j:j + 1].permute(1, 0, 2, 3) for j in range(n)]); gc = aug[0]
        for j in range(n): gm[j] = aug[1 + j][:, 0]
    core._gb_in = (gc[None], gm[None])
    ctx = b["context"].to(dev).to(torch.bfloat16)
    if random.random() < a.p_text_drop: ctx = neg.to(dev).to(torch.bfloat16)
    x = torch.cat([mem, noisy], 2); cond = torch.cat([torch.zeros_like(mem), act], 2); tsa = torch.cat([torch.zeros(B, M_MAX, device=dev), ts], 1)
    ref = dict(lat=b["ref_lat"].to(dev), slots=b["ref_slots"].to(dev), cover=b["cover"].to(dev), mem_use=b["mem_use"], mem_n=b["mem_n"], mem_hide=b["mem_hide"])
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        v = model(cat_cond(x, cond), tsa, ctx, ref=ref)
    core._gb_in = None
    se = (v.float() - (noise - lat)).pow(2).mean(dim=(1, 3, 4)); keep = torch.ones_like(se); keep[:, 0] = 0.0
    w = sched.training_weight(idx).float()[:, None]; return (w * se * keep).sum() / keep.sum().clamp(min=1.0)

