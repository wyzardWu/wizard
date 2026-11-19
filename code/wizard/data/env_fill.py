"""Completing the scene depth behind the characters.

``fill_env(z, actor)``: z [H, W] planar depth (0 = sky / nothing), actor [H, W] bool (cells covered by characters). Row by row, each
run of actor cells takes the environment of its nearest non-actor neighbours on the left and right: both sides have depth ->
linear in 1/z between them; both sides sky -> sky; one side sky -> the nearer side. Memory frames keep their actor cells empty."""
import numpy as np


def fill_env(z, actor):
    z = np.asarray(z, np.float32); actor = np.asarray(actor, bool); out = z.copy()
    if not actor.any(): return out
    out[actor] = 0.0; H, W = z.shape
    for y in np.nonzero(actor.any(1))[0]:
        row = actor[y]; x = 0
        while x < W:
            if not row[x]: x += 1; continue
            x1 = x
            while x1 < W and row[x1]: x1 += 1                                     # run [x, x1)
            L = z[y, x - 1] if x > 0 else None; Rv = z[y, x1] if x1 < W else None
            if L is None and Rv is None: x = x1; continue
            if L is None or Rv is None:
                v = L if Rv is None else Rv; out[y, x:x1] = v if v > 0 else 0.0
            elif L > 0 and Rv > 0:
                t = (np.arange(x, x1) - (x - 1)) / (x1 - (x - 1)); out[y, x:x1] = 1.0 / ((1 - t) / L + t / Rv)
            elif L <= 0 and Rv <= 0: out[y, x:x1] = 0.0
            else:                                                                  # one side sky: each cell takes its nearer side
                for c in range(x, x1):
                    near_left = (c - (x - 1)) <= (x1 - c); v = L if near_left else Rv; out[y, c] = v if v > 0 else 0.0
            x = x1
    return out


if __name__ == "__main__":
    H, W = 60, 104; v = np.arange(H)[:, None] + 0.5; z = np.where(v > 30, 1.6 * 60 / np.maximum(v - 30, 1e-3), 0.0) * np.ones((1, W)); z[z > 300] = 0
    a = np.zeros((H, W), bool); a[35:58, 45:55] = True; zf = fill_env(np.where(a, 0, z), a)
    err = np.abs(zf[a] - z[a]) / z[a]; print("plane behind a person: median rel. error %.4f, max %.4f" % (np.median(err), err.max()))
    a2 = np.zeros((H, W), bool); a2[20:40, 45:55] = True; zf2 = fill_env(np.where(a2, 0, z), a2)
    print("person against the sky: cells above the horizon filled:", int((zf2[20:30, 45:55] > 0).sum()), "of 100 (want 0); below filled:", int((zf2[31:40, 45:55] > 0).sum()), "of 90")
    zw = z.copy(); zw[:, 60:] = np.where(zw[:, 60:] > 0, np.minimum(zw[:, 60:], 5.0), 5.0)        # a wall 5 m away on the right half
    a3 = np.zeros((H, W), bool); a3[10:50, 55:65] = True; zf3 = fill_env(np.where(a3, 0, zw), a3)
    print("person in front of a wall edge: left part ~sky/ground, right part ~5 m:", np.round(zf3[15, 55:65], 1).tolist())


def smooth_env(z, actor, iters=300):
    """the row-wise fill leaves rows independent -> horizontal stripes in the normals that trace the character's outline (a shape leak).
    z [T,H,W] already row-filled, actor [T,H,W]: keep the row rule's sky / geometry decision, then make the filled geometry cells HARMONIC in
    disparity (Jacobi, the other cells fixed; sky neighbours ignored): exact on any plane (disparity affine in the image), smooth across rows."""
    z = np.asarray(z, np.float32); U = np.asarray(actor, bool) & (z > 0)
    if not U.any(): return z
    d = np.where(z > 0, 1.0 / np.maximum(z, 1e-6), 0.0).astype(np.float32); ok = (z > 0).astype(np.float32)
    for _ in range(iters):
        p = np.pad(d, ((0, 0), (1, 1), (1, 1))); q = np.pad(ok, ((0, 0), (1, 1), (1, 1)))
        s = p[:, :-2, 1:-1] + p[:, 2:, 1:-1] + p[:, 1:-1, :-2] + p[:, 1:-1, 2:]; n = q[:, :-2, 1:-1] + q[:, 2:, 1:-1] + q[:, 1:-1, :-2] + q[:, 1:-1, 2:]
        d = np.where(U, s / np.maximum(n, 1), d)
    return np.where(U, 1.0 / np.maximum(d, 1e-6), z).astype(np.float32)
