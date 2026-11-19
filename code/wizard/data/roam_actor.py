"""Finding the player's cells in exploration windows, whose stored depth is 0 for both the sky and the (removed) player.

``roam_actor(depth [T, H, W]) -> actor [T, H, W] bool``: zero components that do not touch the top row and whose rim is near, or a
short zero run of a row with near depth at both ends. The window depth is then ``fill_env(depth, actor)`` as for every other source."""
import numpy as np
from scipy import ndimage
NEAR = 15.0; RUN = 30


def roam_actor(depth):
    d = np.asarray(depth, np.float32); out = np.zeros(d.shape, bool)
    for t in range(d.shape[0]):
        z = d[t]; lab, n = ndimage.label(z <= 0)
        if n == 0: continue
        top = set(np.unique(lab[0])) - {0}; dil = ndimage.grey_dilation(lab, size=3)
        rim = (z > 0) & (dil > 0)
        for k in range(1, n + 1):
            if k in top: continue
            rk = rim & (dil == k)
            if rk.sum() >= 3 and np.percentile(z[rk], 25) < NEAR: out[t] |= lab == k
        for y in range(z.shape[0]):
            row = z[y]; x = 0; W = len(row)
            while x < W:
                if row[x] > 0: x += 1; continue
                x1 = x
                while x1 < W and row[x1] <= 0: x1 += 1
                if x > 0 and x1 < W and x1 - x <= RUN and 0 < row[x - 1] < NEAR and 0 < row[x1] < NEAR: out[t, y, x:x1] = True
                x = x1
    return out
