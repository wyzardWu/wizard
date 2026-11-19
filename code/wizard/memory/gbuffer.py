"""Per-cell scene geometry (the "G-buffer") on the 104x60 token grid, computed on the GPU from planar depth and cameras.

The same function is used for training, for visual checks and for inference. Channels per grid cell: world coordinate relative to
the window's origin in player heights, log depth, normal, and validity bits. Sky and character cells are zero."""
import torch, math
import torch.nn.functional as F


def world_points(depth, pos, R, K):
    """depth [T,H,W] (m, 0 = none), pos [T,3], R [T,3,3] rows = right, up, forward; K = (fx, fy, cx, cy) of the grid -> [T,H,W,3] absolute metres."""
    T, H, W = depth.shape; fx, fy, cx, cy = [float(v) for v in K]
    v, u = torch.meshgrid(torch.arange(H, device=depth.device, dtype=torch.float32) + 0.5, torch.arange(W, device=depth.device, dtype=torch.float32) + 0.5, indexing="ij")
    z = depth.float(); X = (u - cx) / fx * z; Y = (v - cy) / fy * z
    return pos[:, None, None, :] + X[..., None] * R[:, None, None, 0, :] - Y[..., None] * R[:, None, None, 1, :] + z[..., None] * R[:, None, None, 2, :]


def gbuf(depth, pos, R, K, origin, hp, normal=None):
    """normal: the stored int8 normals [T,H,W,3] (computed from the full-resolution depth); None = central differences on the grid (noisy far away)."""
    known = depth > 0; P = world_points(depth, pos, R, K)
    Wc = (P - origin.view(1, 1, 1, 3)) / hp                                       # player heights from the window's origin
    Pc = P.permute(0, 3, 1, 2)                                                    # normals from neighbouring cells (central differences, known cells only)
    dx = torch.zeros_like(Pc); dy = torch.zeros_like(Pc)
    dx[..., 1:-1] = Pc[..., 2:] - Pc[..., :-2]; dy[..., 1:-1, :] = Pc[..., 2:, :] - Pc[..., :-2, :]
    kn = known.float()[:, None]; okx = torch.zeros_like(kn); oky = torch.zeros_like(kn)
    okx[..., 1:-1] = kn[..., 2:] * kn[..., :-2]; oky[..., 1:-1, :] = kn[..., 2:, :] * kn[..., :-2, :]
    n = torch.cross(dx, dy, dim=1); n = n / (n.norm(dim=1, keepdim=True) + 1e-6)
    ray = Pc - pos.view(-1, 3, 1, 1); n = torch.where((n * ray).sum(1, keepdim=True) > 0, -n, n)   # facing the camera: against the ray to the point
    n = n * (okx * oky)
    if normal is not None: n = normal.float().permute(0, 3, 1, 2) / 127.0 * known.float()[:, None]
    ld = torch.where(known, torch.log1p(depth.float()) / math.log1p(300.0), torch.zeros_like(depth.float()))[:, None]
    g = torch.cat([Wc.permute(0, 3, 1, 2), ld, n, known.float()[:, None], torch.zeros_like(ld)], 1)
    return g * known.float()[:, None].clamp(max=1.0) + torch.zeros_like(g)          # unknown / sky / player cells: all zero (known = 0)


def unknown_mask(P, known, hp, n_blobs=(1, 4), radius=(1.0, 5.0), gen=None):
    """P [T,H,W,3] the TRUE world points, known [T,H,W]: a few spheres of the WORLD (centred on world points of a random frame, radius in player
    heights) whose geometry is withheld -- a region of the world, so it moves with the camera like the place it is."""
    T = P.shape[0]; t = int(torch.randint(T, (1,), generator=gen)); idx = known[t].nonzero()
    m = torch.zeros(known.shape, dtype=torch.bool, device=P.device)
    if len(idx) == 0: return m
    for _ in range(int(torch.randint(n_blobs[0], n_blobs[1] + 1, (1,), generator=gen))):
        c = idx[int(torch.randint(len(idx), (1,), generator=gen))]; ctr = P[t, c[0], c[1]]
        r = (radius[0] + (radius[1] - radius[0]) * float(torch.rand(1, generator=gen))) * hp
        m |= ((P - ctr).norm(dim=-1) < r) & known
    return m


def withhold(g, mask):
    """g [T,9,H,W], mask [T,H,W] -> the masked cells that had geometry: everything zero except withheld = 1 (sky / player cells stay as they are)."""
    hit = (mask & (g[:, 7] > 0)).float()[:, None]; g = g * (1 - hit); g[:, 8:9] = hit; return g
