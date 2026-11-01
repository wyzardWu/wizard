# Make the white that is connected to a figure's outer border transparent, with soft edges, then trim the margin.
# Usage: python3 tools/knockout.py in.jpg out.webp [white_threshold=250]
# Keep the threshold high: pale panel tints (e.g. 243,246,251 in fig3) must survive.
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

src, dst = sys.argv[1], sys.argv[2]
thr = int(sys.argv[3]) if len(sys.argv) > 3 else 250
im = Image.open(src).convert("RGB")
a = np.asarray(im).astype(np.float32)
w, h = im.size

# near-white pixels, padded with a white frame so one flood fill reaches every border pixel
near = (a.min(axis=2) >= thr).astype(np.uint8) * 255
m = Image.new("L", (w + 2, h + 2), 255)
m.paste(Image.fromarray(near), (1, 1))
ImageDraw.floodfill(m, (0, 0), 128)
bg = np.asarray(m)[1:-1, 1:-1] == 128

# soft edge: pixels near the background lose their white by colour-to-alpha
zone = np.asarray(Image.fromarray(bg.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(5))) > 0
alpha = np.ones((h, w), np.float32)
rgb = a.copy()
ca = ((255 - a) / 255).max(axis=2)              # alpha needed to reach this colour from white
ca = np.clip(ca, 1e-3, 1)
un = (a - 255 * (1 - ca[..., None])) / ca[..., None]
alpha[zone] = ca[zone]
rgb[zone] = un[zone]
alpha[bg] = 0
out = Image.fromarray(np.dstack([np.clip(rgb, 0, 255), alpha * 255]).astype(np.uint8))
# crop the empty margin so the figure's visible edges sit on the page grid
out = out.crop(out.getchannel("A").point(lambda v: 255 if v > 8 else 0).getbbox())
out.save(dst, "WEBP", quality=90, method=6)
print(dst, f"{bg.mean():.0%} transparent")
