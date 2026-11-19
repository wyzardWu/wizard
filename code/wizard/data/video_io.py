"""Reading a video as the training pipeline does: scaled to height 480 and cropped to 832 columns."""
import subprocess
import numpy as np
W, H, N = 832, 480, 121; GREY = (165, 165, 169)


def read(path, frames):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vf", f"scale=-2:{H}:flags=bicubic,crop={W}:{H}", "-frames:v", str(frames), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True).stdout
    v = np.frombuffer(raw, np.uint8).reshape(-1, H, W, 3)
    return v if len(v) >= frames else None
