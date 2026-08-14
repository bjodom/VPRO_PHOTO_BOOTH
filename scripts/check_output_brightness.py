"""Report mean/max luminance for rendered outputs to spot black or near-black images."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
patterns = sys.argv[1:] or [
    "outputs/juggernaut_prompt_tests/*.jpg",
    "outputs/img2img_probe/*.jpg",
]

for pattern in patterns:
    print(f"\n== {pattern}")
    for path in sorted(REPO_ROOT.glob(pattern)):
        gray = np.asarray(Image.open(path).convert("L"))
        flag = "  <-- BLACK" if gray.max() < 8 else ("  <-- dark" if gray.mean() < 40 else "")
        print(f"{path.name:48s} mean={gray.mean():7.2f} max={gray.max():4d}{flag}")
