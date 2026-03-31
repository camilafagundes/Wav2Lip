"""cv2.imread não lida bem com caminhos não-ASCII no Windows; imdecode + fromfile contorna isso."""

from __future__ import annotations

import numpy as np
import cv2


def imread(path: str, flags: int = cv2.IMREAD_COLOR):
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)
