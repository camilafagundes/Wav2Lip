import os
import sys
from pathlib import Path

import cv2
import torch
from torch.utils.model_zoo import load_url

# PyTorch >= 2.6: pesos S3FD carregados com weights_only=False via safe_torch_load.
_p = Path(__file__).resolve().parent
while _p != _p.parent:
	if (_p / "utils" / "torch_compat.py").is_file():
		sys.path.insert(0, str(_p))
		break
	_p = _p.parent
else:
	raise ImportError(
		"Não encontrada avatar-ai/utils/torch_compat.py. Defina PYTHONPATH para a pasta avatar-ai."
	)
from utils.torch_compat import safe_torch_load

from ..core import FaceDetector

from .net_s3fd import s3fd
from .bbox import *
from .detect import *

models_urls = {
    's3fd': 'https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth',
}


class SFDDetector(FaceDetector):
    def __init__(self, device, path_to_detector=os.path.join(os.path.dirname(os.path.abspath(__file__)), 's3fd.pth'), verbose=False):
        super(SFDDetector, self).__init__(device, verbose)

        # Initialise the face detector
        if not os.path.isfile(path_to_detector):
            model_weights = load_url(models_urls['s3fd'])
        else:
            model_weights = safe_torch_load(path_to_detector)

        self.face_detector = s3fd()
        self.face_detector.load_state_dict(model_weights)
        self.face_detector.to(device)
        self.face_detector.eval()

    def detect_from_image(self, tensor_or_path):
        image = self.tensor_or_path_to_ndarray(tensor_or_path)

        bboxlist = detect(self.face_detector, image, device=self.device)
        keep = nms(bboxlist, 0.3)
        bboxlist = bboxlist[keep, :]
        bboxlist = [x for x in bboxlist if x[-1] > 0.32]

        return bboxlist

    def detect_from_batch(self, images):
        bboxlists = batch_detect(self.face_detector, images, device=self.device)
        keeps = [nms(bboxlists[:, i, :], 0.3) for i in range(bboxlists.shape[1])]
        bboxlists = [bboxlists[keep, i, :] for i, keep in enumerate(keeps)]
        bboxlists = [[x for x in bboxlist if x[-1] > 0.32] for bboxlist in bboxlists]

        return bboxlists

    @property
    def reference_scale(self):
        return 195

    @property
    def reference_x_shift(self):
        return 0

    @property
    def reference_y_shift(self):
        return 0
