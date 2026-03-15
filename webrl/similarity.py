import threading

import numpy as np
import torch
import lpips
from PIL import Image
from skimage.metrics import structural_similarity

TARGET_SIZE = (1280, 800)

_lpips_net = None
_lpips_lock = threading.Lock()


def _get_lpips_net():
    global _lpips_net
    if _lpips_net is None:
        with _lpips_lock:
            if _lpips_net is None:
                _lpips_net = lpips.LPIPS(net="alex").eval()
    return _lpips_net


def _prepare_images(img1: Image.Image, img2: Image.Image) -> tuple[Image.Image, Image.Image]:
    img1 = img1.convert("RGB").resize(TARGET_SIZE, Image.LANCZOS)
    img2 = img2.convert("RGB").resize(TARGET_SIZE, Image.LANCZOS)
    return img1, img2


def ssim_score(img1: Image.Image, img2: Image.Image) -> float:
    img1, img2 = _prepare_images(img1, img2)
    arr1 = np.array(img1)
    arr2 = np.array(img2)
    score = structural_similarity(arr1, arr2, channel_axis=2, data_range=255)
    return float(score)


def lpips_score(img1: Image.Image, img2: Image.Image) -> float:
    img1, img2 = _prepare_images(img1, img2)

    def to_tensor(img: Image.Image) -> torch.Tensor:
        arr = np.array(img).astype(np.float32) / 255.0
        arr = arr * 2.0 - 1.0  # normalize to [-1, 1]
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)

    t1 = to_tensor(img1)
    t2 = to_tensor(img2)
    with torch.no_grad():
        distance = _get_lpips_net()(t1, t2).item()
    return float(distance)
