from PIL import Image
import numpy as np


def initialize_input(image_path):
    if image_path.endswith('.npy'):
        res = np.load(image_path)[:, :, ::-1]

    else:
        res = np.array(Image.open(image_path)) / 255

    spectral_stencil = np.array([650, 525, 480])

    if res.shape[2] > 3:
        return res[:, :, :3], spectral_stencil

    return res, spectral_stencil


def initialize_inverse_input(image_path):
    if image_path.endswith('.png'):
        res = np.array(Image.open(image_path)) / 255

    elif image_path.endswith('.tiff'):
        res = np.array(Image.open(image_path), dtype='float') / 4095

    if len(res.shape) > 2:
        return res[:, :, 0]

    return res