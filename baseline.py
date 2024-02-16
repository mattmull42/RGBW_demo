import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from skimage import io, exposure, metrics
from scipy import interpolate, ndimage

from src.forward_model.forward_operator import forward_operator
from src.forward_model.operators import *

from src.inversions.baseline_method.inversion_baseline import *


error = []
error_noise = []

for image_name in listdir('PAirMax/'):
    if image_name.startswith('W4_Mexi_Nat'):
        img_ms = io.imread('PAirMax/' + image_name + '/RR/GT.tif').astype(np.float64)
        img_pan = io.imread('PAirMax/' + image_name + '/RR/PAN.tif').astype(np.float64)

        if image_name.startswith('Pl'):
            img_ms /= 4095
            img_pan /= 4095

            np.clip(img_pan, 0, 1, img_pan)

            img_ms = img_ms[:, :, [2, 1, 0]]

            img_ms = exposure.rescale_intensity(img_ms, in_range=(0, 0.3))
            img_pan = exposure.rescale_intensity(img_pan, in_range=(0, 0.3))

        elif image_name.startswith('W2'):
            img_ms /= 2047
            img_pan /= 2047

            np.clip(img_pan, 0, 1, img_pan)

            img_ms = img_ms[:, :, [5, 2, 1]]

            img_ms = exposure.rescale_intensity(img_ms, in_range=(0, 0.6))
            img_pan = exposure.rescale_intensity(img_pan, in_range=(0, 0.6))

        elif image_name.startswith('W3'):
            img_ms /= 2047
            img_pan /= 2047

            np.clip(img_pan, 0, 1, img_pan)

            img_ms = img_ms[:, :, [2, 1, 0]]

            img_ms = exposure.rescale_intensity(img_ms, in_range=(0, 0.35))
            img_pan = exposure.rescale_intensity(img_pan, in_range=(0, 0.35))

        elif image_name.startswith('W4'):
            img_ms /= 2047
            img_pan /= 2047

            np.clip(img_pan, 0, 1, img_pan)

            img_ms = img_ms[:, :, [2, 1, 0]]

            img_ms[:, :, 0] = exposure.equalize_hist(img_ms[:, :, 0])
            img_ms[:, :, 1] = exposure.equalize_hist(img_ms[:, :, 1])
            img_ms[:, :, 2] = exposure.equalize_hist(img_ms[:, :, 2])

            img_pan = exposure.equalize_hist(img_pan)

        elif image_name.startswith('GE'):
            img_ms /= 2047
            img_pan /= 2047

            np.clip(img_pan, 0, 1, img_pan)

            img_ms = img_ms[:, :, [2, 1, 0]]

            img_ms[:, :, 0] = exposure.equalize_hist(img_ms[:, :, 0])
            img_ms[:, :, 1] = exposure.equalize_hist(img_ms[:, :, 1])
            img_ms[:, :, 2] = exposure.equalize_hist(img_ms[:, :, 2])

            img_pan = exposure.equalize_hist(img_pan)

        elif image_name.startswith('S7'):
            img_ms /= 4095
            img_pan /= 4095

            np.clip(img_pan, 0, 1, img_pan)

            img_ms = img_ms[:, :, [2, 1, 0]]

            img_ms = exposure.rescale_intensity(img_ms, in_range=(0, 0.5))
            img_pan = exposure.rescale_intensity(img_pan, in_range=(0, 0.5))

        y = img_pan.copy()

        # sparse_3
        y[::8, ::8] = img_ms[::8, ::8, 0]

        y[4::8, ::8] = img_ms[4::8, ::8, 1]
        y[::8, 4::8] = img_ms[::8, 4::8, 1]

        y[4::8, 4::8] = img_ms[4::8, 4::8, 2]

        # kodak
        # y[3::4, 2::4] = img_ms[3::4, 2::4, 0]
        # y[2::4, 3::4] = img_ms[2::4, 3::4, 0]

        # y[3::4, ::4] = img_ms[3::4, ::4, 1]
        # y[2::4, 1::4] = img_ms[2::4, 1::4, 1]
        # y[1::4, 2::4] = img_ms[1::4, 2::4, 1]
        # y[::4, 3::4] = img_ms[::4, 3::4, 1]

        # y[1::4, ::4] = img_ms[1::4, ::4, 2]
        # y[::4, 1::4] = img_ms[::4, 1::4, 2]

        # sony
        # y[3::4, 2::4] = img_ms[3::4, 2::4, 0]
        # y[1::4, ::4] = img_ms[1::4, ::4, 0]

        # y[3::4, ::4] = img_ms[3::4, ::4, 1]
        # y[2::4, 1::4] = img_ms[2::4, 1::4, 1]
        # y[1::4, 2::4] = img_ms[1::4, 2::4, 1]
        # y[::4, 3::4] = img_ms[::4, 3::4, 1]

        # y[2::4, 3::4] = img_ms[2::4, 3::4, 2]
        # y[::4, 1::4] = img_ms[::4, 1::4, 2]

        p_inv = Inverse_problem('sparse_3', False, 0, img_ms.shape, np.array([650, 525, 480]), 'dirac')
        res = p_inv(y)


        # y_interp = np.zeros_like(img_ms)

        # d1, d2 = np.arange(img_ms.shape[0]), np.arange(img_ms.shape[1])
        # xg, yg = np.meshgrid(d1, d2, indexing='ij')

        # x_r = np.sort(list(range(1, img_ms.shape[0], 4)) + list(range(3, img_ms.shape[0], 4)))
        # y_r = np.sort(list(range(2, img_ms.shape[0], 4)) + list(range(0, img_ms.shape[0], 4)))
        # xg_r, yg_r = np.meshgrid(x_r, y_r, indexing='ij')
        # data_r = y[xg_r, yg_r]

        # x_g = np.arange(img_ms.shape[0])
        # y_g = np.arange(img_ms.shape[1])
        # xg_g, yg_g = np.meshgrid(x_g, y_g, indexing='ij')
        # data_g = y[xg_g, yg_g]

        # x_b = np.sort(list(range(0, img_ms.shape[0], 4)) + list(range(2, img_ms.shape[0], 4)))
        # y_b = np.sort(list(range(1, img_ms.shape[0], 4)) + list(range(3, img_ms.shape[0], 4)))
        # xg_b, yg_b = np.meshgrid(x_b, y_b, indexing='ij')
        # data_b = y[xg_b, yg_b]

        # interp_r = interpolate.RegularGridInterpolator((x_r, y_r), data_r, bounds_error=False, fill_value=0.5)
        # y_interp[:, :, 0] = interp_r((xg, yg))

        # interp_g = interpolate.RegularGridInterpolator((x_g, y_g), data_g, bounds_error=False, fill_value=0.5)
        # y_interp[:, :, 1] = interp_g((xg, yg))

        # interp_b = interpolate.RegularGridInterpolator((x_b, y_b), data_b, bounds_error=False, fill_value=0.5)
        # y_interp[:, :, 2] = interp_b((xg, yg))

        # W_HR = y.copy()
        # W_HR[1::2, ::2] = 0
        # W_HR[::2, 1::2] = 0
        # ker = np.array([[0, 1, 0], [1, 4, 1], [0, 1, 0]]) / 4
        # W_HR = ndimage.convolve(W_HR, ker)

        # Y_LF_HR = np.mean(y_interp, axis=2)

        # res = y_interp + (W_HR - Y_LF_HR)[:, :, np.newaxis]

        # np.clip(res, 0, 1, res)

        plt.imsave(f'sony_baseline.png', res)



    # error.append(metrics.mean_squared_error(img_ms, res))
    # error_noise.append(metrics.mean_squared_error(img_ms, x_hat_noise))

# print(f'Normal : {np.mean(error):.4f}, {np.std(error):.4f}')
# print(f'Bruit : {np.mean(error_noise):.4f}, {np.std(error_noise):.4f}')