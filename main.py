import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate, ndimage

from src.forward_operator.forward_operator import forward_operator
from src.forward_operator.operators import *

from src.inversions.baseline_method.inversion_baseline import *

from src.input_initialization import initialize_input


CFA = 'kodak'

x, spectral_stencil = initialize_input('input/01690.png')
spectral_stencil = np.flip(spectral_stencil)

cfa_op = cfa_operator(CFA, x.shape, spectral_stencil, 'dirac')
forward_op = forward_operator([cfa_op])

y = forward_op.direct(x)

y_interp = np.zeros_like(x)

d1, d2 = np.arange(x.shape[0]), np.arange(x.shape[1])
xg, yg = np.meshgrid(d1, d2, indexing='ij')

x_r = np.sort(list(range(2, x.shape[0], 4)) + list(range(3, x.shape[0], 4)))
y_r = x_r
xg_r, yg_r = np.meshgrid(x_r, y_r, indexing='ij')
data_r = y[xg_r, yg_r]

x_g = np.arange(x.shape[0])
y_g = np.arange(x.shape[1])
xg_g, yg_g = np.meshgrid(x_g, y_g, indexing='ij')
data_g = y[xg_g, yg_g]

x_b = np.sort(list(range(0, x.shape[0], 4)) + list(range(1, x.shape[0], 4)))
y_b = x_b
xg_b, yg_b = np.meshgrid(x_b, y_b, indexing='ij')
data_b = y[xg_b, yg_b]

interp_r = interpolate.RegularGridInterpolator((x_r, y_r), data_r, bounds_error=False, fill_value=0.5)
y_interp[:, :, 0] = interp_r((xg, yg))

interp_g = interpolate.RegularGridInterpolator((x_g, y_g), data_g, bounds_error=False, fill_value=0.5)
y_interp[:, :, 1] = interp_g((xg, yg))

interp_b = interpolate.RegularGridInterpolator((x_b, y_b), data_b, bounds_error=False, fill_value=0.5)
y_interp[:, :, 2] = interp_b((xg, yg))

W_HR = y.copy()
W_HR[1::2, ::2] = 0
W_HR[::2, 1::2] = 0
ker = np.array([[0, 1, 0], [1, 4, 1], [0, 1, 0]]) / 4
W_HR = ndimage.convolve(W_HR, ker)

Y_LF_HR = np.mean(y_interp, axis=2)

res = y_interp + (W_HR - Y_LF_HR)[:, :, np.newaxis]

np.clip(res, 0, 1, res)

print(np.sum((x - res)**2) / 512**2)


fig, axs = plt.subplots(1, 3, sharex=True, sharey=True)
axs[0].imshow(x)
axs[1].imshow(y, cmap='gray')
axs[2].imshow(res)
plt.show()