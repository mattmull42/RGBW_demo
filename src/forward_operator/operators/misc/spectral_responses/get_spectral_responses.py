import numpy as np
import scipy as sp
from os import path


DATA_DIR = path.join(path.dirname(__file__), 'data')


def get_filter_response(spectral_stencil, responses_file, band):
    if responses_file == 'dirac':
        return get_dirac_filter(spectral_stencil, band)

    else:
        array = np.load(path.join(DATA_DIR, responses_file))

        f = sp.interpolate.interp1d(array['spectral_stencil'], array['data'][band])

        return f(spectral_stencil)


def get_dirac_filter(spectral_stencil, filter_type):
    stencil = np.array(spectral_stencil)

    if filter_type == 'red':
        return np.eye(1, stencil.shape[0], np.abs(stencil - 650).argmin()).reshape(-1)

    elif filter_type == 'green':
        return np.eye(1, stencil.shape[0], np.abs(stencil - 525).argmin()).reshape(-1)

    elif filter_type == 'blue':
        return np.eye(1, stencil.shape[0], np.abs(stencil - 480).argmin()).reshape(-1)

    elif filter_type == 'pan':
        return np.full_like(spectral_stencil, 1 /len(spectral_stencil), dtype=np.float64)