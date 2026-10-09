# Exp. 1: efecto de epsilon en TV (mapas c_TV, resultado, planas/bordes/textura)

# lam = 0.5 * 1/(4*cmax) = eps/8 para cada eps (mitad de la cota de estabilidad).

# Detención en la iteración n* que minimiza el RMSE (n_max = N_MAX).


import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


from implementacion import (agregar_ruido_gaussiano, rmse, grad_mag,
                      coef_tv, lambda_estable, difusion_anisotropica, SIGMA_RUIDO, SEMILLA)

EPSILONS = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
FACTOR_LAM = 0.5          # lam = FACTOR_LAM * 1/(4*cmax)
N_MAX = 300
ROIS = {}

def cargar_imagen(ruta=None):
    if ruta is None:
        from skimage import data
        img = data.camera()
    else:
        import imageio.v3 as iio
        img = iio.imread(ruta)
        if img.ndim == 3:
            img = img[..., :3].mean(axis=2)
    img = img.astype(np.float64)
    if img.max() > 1.0:
        return img / 255.0
    else:
        return img

def sl(r):
    return (slice(r[0], r[1]), slice(r[2], r[3]))

