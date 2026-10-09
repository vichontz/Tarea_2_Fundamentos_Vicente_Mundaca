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

r_cordillera = sl([1230, 1480, 550, 1070])

if __name__ == '__main__':

    os.makedirs('Pregunta 2/figures_p2', exist_ok=True)
    #x = cargar_imagen('Pregunta 2/imagen2.png') # Taza
    #x = cargar_imagen()                         # Cámara (skimage)
    x = cargar_imagen('Pregunta 2/imagen2.png') # Cordillera
    y = agregar_ruido_gaussiano(x)
    gx_, gy_ = grad_mag(x), grad_mag(y)
    res = {}

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    ax[0].imshow(x, cmap="gray", vmin=0, vmax=1)
    ax[0].set_title("Imagen ideal ")
    ax[0].axis("off")
    

    ax[1].imshow(y, cmap="gray", vmin=0, vmax=1)
    ax[1].set_title("Ruidosa")
    ax[1].axis("off")
    

    fig_slice, ax_slice = plt.subplots(1, 2, figsize=(12, 5))
    ax_slice[0].imshow(x[r_cordillera], cmap="gray", vmin=0, vmax=1)
    ax_slice[0].set_title("Recorte: imagen ideal")
    ax_slice[0].axis("off")

    ax_slice[1].imshow(y[r_cordillera], cmap="gray", vmin=0, vmax=1)
    ax_slice[1].set_title("Recorte: imagen ruidosa")
    ax_slice[1].axis("off")

    fig.tight_layout()
    fig_slice.tight_layout()
    fig_slice.savefig(os.path.join("Pregunta 2/figures_p2", "comparacion_recorte.png"),
                      dpi=200, bbox_inches="tight")
    
    plt.show()


    