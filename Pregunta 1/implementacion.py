import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve      # convolución
 
from generacion_datos import imagen_ideal, agregar_poisson, rmse
 
#   1. Kernel Gaussiano isotrópico
#   se trunca en radio r = ceil(K·σ), es decir tamaño (2r+1)x(2r+1).
#   Con K=4 la masa fuera del soporte es approx 6e-5 por eje (K=3 sería ~2.7e-3), despreciable
#   frente al RMSE que se mide (~1e-2). Como se renormaliza, el truncamiento no altera la ganancia DC.
    

def radio_kernel(sigma, K=4.0):  
    return max(1, int(np.ceil(K * sigma)))  #ceil para que sea entero, max(1,) para que no sea menor a un pixel.
 
 
def gaussian_kernel(sigma, K=4.0):
    if sigma == 0:
        return np.ones((1, 1))
    r = radio_kernel(sigma, K)
    x, y = np.mgrid[-r:r+1, -r:r+1]
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel /= kernel.sum()      # normalizamos
    return kernel
 
 
def filtrar_gaussiano(imagen, sigma, K=4.0):
    kernel = gaussian_kernel(sigma, K)
    return convolve(imagen, kernel, mode='reflect')



MU_CONTROL = np.array([0.15, 0.45, 0.80])
SIGMA_CONTROL = np.array([1.72, 1.31, 1.62])


def sigma_por_intensidad(intensidad):
    return np.interp(intensidad, MU_CONTROL, SIGMA_CONTROL)


def mapa_sigma(y_ruidosa, sigma_aux=2.0):

    #estimamos la intesidad con un filtro gaussiano con sigma_aux (lo demas es como la T1)
    mu_hat = filtrar_gaussiano(y_ruidosa, sigma=sigma_aux)

    # np.interp mantiene el sigma del extremo fuera del rango de control.
    mapa_sigma = sigma_por_intensidad(mu_hat)
    return mapa_sigma, mu_hat


def filtrar_adaptativo(y_ruidosa, mapa_sigma, K=4.0):

    filas, columnas = y_ruidosa.shape
    imagen_filtrada = np.zeros_like(y_ruidosa)

    # Encontramos los maximos de la imagen para hacer padding
    sigma_max = np.max(mapa_sigma)
    radio_max = radio_kernel(sigma_max, K)
    
    # Reflejamos en bordes para no perder info
    y_padded = np.pad(y_ruidosa, pad_width=radio_max, mode='reflect')

    for i in range(filas):
        for j in range(columnas):

            # Obtenemos el sigma para cada pixel
            sigma_actual = mapa_sigma[i, j]

            # construimos su kernel y radio
            kernel = gaussian_kernel(sigma_actual, K)
            r = radio_kernel(sigma_actual, K)

            # (hay que desplazar por radio_max debido al padding)
            centro_i = i + radio_max
            centro_j = j + radio_max
            ventana = y_padded[centro_i - r : centro_i + r + 1, 
                               centro_j - r : centro_j + r + 1]
            # Mulplicamos y sumamos para obtener el filtrado
            imagen_filtrada[i, j] = np.sum(ventana * kernel)

    return imagen_filtrada