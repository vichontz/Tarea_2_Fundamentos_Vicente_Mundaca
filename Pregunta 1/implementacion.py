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


