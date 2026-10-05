
import numpy as np

SEMILLA = 7
SIGMA_RUIDO = 0.05


#

def agregar_ruido_gaussiano(imagen, sigma=SIGMA_RUIDO, semilla=SEMILLA): #copiado del colab
    rng = np.random.default_rng(semilla)
    return imagen + rng.normal(0.0, sigma, imagen.shape)


def rmse(ref, est):
    return float(np.sqrt(np.mean((ref.astype(np.float64) - est.astype(np.float64)) ** 2))) #copiado de la P1



# ---------------------------------------------------------------- cantidades locales
def grad_mag(u):
    gy, gx = np.gradient(u)
    return np.hypot(gx, gy)


def laplaciano(u):
    up = np.pad(u, 1, mode='edge')
    return up[:-2, 1:-1] + up[2:, 1:-1] + up[1:-1, :-2] + up[1:-1, 2:] - 4.0 * u


# ---------------------------------------------------------------- coeficientes
def coef_tv(eps):
    if eps <= 0:
        raise ValueError("eps debe ser > 0")
    def c(u):
        return 1.0 / np.sqrt(grad_mag(u) ** 2 + eps ** 2)
    c.cmax = 1.0 / eps          # cota analítica (se alcanza donde grad u = 0)
    c.nombre = f"TV(eps={eps:g})"
    return c


def coef_constante(valor=1.0):
    def c(u):
        return np.full(u.shape, valor, dtype=np.float64)
    c.cmax = float(valor)
    c.nombre = f"const({valor:g})"
    return c
