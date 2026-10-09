
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


# ---------------------------------------------------------------- estabilidad
def lambda_estable(cmax):
    
    return 1.0 / (4.0 * cmax)
 
 
# ---------------------------------------------------------------- un paso y bucle
def paso_difusion(u, c, lam):

    up = np.pad(u, 1, mode='edge')
    cp = np.pad(c, 1, mode='edge')
    U, C = up[1:-1, 1:-1], cp[1:-1, 1:-1]
    flujo = np.zeros_like(u)
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        Un = up[1 + dy:1 + dy + u.shape[0], 1 + dx:1 + dx + u.shape[1]]
        Cn = cp[1 + dy:1 + dy + u.shape[0], 1 + dx:1 + dx + u.shape[1]]
        flujo += 0.5 * (C + Cn) * (Un - U)
    return u + lam * flujo
 
 
def difusion_anisotropica(u0, coef, lam, n_iter, referencia=None, guardar=()):
    """Difusión general.
    coef: función u -> mapa c (misma forma que u); se evalúa sobre u^t en toda iteración.
    lam: paso temporal.   
    n_iter: número de iteraciones.
    referencia: imagen para registrar RMSE(t).
    guardar: iteraciones cuyo estado se guarda.
    Retorna (u_final, info). Si lam excede la cota 1/(4*cmax_observado) se reporta
    en info['viola_estabilidad'] para estudiar la instabilidad.
    """
    u = u0.astype(np.float64).copy()
    info = {'cmax': [], 'rmse': [], 'snap': {}, 'viola_estabilidad': False, 'diverge_en': None}
    if referencia is not None:
        info['rmse'].append(rmse(referencia, u))
    for t in range(1, n_iter + 1):
        c = coef(u)                                  # c(x,y,t) a partir de u^t
        cmax_t = float(np.max(c))
        info['cmax'].append(cmax_t)
        if lam * 4.0 * cmax_t > 1.0 + 1e-12:
            info['viola_estabilidad'] = True
        u = paso_difusion(u, c, lam)
        if not np.all(np.isfinite(u)) or np.abs(u).max() > 1e6:
            info['diverge_en'] = t
            break
        if referencia is not None:
            info['rmse'].append(rmse(referencia, u))
        if t in guardar:
            info['snap'][t] = u.copy()
    return u, info

 #los 2 coeficientes pedidos:

 # 1) coeficiente con exponente 4, difunde mas en las zonas planas y menos en los bordes, es mas agresivo que el coeficiente con exponente 2 
def coef_v1(k):
    if k <= 0:
        raise ValueError("k debe ser > 0")
    
    def c(u):
        # h(E) = exp(-(E/k)^4) donde E = |grad u|
        return np.exp(-(grad_mag(u) / k) ** 4)
        
    c.cmax = 1.0  # max cuando gradiente = 0 (e^0 = 1)
    c.nombre = f"V1_exp4(k={k:g})"
    return c