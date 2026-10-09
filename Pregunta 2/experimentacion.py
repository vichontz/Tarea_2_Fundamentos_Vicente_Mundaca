# Exp. 1: efecto de epsilon en TV (mapas c_TV, resultado, planas/bordes/textura)

# lam = 0.5 * 1/(4*cmax) = eps/8 para cada eps (mitad de la cota de estabilidad).

# Detención en la iteración n* que minimiza el RMSE (n_max = N_MAX).


import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


from implementacion import (agregar_ruido_gaussiano, rmse, grad_mag,
                      coef_tv, lambda_estable, difusion_anisotropica, SIGMA_RUIDO, SEMILLA)

EPSILONS = [0.005, 0.05, 0.1, 0.5, 1.0]
FACTOR_LAM = 0.5          # lam = FACTOR_LAM * 1/(4*cmax)
N_MAX = 300

#ROIS = {'Borde': (1230,1430, 550, 750), 'Plana': (830, 1030, 330, 530),'Textura': (1180, 1380, 1300, 1500),} 
ROIS = {                  # (fila0, fila1, col0, col1) en la imagen 'camera' 512x512
    'Plana'     :      (10, 74, 10, 74),
    'Borde'     :      (100, 164, 51, 115),
    'Textura'   :    (80, 144, 180, 244),
}
ruta = "Pregunta 2/figures_p2"
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

#r_corte = sl([1230, 1480, 550, 1070])
r_corte = sl ( [60, 124, 190, 254] ) # Borde


if __name__ == '__main__':

    os.makedirs(ruta, exist_ok=True)
    #x = cargar_imagen('Pregunta 2/imagen1.png') # Taza
    x = cargar_imagen()                         # Cámara (skimage)
    #x = cargar_imagen('Pregunta 2/imagen2.png') # Cordillera
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
    ax_slice[0].imshow(x[r_corte], cmap="gray", vmin=0, vmax=1)
    ax_slice[0].set_title("Recorte: imagen ideal")
    ax_slice[0].axis("off")

    ax_slice[1].imshow(y[r_corte], cmap="gray", vmin=0, vmax=1)
    ax_slice[1].set_title("Recorte: imagen ruidosa")
    ax_slice[1].axis("off")

    fig_rois, ax_rois = plt.subplots(1, len(ROIS), figsize=(5 * len(ROIS), 5),
                                     squeeze=False)
    for axis, (nombre, region) in zip(ax_rois.ravel(), ROIS.items()):
        ideal_roi = x[sl(region)]
        ruidosa_roi = y[sl(region)]
        comparacion = np.concatenate((ideal_roi, ruidosa_roi), axis=1)

        axis.imshow(comparacion, cmap="gray", vmin=0, vmax=1)
        axis.axvline(ideal_roi.shape[1] - 0.5, color="red", linewidth=1.5)
        axis.set_title(f"{nombre}\nIdeal | Ruidosa")
        axis.axis("off")

    fig.tight_layout()
    fig_slice.tight_layout()
    fig_rois.tight_layout()
    fig_slice.savefig(os.path.join(ruta, "comparacion_recorte.png"),
                      dpi=200, bbox_inches="tight")
    fig_rois.savefig(os.path.join(ruta, "comparacion_rois.png"),
                     dpi=200, bbox_inches="tight")
    plt.show()

    res = {}
    for eps in EPSILONS:
        lam = FACTOR_LAM * lambda_estable(1 / eps)
        u, info = difusion_anisotropica(y, coef_tv(eps), lam, N_MAX, referencia=x)
        r = np.array(info['rmse']); n_star = int(np.argmin(r))
        u_star, _ = difusion_anisotropica(y, coef_tv(eps), lam, n_star)
        res[eps] = dict(lam=lam, rmse_curva=r, n_star=n_star, rmse_min=float(r.min()), u=u_star)
        print(f'eps={eps:<6g} lam={lam:.6f} n*={n_star:3d}  RMSE*={r.min():.4f}  RMSE@{N_MAX}={r[-1]:.4f}')


    # ---- Tabla por ROI ----
    print('\nDiferencia normalizada c_n = eps*c_TV = 1/sqrt(1+(|grad|/eps)^2) en t=0 (media por ROI):')
    print(f"{'eps':>7s}" + ''.join(f'{k:>22s}' for k in ROIS))
    for eps in EPSILONS:
        cn = eps * coef_tv(eps)(y)
        print(f'{eps:7g}' + ''.join(f'{cn[sl(r)].mean():22.3f}' for r in ROIS.values()))
    print('\n|grad y| mediana por ROI (ruidosa) y |grad x| mediana (limpia):')
    for k, r in ROIS.items():
        print(f'  {k:18s} y: {np.median(gy_[sl(r)]):.4f}   x: {np.median(gx_[sl(r)]):.4f}   p99 x: {np.percentile(gx_[sl(r)],99):.3f}')
    print('\nRMSE por ROI en n* (ruidosa:', {k: round(rmse(x[sl(r)], y[sl(r)]), 4) for k, r in ROIS.items()}, ')')
    print(f"{'eps':>7s}" + ''.join(f'{k:>22s}' for k in ROIS))
    for eps in EPSILONS:
        u = res[eps]['u']
        print(f'{eps:7g}' + ''.join(f'{rmse(x[sl(r)], u[sl(r)]):22.4f}' for r in ROIS.values()))
    print('\nRetención: textura = std(u)/std(x); borde = p99|grad u| / p99|grad x|')
    rt, rb = ROIS['Textura'], ROIS['Borde']
    print(f'  referencia ruidosa: textura {y[sl(rt)].std()/x[sl(rt)].std():.3f}  borde {np.percentile(gy_[sl(rb)],99)/np.percentile(gx_[sl(rb)],99):.3f}')
    for eps in EPSILONS:
        u = res[eps]['u']; gu = grad_mag(u)
        print(f'  eps={eps:<6g} textura {u[sl(rt)].std()/x[sl(rt)].std():.3f}   borde {np.percentile(gu[sl(rb)],99)/np.percentile(gx_[sl(rb)],99):.3f}')


    # Figura 1: curva c_n(g) con la mediana |grad y| de cada ROI
    g = np.logspace(-3, 0, 300)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for eps in EPSILONS: ax.semilogx(g, 1 / np.sqrt(1 + (g / eps) ** 2), label=f'ε={eps:g}')
    roi_colors = plt.get_cmap('tab10').colors
    for i, (k, r) in enumerate(ROIS.items()):
        c = roi_colors[i % len(roi_colors)]
        ax.axvline(np.median(gy_[sl(r)]), color=c, ls='--', alpha=.8); ax.text(np.median(gy_[sl(r)]), 1.02, k.split()[0], color=c, rotation=0, ha='center', fontsize=8)
    ax.set_xlabel('|∇u| (ruidosa, mediana por ROI en líneas)'); ax.set_ylabel('c normalizado = ε·c_TV'); ax.set_ylim(0, 1.1)
    ax.grid(alpha=.3); ax.legend(ncol=2, fontsize=8); ax.set_title('Difusividad relativa vs |∇u| para cada ε')
    plt.tight_layout(); plt.savefig(f'{ruta}/p2_exp1_curvas_c.png', dpi=130); plt.close()


    # Figura 2: mapas c_n sobre la imagen ruidosa (t=0)
    n_map_cols = min(3, len(EPSILONS))
    n_map_rows = int(np.ceil(len(EPSILONS) / n_map_cols))
    fig, ax = plt.subplots(n_map_rows, n_map_cols, figsize=(5 * n_map_cols, 5 * n_map_rows),
                           squeeze=False)
    map_axes = ax.ravel()
    for a, eps in zip(map_axes, EPSILONS):
        cn = eps * coef_tv(eps)(y)
        im = a.imshow(cn, cmap='magma', vmin=0, vmax=1); a.axis('off')
        a.set_title(f'ε·c_TV (ε={eps:g}), c_max=1/ε={1/eps:g}\nmedia={cn.mean():.2f}', fontsize=10)
        for r in ROIS.values():
            a.add_patch(Rectangle((r[2], r[0]), r[3] - r[2], r[1] - r[0], fill=False, ec='cyan', lw=1))
    for a in map_axes[len(EPSILONS):]:
        fig.delaxes(a)
    fig.colorbar(im, ax=ax, fraction=0.02, label='ε·c_TV (1 = difusión máxima)')
    plt.savefig(f'{ruta}/p2_exp1_mapas_c.png', dpi=100, bbox_inches='tight'); plt.close()


    # Figura 3: resultados (en n*)
    items = [(x, 'Original'), (y, f'Ruidosa sigma={SIGMA_RUIDO}, seed={SEMILLA}\nRMSE={rmse(x,y):.4f}')] + \
            [(res[e]['u'], f"ε={e:g}, λ={res[e]['lam']:.5f}, n*={res[e]['n_star']}\nRMSE={res[e]['rmse_min']:.4f}") for e in EPSILONS]
    n_result_cols = min(4, len(items))
    n_result_rows = int(np.ceil(len(items) / n_result_cols))
    fig, ax = plt.subplots(n_result_rows, n_result_cols,
                           figsize=(5 * n_result_cols, 5 * n_result_rows), squeeze=False)
    result_axes = ax.ravel()
    for a, (im, t) in zip(result_axes, items):
        a.imshow(im, cmap='gray', vmin=0, vmax=1); a.set_title(t, fontsize=10); a.axis('off')
    for a in result_axes[len(items):]:
        fig.delaxes(a)
    plt.tight_layout(); plt.savefig(f'{ruta}/p2_exp1_resultados.png', dpi=90); plt.close()


    # Figura 4: recortes por ROI 
    imgs = [x, y] + [res[e]['u'] for e in EPSILONS]
    tit = ['Original', 'Ruidosa'] + [f'ε={e:g}' for e in EPSILONS]
    fig, ax = plt.subplots(len(ROIS), len(imgs),
                           figsize=(3 * len(imgs), 3 * len(ROIS)), squeeze=False)
    for i, (k, r) in enumerate(ROIS.items()):
        for j, (im, t) in enumerate(zip(imgs, tit)):
            ax[i, j].imshow(im[sl(r)], cmap='gray', vmin=x[sl(r)].min()-0.1, vmax=x[sl(r)].max()+0.1); ax[i, j].axis('off')
            ax[i, j].set_title(f'{k.split()[0]}: {t}\nRMSE={rmse(x[sl(r)], im[sl(r)]):.4f}', fontsize=9)
    plt.tight_layout(); plt.savefig(f'{ruta}/p2_exp1_rois.png', dpi=80); plt.close()


    # Figura 5: RMSE vs iteración 
    fig, ax = plt.subplots(figsize=(8, 5))
    for eps in EPSILONS:
        r = res[eps]['rmse_curva']; l, = ax.plot(r, label=f"ε={eps:g}"); ax.plot(res[eps]['n_star'], r.min(), 'o', color=l.get_color())
    ax.axhline(rmse(x, y), color='k', ls=':', label='sin filtrar'); ax.set_xscale('symlog', linthresh=10)
    ax.set_xlabel('iteración'); ax.set_ylabel('RMSE vs original'); ax.grid(alpha=.3); ax.legend(ncol=2, fontsize=8)
    ax.set_title(f'RMSE(t) para TV, λ = ε/8 (mitad de la cota)'); plt.tight_layout(); plt.savefig(f'{ruta}/p2_exp1_rmse_iter.png', dpi=130); plt.close()
