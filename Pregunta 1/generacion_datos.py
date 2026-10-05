from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


# ----------- Datos solicitados ---------- #
T_imagen    =   256
L_cuadrado  =   128
R_circulo   =   32
I_fondo, I_cuadrado, I_circulo = 0.15, 0.45, 0.80
# ---------------------------------------- #


def imagen_ideal(tam=256, i_fondo=0.15, i_cuadrado=0.45, i_circulo=0.80,
                       l_cuadrado=128, r_circulo=32):

    imagen = np.full((tam, tam), i_fondo, dtype=np.float32)

    # Cuadrado centrado 
    ini = (tam - l_cuadrado) // 2       #   inicio  = mitad imagen - medio lado
    fin = ini + l_cuadrado              #   fin     = inicio + lado
    cuadrado = np.zeros((tam, tam), dtype=bool)
    cuadrado[ini:fin, ini:fin] = True

    # Ya que T_imagen es par, el centro real esta entre los pixeles 127 y 128 -> (size-1)/2 = 127.5,
    c = (tam - 1) / 2
    i, j = np.indices((tam, tam))
    circulo = (i - c) ** 2 + (j - c) ** 2 <= r_circulo ** 2

    imagen[cuadrado] = i_cuadrado
    imagen[circulo] = i_circulo

    mascaras = {
        'Fondo':    ~cuadrado,
        'Cuadrado': cuadrado & ~circulo,   # cuadrado excluyendo el círculo
        'Circulo':  circulo,
    }

    return imagen, mascaras

def agregar_poisson(imagen, N):
    imagen_ruidosa = np.random.poisson(imagen * N) / N
    return imagen_ruidosa.astype(np.float32)

def rmse(a, b, mascara=None): #obtenido de las capsulas 
    d = (a.astype(np.float64) - b.astype(np.float64))
    if mascara is not None:
        d = d[mascara]
    return float(np.sqrt(np.mean(d**2)))



if __name__ == '__main__':
    
    SEED = 1234
    np.random.seed(SEED)    # semilla fija

    N = 40             
    x, masks = imagen_ideal()
    y = agregar_poisson(x, N)


   # directorio de salida
    output_dir = Path(__file__).resolve().parent / "figures_p1"
    output_dir.mkdir(exist_ok=True)

    
    # Estadísticas por región
    valores = {'Fondo': 0.15, 'Cuadrado': 0.45, 'Circulo': 0.80}
    print(f'N={N}, semilla={SEED}')
    print(f"{'Región':10s} {'#píxeles':>9s} {'x':>5s} {'media(y)':>9s} {'RMSE':>8s} {'sqrt(x/N)':>10s}")
    for nombre, m in masks.items():
        print(f'{nombre:10s} {m.sum():9d} {valores[nombre]:5.2f} {y[m].mean():9.4f} '
              f'{rmse(y, x, m):8.4f} {np.sqrt(valores[nombre]/N):10.4f}')
    print(f'RMSE global (sin filtrar): {rmse(y, x):.4f}')


    # Gráfico 1: Imagen Ideal vs Ruidosa
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    ax[0].imshow(x, cmap="gray", vmin=0, vmax=1)
    ax[0].set_title("Imagen ideal x")
    ax[0].axis("off")

    ax[1].imshow(y, cmap="gray", vmin=0, vmax=1)
    ax[1].set_title(f"Ruidosa y = Poisson(N·x)/N, N={N}")
    ax[1].axis("off")

    fig.tight_layout()

    # Gráfico 2: Máscaras de las regiones
    fig_masks, ax_masks = plt.subplots(1, 3, figsize=(12, 4))
    nombres_mascaras = ["fondo", "cuadrado", "circulo"]

    for axis, name in zip(ax_masks, nombres_mascaras):
        axis.imshow(masks[name], cmap="gray", vmin=0, vmax=1)
        axis.set_title(f"Máscara: {name}")
        axis.axis("off")

    fig_masks.tight_layout()


    fig.savefig(output_dir / "1.1_ideal_y_ruidosa.png", dpi=200, bbox_inches="tight")
    fig_masks.savefig(output_dir / "1.1_mascaras.png", dpi=200, bbox_inches="tight")
    
    plt.show()




