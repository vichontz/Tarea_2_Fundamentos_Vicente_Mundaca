import numpy as np

# ----------- Datos solicitados ---------- #
T_imagen    =   256
L_cuadrado  =   128
R_circulo   =   32
I_fondo, I_cuadrado, I_circulo = 0.15, 0.45, 0.80

N       =   40                          # N en y = Poisson(N*x)/N
SEED    =   1234                    

# ---------------------------------------- #

def imagen_sola(size=T_imagen):

    x = np.full((size, size), I_fondo, dtype=np.float64)

    # Cuadrado centrado 
    ini = (size - L_cuadrado) // 2       #   inicio  = mitad imagen - medio lado 
    fin = ini + L_cuadrado                #   fin     = inicio + lado
    square = np.zeros((size, size), dtype=bool)
    square[ini:fin, ini:fin] = True

    # Ya que T_imagen es par, el centro real esta entre los pixeles 127 y 128 -> (size-1)/2 = 127.5,
    c = (size - 1) / 2
    i, j = np.indices((size, size))
    circle = (i - c) ** 2 + (j - c) ** 2 <= R_circulo ** 2

    x[square] = I_cuadrado
    x[circle] = I_circulo

    masks = {
        "fondo": ~square,
        "cuadrado": square & ~circle,   # cuadrado excluyendo el circulo
        "circulo": circle,
    }
    return x, masks

def añadir_ruido(x, N=N, seed=SEED):
    rng = np.random.default_rng(seed)
    return rng.poisson(N * x).astype(np.float64) / N


 
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from pathlib import Path

    # 1. Configuración inicial y generación de datos
    N = 40  # Asegúrate de definir N si no está como variable global
    x, masks = imagen_sola()
    y = añadir_ruido(x)

    # 2. Configuración del directorio de salida
    output_dir = Path(__file__).resolve().parent / "figures_p1"
    output_dir.mkdir(exist_ok=True)

    # 3. Gráfico 1: Imagen Ideal vs Ruidosa
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    ax[0].imshow(x, cmap="gray", vmin=0, vmax=1)
    ax[0].set_title("Imagen ideal x")
    ax[0].axis("off")

    ax[1].imshow(y, cmap="gray", vmin=0, vmax=1)
    ax[1].set_title(f"Ruidosa y = Poisson(N·x)/N, N={N}")
    ax[1].axis("off")

    fig.tight_layout()

    # 4. Gráfico 2: Máscaras de las regiones
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