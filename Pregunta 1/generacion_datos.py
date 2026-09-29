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

