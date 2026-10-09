# Corresponde al punto 5 de la exp por que lo otro se explico en la 3

import os
import numpy as np
import matplotlib.pyplot as plt

# Asegúrate de importar tus funciones correctamente
from implementacion import (agregar_ruido_gaussiano, rmse, 
                            coef_tv, coef_v1, coef_v2, 
                            difusion_anisotropica, lambda_estable)

from experimentacion import cargar_imagen, sl

if __name__ == '__main__':
    ruta = "Pregunta 2/figures_p2"
    os.makedirs(ruta, exist_ok=True)


    # Preparación de imagen
    #x = cargar_imagen('Pregunta 2/imagen1.png') # Taza
    x = cargar_imagen()                         # Cámara (skimage)
    #x = cargar_imagen('Pregunta 2/imagen2.png') # Cordillera
    y = agregar_ruido_gaussiano(x)


     
    # CONFIGURACIÓN DE LOS MEJORES PARÁMETROS
    
    # TV: Buscamos un buen balance
    eps_tv = 0.01
    lam_tv = 0.9 * lambda_estable(1.0 / eps_tv) # Margen de seguridad
    iter_tv = 50
    
    # V1: Exponencial a la cuarta (El mejor fue k=0.03 según exp3)
    k_v1 = 0.03
    lam_v1 = 0.2
    iter_v1 = 100
    
    # V2: Laplaciano pre-suavizado (El menos malo fue k=0.05, sigma=1.5)
    k_v2 = 0.05
    sig_v2 = 1.5
    lam_v2 = 0.2
    iter_v2 = 100

     
    # 2. EJECUCIÓN DE LOS FILTROS
     
   
    u_tv, info_tv = difusion_anisotropica(y, coef_tv(eps_tv), lam_tv, iter_tv, referencia=x)
    
    u_v1, info_v1 = difusion_anisotropica(y, coef_v1(k_v1), lam_v1, iter_v1, referencia=x)
    
    u_v2, info_v2 = difusion_anisotropica(y, coef_v2(k_v2, sig_v2), lam_v2, iter_v2, referencia=x)

     
    r_corte = sl ( [60, 124, 190, 254] ) # Borde

    r0, r1, c0, c1 = 70, 150, 100, 180 
    recorte = np.s_[r0:r1, c0:c1]
    fila_perfil = 110 # Fila que cruza un borde fuerte
    col_inicio, col_fin = 80, 200

     
    # 4. VISUALIZACIÓN MULTIPANEL
     
    metodos = [
        ("Ruidosa", y, rmse(x,y), ""),
        ("TV", u_tv, info_tv['rmse'][-1], f"$\epsilon={eps_tv}$, iter={iter_tv}"),
        ("V1 (Exp4)", u_v1, info_v1['rmse'][-1], f"$k={k_v1}$, iter={iter_v1}"),
        ("V2 (Lap)", u_v2, info_v2['rmse'][-1], f"$k={k_v2}, \sigma={sig_v2}$, iter={iter_v2}")
    ]

    fig, ax = plt.subplots(3, 4, figsize=(18, 11))
    
    for i, (nombre, img, error, params) in enumerate(metodos):
        # Fila 1: Imagen Completa
        ax[0, i].imshow(img, cmap='gray', vmin=0, vmax=1)
        titulo = f"{nombre}\nRMSE: {error:.4f}"
        if params: titulo += f"\n{params}"
        ax[0, i].set_title(titulo)
        ax[0, i].axis('off')
        
        # Fila 2: Recorte Ampliado
        ax[1, i].imshow(img[recorte], cmap='gray', vmin=0, vmax=1)
        ax[1, i].set_title(f"Recorte (RMSE: {rmse(x[recorte], img[recorte]):.4f})")
        ax[1, i].axis('off')
        
        # Fila 3: Perfil 1D
        ax[2, i].plot(range(col_inicio, col_fin), x[fila_perfil, col_inicio:col_fin], 'k--', label='Ideal', alpha=0.6)
        ax[2, i].plot(range(col_inicio, col_fin), img[fila_perfil, col_inicio:col_fin], 'r-', label='Filtrada')
        ax[2, i].set_title(f"Perfil Fila {fila_perfil}")
        ax[2, i].grid(alpha=0.3)
        if i == 0: ax[2, i].legend()

    plt.tight_layout()
    plt.savefig('Pregunta 2/figures_p2/p2_exp4_comparativa_global.png', dpi=150)
    print("¡Comparativa guardada en 'Pregunta 2/figures_p2/p2_exp4_comparativa_global.png'!")
    plt.show()