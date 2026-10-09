import os
import numpy as np
import matplotlib.pyplot as plt

# Asegúrate de que coef_v1 y coef_v2 estén en implementacion.py
from implementacion import (agregar_ruido_gaussiano, rmse, grad_mag,
                            difusion_anisotropica, SIGMA_RUIDO, SEMILLA,
                            coef_v1, coef_v2)
from experimentacion import cargar_imagen, sl

if __name__ == '__main__':
    ruta = "Pregunta 2/figures_p2"
    os.makedirs(ruta, exist_ok=True)
    
    # 1. Preparación de imagen: se toma la imagen taza y cámara por que cordillera es muy grande y los tiempos de ejecución son muy largos.
    #x = cargar_imagen('Pregunta 2/imagen1.png') # Taza
    x = cargar_imagen()                         # Cámara (skimage)
    #x = cargar_imagen('Pregunta 2/imagen2.png') # Cordillera
    y = agregar_ruido_gaussiano(x)
    gx_, gy_ = grad_mag(x), grad_mag(y)



    # Parámetros generales para la difusión
    N_ITER = 100
    LAMBDA_FIJO = 0.2  # Seguro porque para V1 y V2, c_max = 1.0 -> lam <= 0.25

    print("Iniciando Experimento 3: Exploración de Propuestas V1 y V2...")

    # =====================================================================
    # 1. EXPLORACIÓN PROPUESTA V1 (Exponencial de potencia 4)
    # Parámetro a explorar: k (umbral de gradiente)
    # =====================================================================
    print("\n--- Explorando V1 (Exponencial potencia 4) ---")
    K_VALS_V1 = [0.01, 0.03, 0.08, 0.2]
    
    fig_v1, ax_v1 = plt.subplots(2, 4, figsize=(18, 9))
    fig_v1.suptitle("Propuesta V1: $c(u) = \exp(-(|\\nabla u| / k)^4)$", fontsize=16)
    
    for i, k in enumerate(K_VALS_V1):
        c_func = coef_v1(k)
        
        # Mapa C inicial (evaluado sobre la imagen ruidosa y)
        mapa_c = c_func(y)
        
        # Filtrado final
        u_final, info = difusion_anisotropica(y, c_func, LAMBDA_FIJO, N_ITER, referencia=x)
        
        print(f"V1 (k={k:g}): RMSE={info['rmse'][-1]:.4f}")
        
        # Graficar Mapa C
        im0 = ax_v1[0, i].imshow(mapa_c, cmap='magma', vmin=0, vmax=1)
        ax_v1[0, i].set_title(f"Mapa c (t=0) | k={k:g}\nMedia $c$={mapa_c.mean():.2f}")
        ax_v1[0, i].axis('off')
        
        # Graficar Resultado
        ax_v1[1, i].imshow(u_final, cmap='gray', vmin=0, vmax=1)
        ax_v1[1, i].set_title(f"Resultado (100 iter)\nRMSE={info['rmse'][-1]:.4f}")
        ax_v1[1, i].axis('off')
        
    fig_v1.colorbar(im0, ax=ax_v1[0, :].ravel().tolist(), label='Conductividad $c$ (1=Difunde, 0=Protege)')
    plt.savefig('Pregunta 2/figures_p2/p2_exp3_V1.png', dpi=130, bbox_inches='tight')
    plt.close()

    # =====================================================================
    # 2. EXPLORACIÓN PROPUESTA V2 (Racional con Laplaciano)
    # Parámetros a explorar: k (umbral) y sigma (suavizado previo)
    # =====================================================================
    print("\n--- Explorando V2 (Laplaciano pre-suavizado) ---")
    
    # Vamos a fijar un k razonable y variar sigma (que controla qué tanto mitigamos el ruido)
    K_FIJO = 0.05
    SIGMA_VALS = [0.1, 0.5, 1.5, 3.0]
    
    fig_v2, ax_v2 = plt.subplots(2, 4, figsize=(18, 9))
    fig_v2.suptitle(f"Propuesta V2: Laplaciano pre-suavizado (k={K_FIJO})", fontsize=16)
    
    for i, sig in enumerate(SIGMA_VALS):
        c_func = coef_v2(k=K_FIJO, sigma=sig)
        
        # Mapa C inicial
        mapa_c = c_func(y)
        
        # Filtrado final
        u_final, info = difusion_anisotropica(y, c_func, LAMBDA_FIJO, N_ITER, referencia=x)
        
        print(f"V2 (k={K_FIJO}, sigma={sig:g}): RMSE={info['rmse'][-1]:.4f}")
        
        # Graficar Mapa C
        im1 = ax_v2[0, i].imshow(mapa_c, cmap='magma', vmin=0, vmax=1)
        ax_v2[0, i].set_title(f"Mapa c (t=0) | $\sigma$={sig:g}\nMedia $c$={mapa_c.mean():.2f}")
        ax_v2[0, i].axis('off')
        
        # Graficar Resultado
        ax_v2[1, i].imshow(u_final, cmap='gray', vmin=0, vmax=1)
        ax_v2[1, i].set_title(f"Resultado (100 iter)\nRMSE={info['rmse'][-1]:.4f}")
        ax_v2[1, i].axis('off')
        
    fig_v2.colorbar(im1, ax=ax_v2[0, :].ravel().tolist(), label='Conductividad $c$ (1=Difunde, 0=Protege)')
    plt.savefig('Pregunta 2/figures_p2/p2_exp3_V2.png', dpi=130, bbox_inches='tight')
    plt.close()
    
    print("\n¡Gráficos guardados en 'Pregunta 2/figures_p2/'!")