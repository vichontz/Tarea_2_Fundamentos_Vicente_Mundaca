import os
import numpy as np
import matplotlib.pyplot as plt
from implementacion import (agregar_ruido_gaussiano, grad_mag, rmse, coef_tv, 
                            difusion_anisotropica, lambda_estable, SIGMA_RUIDO, SEMILLA)
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

    
    resultados = {}
    
    # ESCENARIO A: Evolución demasiado lenta
    # Causa: Un eps minúsculo obliga a un lam minúsculo. Pocas iteraciones no hacen nada.
    
    eps_lento = 0.0001
    lam_lento = lambda_estable(1.0 / eps_lento) # Queda en 0.000025
    iter_lento = 100
    
    print(f"-> A) Ejecutando evolución lenta (eps={eps_lento}, lam={lam_lento:.6f}, iter={iter_lento})")
    u_lento, info_lento = difusion_anisotropica(y, coef_tv(eps_lento), lam_lento, iter_lento, referencia=x)
    resultados['Lento'] = (u_lento, f"Evolución muy lenta\n$\epsilon={eps_lento}$, $\lambda={lam_lento:.6f}$\nIter: {iter_lento}\nRMSE={info_lento['rmse'][-1]:.4f}")

    
    # ESCENARIO B: Sobre-suavizado
    # Causa: Un eps muy grande convierte a TV casi en un filtro Gaussiano global.
    # Con muchas iteraciones, destruye la imagen.
    
    eps_sobre = 2.0
    lam_sobre = 0.2  # Seguro, ya que c_max es aprox 0.5 (1/2)
    iter_sobre = 400
    
    print(f"-> B) Ejecutando sobre-suavizado (eps={eps_sobre}, lam={lam_sobre}, iter={iter_sobre})")
    u_sobre, info_sobre = difusion_anisotropica(y, coef_tv(eps_sobre), lam_sobre, iter_sobre, referencia=x)
    resultados['Sobresuavizado'] = (u_sobre, f"Sobre-suavizado\n$\epsilon={eps_sobre}$, $\lambda={lam_sobre}$\nIter: {iter_sobre}\nRMSE={info_sobre['rmse'][-1]:.4f}")

    
    # ESCENARIO C: Problemas Numéricos (Inestabilidad / Explosión)
    # Causa: Violar la condición de estabilidad CFL.
    # lam > 1 / (4 * c_max).
    
    eps_inestable = 0.01
    cmax_teorico = 1.0 / eps_inestable # 100
    lam_limite = 1.0 / (4.0 * cmax_teorico) # 0.0025
    
    lam_inestable = lam_limite * 1.5 # FORZAMOS LA RUPTURA DEL LÍMITE (0.00375)
    iter_inestable = 150
    
    print(f"-> C) Ejecutando inestabilidad (eps={eps_inestable}, lam={lam_inestable} [Límite era {lam_limite}], iter={iter_inestable})")
    u_inestable, info_inestable = difusion_anisotropica(y, coef_tv(eps_inestable), lam_inestable, iter_inestable, referencia=x)
    
    # Manejar si la imagen tiene NaNs o valores gigantes por la explosión numérica
    if info_inestable['diverge_en'] is not None:
        msj_diverge = f"¡DIVERGIÓ en iter {info_inestable['diverge_en']}!"
        # Recuperar la última captura guardada sana si existe, o usar clip
        u_inestable = np.clip(u_inestable, 0, 1) 
    else:
        msj_diverge = "Inestable pero no colapsó"
        
    resultados['Inestable'] = (u_inestable, f"Inestabilidad numérica\n$\epsilon={eps_inestable}$, $\lambda={lam_inestable}$\n{msj_diverge}")

    
    # VISUALIZACIÓN
    
    fig, ax = plt.subplots(1, 4, figsize=(20, 5.5))
    
    # 1. Ruidosa (Referencia)
    ax[0].imshow(y, cmap='gray', vmin=0, vmax=1)
    ax[0].set_title(f"Imagen Ruidosa Original\nRMSE={rmse(x,y):.4f}")
    ax[0].axis('off')
    
    # 2. Lenta
    ax[1].imshow(resultados['Lento'][0], cmap='gray', vmin=0, vmax=1)
    ax[1].set_title(resultados['Lento'][1])
    ax[1].axis('off')
    
    # 3. Sobre-suavizada
    ax[2].imshow(resultados['Sobresuavizado'][0], cmap='gray', vmin=0, vmax=1)
    ax[2].set_title(resultados['Sobresuavizado'][1])
    ax[2].axis('off')
    
    # 4. Inestable
    # Si la imagen explotó, mostrar el "ruido de sal y pimienta" extremo truncado
    ax[3].imshow(resultados['Inestable'][0], cmap='gray', vmin=0, vmax=1)
    ax[3].set_title(resultados['Inestable'][1], color='red')
    ax[3].axis('off')
    
    plt.tight_layout()
    plt.savefig('Pregunta 2/figures_p2/p2_exp2_inadecuados.png', dpi=130)
    plt.show()

    # ---- Gráfico extra: Comportamiento del RMSE en la explosión ----
    fig_err, ax_err = plt.subplots(figsize=(7, 4))
    
    rmse_inestable = info_inestable['rmse']
    ax_err.plot(rmse_inestable, color='red', label=f'Inestable (λ={lam_inestable})')
    ax_err.plot(info_lento['rmse'], color='blue', label='Evolución lenta')
    
    ax_err.axhline(rmse(x,y), color='gray', linestyle='--', label='RMSE Inicial (ruidosa)')
    ax_err.set_title("RMSE a través de las iteraciones")
    ax_err.set_xlabel("Iteraciones")
    ax_err.set_ylabel("RMSE")
    ax_err.set_ylim(0, max(rmse(x,y) * 2, np.nanmax(rmse_inestable[:50]))) # Acotamos el eje Y porque la inestable se va a infinito
    ax_err.grid(alpha=0.3)
    ax_err.legend()
    
    plt.tight_layout()
    plt.savefig('Pregunta 2/figures_p2/p2_exp2_inadecuados_rmse.png', dpi=130)
    plt.show()