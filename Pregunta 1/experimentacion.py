
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from implementacion import (
    MU_CONTROL,
    SIGMA_CONTROL,
    filtrar_gaussiano,
    filtrar_adaptativo,
    mapa_sigma,
    sigma_por_intensidad,
)
from generacion_datos import imagen_ideal, agregar_poisson, rmse


if __name__ == "__main__":


    #   Definiciones iniciales; sigmas a barrer, semilla, imagen ideal y ruidosa.
    sigmas = np.linspace(0, 5, 100)
    SEED = 1234  #111 #777 #6767 #999
    np.random.seed(SEED) 
    N = 40 
    x, masks = imagen_ideal()
    y = agregar_poisson(x, N)

    #para guardar imagenes
    output_dir = Path(__file__).resolve().parent / "figures_p1"

    print("indique que experimento desea ejcutar: \n 1. Barrido de sigma \n 2. Filtrado adaptativo \n 3. Barrido de sigma_aux")
    opcion = input("Ingrese su elección (1, 2 o 3): ")


    if int(opcion) == 1:
            #   Experimentación inicial, barrido de sigma.
           
        #   Estructura para almacenar resultados
        errores = {
            "global": [],
            "fondo": [],
            "cuadrado": [],
            "circulo": []
        }

        #   Barrido de sigma
        for s in sigmas:
            y_filtrada = filtrar_gaussiano(y, s)
            
            # RMSE Global
            errores["global"].append(rmse(x, y_filtrada))
            
            # RMSE por regiones 
            errores["fondo"].append(rmse(x, y_filtrada, masks["Fondo"]))
            errores["cuadrado"].append(rmse(x, y_filtrada, masks["Cuadrado"]))
            errores["circulo"].append(rmse(x, y_filtrada, masks["Circulo"]))

        #   Identificación de los mínimos por región
        minimos = {}
        for region, lista_errores in errores.items():
            idx_min = np.argmin(lista_errores)
            minimos[region] = {
                "sigma": sigmas[idx_min],
                "rmse": lista_errores[idx_min]
            }

        #   Graficación
        fig, ax = plt.subplots(figsize=(10, 6))

        colores = {"global": "black", "fondo": "blue", "cuadrado": "green", "circulo": "red"}
        estilos = {"global": "--", "fondo": "-", "cuadrado": "-", "circulo": "-"}

        for region in errores.keys():
            ax.plot(sigmas, errores[region], color=colores[region], linestyle=estilos[region], label=f"{region.capitalize()}")
            
            # Marcador y línea vertical para el óptimo
            opt_sigma = minimos[region]["sigma"]
            opt_rmse = minimos[region]["rmse"]
            ax.plot(opt_sigma, opt_rmse, marker="o", color=colores[region])
            ax.axvline(x=opt_sigma, color=colores[region], linestyle=":", alpha=0.5)

        ax.set_title("Desempeño del Filtro Gaussiano: RMSE vs sigma")
        ax.set_xlabel("Parámetro de escala sigma")
        ax.set_ylabel("RMSE")
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
       
        output_dir.mkdir(exist_ok=True)
        fig.savefig(output_dir / "rmse_vs_sigma.png", dpi=200, bbox_inches="tight")
        plt.show()


        print("Valores óptimos encontrados:")
        for region, datos in minimos.items():
            print(f"  {region.capitalize():>8s}: sigma = {datos['sigma']:.2f} | RMSE = {datos['rmse']:.4f}")

    elif int(opcion) == 2:
        
        
        # 1. Obtener el mapa sigma y aplicar el filtro adaptativo
        # Usamos sigma_aux=2.0 como punto de partida para estimar la intensidad local
        mapa, mu_hat = mapa_sigma(y)
        y_adaptativa = filtrar_adaptativo(y, mapa)
        
        # 2. Filtrado global óptimo (asumiendo sigma_global_opt = 1.52 de tu tabla)
        sigma_global_opt = 1.52
        y_global = filtrar_gaussiano(y, sigma=sigma_global_opt, K=4.0)
        
        # 3. Cálculo de RMSE para comparación
        regiones = ["global", "fondo", "cuadrado", "circulo"]
        rmse_global = {
            "global": rmse(x, y_global),
            "fondo": rmse(x, y_global, masks["Fondo"]),
            "cuadrado": rmse(x, y_global, masks["Cuadrado"]),
            "circulo": rmse(x, y_global, masks["Circulo"])
        }
        
        rmse_adaptativo = {
            "global": rmse(x, y_adaptativa),
            "fondo": rmse(x, y_adaptativa, masks["Fondo"]),
            "cuadrado": rmse(x, y_adaptativa, masks["Cuadrado"]),
            "circulo": rmse(x, y_adaptativa, masks["Circulo"])
        }
        
        # 4. Impresión de tabla comparativa
        print(f"{'Región':>10} | {'RMSE Global':>12} | {'RMSE Adaptat.':>12}")
        print("-" * 40)
        for reg in regiones:
            print(f"{reg.capitalize():>10} | {rmse_global[reg]:>12.4f} | {rmse_adaptativo[reg]:>12.4f}")

        # 5. Comparación de imagen original y filtros
        fig_comparacion, ax_comparacion = plt.subplots(1, 3, figsize=(15, 4.5))

        ax_comparacion[0].imshow(y, cmap="gray", vmin=0, vmax=1)
        ax_comparacion[0].set_title("Imagen original (ruidosa)")
        ax_comparacion[0].axis("off")

        ax_comparacion[1].imshow(y_global, cmap="gray", vmin=0, vmax=1)
        ax_comparacion[1].set_title(rf"Filtro global ($\sigma$={sigma_global_opt})")
        ax_comparacion[1].axis("off")

        ax_comparacion[2].imshow(y_adaptativa, cmap="gray", vmin=0, vmax=1)
        ax_comparacion[2].set_title("Filtro adaptativo")
        ax_comparacion[2].axis("off")

        # 6. Visualización del mapa sigma y el resultado adaptativo
        fig_adaptativo, ax_adaptativo = plt.subplots(1, 2, figsize=(10, 4.5))

        im_sigma = ax_adaptativo[0].imshow(mapa, cmap="viridis")
        ax_adaptativo[0].set_title(r"Mapa de $\sigma(x,y)$")
        ax_adaptativo[0].axis("off")
        fig_adaptativo.colorbar(im_sigma, ax=ax_adaptativo[0], shrink=0.7)

        ax_adaptativo[1].imshow(y_adaptativa, cmap="gray", vmin=0, vmax=1)
        ax_adaptativo[1].set_title("Filtro adaptativo")
        ax_adaptativo[1].axis("off")

        # 7. Curva de interpolación intensidad-sigma
        intensidades = np.linspace(0, 1, 501)
        sigmas_interpolados = sigma_por_intensidad(intensidades)
        fig_interp, ax_interp = plt.subplots(figsize=(8, 5))
        ax_interp.plot(intensidades, sigmas_interpolados, color="black")
        ax_interp.scatter(MU_CONTROL, SIGMA_CONTROL, color="red", zorder=3, label="Puntos de control")
        ax_interp.set_title("Interpolación lineal para construir el mapa sigma")
        ax_interp.set_xlabel("Intensidad estimada")
        ax_interp.set_ylabel(r"Sigma asignado ($\sigma$)")
        ax_interp.set_xlim(0, 1)
        ax_interp.grid(alpha=0.3)
        ax_interp.legend()

        output_dir.mkdir(exist_ok=True)
        fig_comparacion.tight_layout()
        fig_adaptativo.tight_layout()
        fig_interp.tight_layout()
        fig_comparacion.savefig(output_dir / "comparacion_filtros.png", dpi=200)
        fig_adaptativo.savefig(output_dir / "mapa_sigma_adaptativo.png", dpi=200)
        fig_interp.savefig(output_dir / "interpolacion_mapa_sigma.png", dpi=200, bbox_inches="tight")
        plt.show()

    

