
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from implementacion import filtrar_gaussiano
from generacion_datos import imagen_ideal, agregar_poisson, rmse


if __name__ == "__main__":
    #   Experimentación inicial, barrido de sigma.
    
    #   Definiciones iniciales; sigmas a barrer, semilla, imagen ideal y ruidosa.
    sigmas = np.linspace(0, 5, 100)
    SEED = 1234
    np.random.seed(SEED) 
    N = 40 
    x, masks = imagen_ideal()
    y = agregar_poisson(x, N)

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
    output_dir = Path(__file__).resolve().parent / "figures_p1"
    output_dir.mkdir(exist_ok=True)
    fig.savefig(output_dir / "rmse_vs_sigma.png", dpi=200, bbox_inches="tight")
    plt.show()


    print("Valores óptimos encontrados:")
    for region, datos in minimos.items():
        print(f"  {region.capitalize():>8s}: sigma = {datos['sigma']:.2f} | RMSE = {datos['rmse']:.4f}")


    