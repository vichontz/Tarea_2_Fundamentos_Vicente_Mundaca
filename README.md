 Tarea 2: IEE2714 - Fundamentos de Procesamiento de Imágenes

**Autor:** Vicente Agustín Mundaca Candia

Este repositorio contiene la implementación y los scripts de experimentación de la Tarea 2 del curso IEE2714. La entrega se divide en dos partes:

1. **Pregunta 1:** filtrado Gaussiano espacialmente adaptativo basado en ruido Poisson.
2. **Pregunta 2:** difusión anisotrópica explícita (variación total y coeficientes de diseño propio).

---

## Requisitos

- Python 3.10 o superior
- `pip`
- Paquetes: `numpy`, `scipy`, `matplotlib`, `scikit-image`

Si no están instalados, se pueden instalar con:

```bash
python -m pip install numpy scipy matplotlib scikit-image
```


---

## Estructura del repositorio

```text
Tarea_2_Fundamentos_Vicente_Mundaca/
├── README.md
├── Pregunta 1/
│   ├── experimentacion.py
│   ├── generacion_datos.py
│   ├── implementacion.py
│   └── figures_p1/
│       ├── rmse_vs_sigma.png
│       ├── comparacion_filtros.png
│       ├── mapa_sigma_adaptativo.png
│       └── interpolacion_mapa_sigma.png
├── Pregunta 2/
│   ├── experimentacion.py
│   ├── experimentacion2.py
│   ├── experimentacion3.py
│   ├── experimentacion4.py
│   ├── implementacion.py
│   ├── imagen1.png
│   ├── imagen2.png
│   └── figures_p2/
│       ├── p2_exp1_resultados.png
│       ├── p2_exp2_inadecuados.png
│       ├── p2_exp2t_inadecuados_rmse.png
│       ├── p2_exp3_V1.png
│       ├── p2_exp3_V2.png
│       └── p2_exp4_comparativa_global.png
└── figures/
```

> Si `Pregunta 2/imagen2.png` no existe, el script usa la imagen `camera` de `skimage.data`.

---

## Instrucciones de ejecución

Ejecuta todos los scripts desde la raíz del repositorio, por ejemplo:

```bash
cd "C:\Users\vicen\OneDrive - Universidad Católica de Chile\UNI\6to semestre\Imagenes\area 2\Tarea_2_Fundamentos_Vicente_Mundaca"
```

### Pregunta 1: filtro Gaussiano adaptativo

```bash
python ".\Pregunta 1\experimentacion.py"
```

Cuando se ejecute, el programa pedirá elegir una opción:

- `1` -> barrido de sigma y comparación de RMSE por región.
- `2` -> comparación entre filtro global y filtro adaptativo.

Los resultados gráficos se guardan en la carpeta:

```text
Pregunta 1/figures_p1/
```

### Pregunta 2: difusión anisotrópica

Ejecuta la experimentación principal:

```bash
python ".\Pregunta 2\experimentacion.py"
```

Luego, para reproducir los estudios complementarios:

```bash
python ".\Pregunta 2\experimentacion2.py"
python ".\Pregunta 2\experimentacion3.py"
python ".\Pregunta 2\experimentacion4.py"
```

Los gráficos generados se almacenan en:

```text
Pregunta 2/figures_p2/
```

---

## Reproducción de resultados

Para reproducir los resultados esperados del proyecto se recomienda ejecutar los scripts en este orden:

1. `python ".\Pregunta 1\experimentacion.py"`
2. `python ".\Pregunta 2\experimentacion.py"`
3. `python ".\Pregunta 2\experimentacion2.py"`
4. `python ".\Pregunta 2\experimentacion3.py"`
5. `python ".\Pregunta 2\experimentacion4.py"`

Con esto se regeneran las figuras principales y los análisis comparativos del trabajo:

- RMSE vs. sigma para el filtro Gaussiano de la Pregunta 1.
- Comparación visual entre filtro global y adaptativo.
- Mapa de `sigma(x, y)` y curva de interpolación intensidad-sigma.
- Estudio de estabilidad y sobre-suavizado de la difusión anisotrópica.
- Exploración de los diseños alternativos V1 y V2.
- Comparativa global final de configuraciones.

---

## Observación

Para la pregunta 1 en experimentacion se pueden descomentar 4 semillas, las cuales fueron utilizadas para verificar que los resultados fueran generalizables para toda semilla.


Para la pregunta 2/experimentación.py en caso de querer aplicar el proceso a la imagen de cordillera descomentar la linea 21 y comentar la linea 22 y descomentar la linea 55 y comentar la 54.

para las experimentaciones 2 3 y 4 si se quiere cambiar la imagen nuevamente hay que comentar y descomentar las lineas donde se define x.
