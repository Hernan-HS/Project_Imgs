from extractores.hog_extractor import extraer_hog
from utils.image_loader import cargar_imagen
from utils.io_utils import guardar_vector
from utils.evaluacion import calcular_rank_normalizado
from pathlib import Path
import numpy as np
import os
import matplotlib.pyplot as plt

dataset_path = Path("datasets/CBIR_50")
salida_base = Path("vectores/hog_experimentos")
medida = "coseno"

# Definir combinaciones de parámetros a probar
param_combos = [
    {"orientations": 8, "pixels_per_cell": (8, 8), "cells_per_block": (2, 2)},
    {"orientations": 9, "pixels_per_cell": (8, 8), "cells_per_block": (2, 2)},
    {"orientations": 9, "pixels_per_cell": (16, 16), "cells_per_block": (2, 2)},
    {"orientations": 12, "pixels_per_cell": (8, 8), "cells_per_block": (2, 2)},
]

resultados = []

for config in param_combos:
    nombre_config = f"o{config['orientations']}_ppc{config['pixels_per_cell'][0]}_cpb{config['cells_per_block'][0]}"
    carpeta_salida = salida_base / nombre_config
    print(f"\nEvaluando configuración: {nombre_config}")

    # === Paso 1: Extraer vectores y guardarlos ===
    for ruta_img in dataset_path.rglob("*.jpg"):
        clase = ruta_img.parent.name
        nombre = ruta_img.stem
        img = cargar_imagen(ruta_img)
        if img is None:
            continue
        vec = extraer_hog(np.array(img), parametros=config)
        guardar_vector(vec, nombre, carpeta_salida / clase)

    # === Paso 2: Calcular Rank Normalizado promedio ===
    nr_list = []
    for i, clase_folder in enumerate(sorted(dataset_path.iterdir())):
        if not clase_folder.is_dir():
            continue
        ruta_consulta = list(clase_folder.glob("*.jpg"))[0]
        nombre_consulta = f"{clase_folder.name}/{ruta_consulta.stem}"
        img = cargar_imagen(ruta_consulta)
        vec = extraer_hog(np.array(img), parametros=config)
        nr = calcular_rank_normalizado(nombre_consulta, vec, carpeta_salida, medida)
        nr_list.append(nr)
        if i == 9:  # Solo 10 clases
            break

    promedio = sum(nr_list) / len(nr_list)
    print(f"→ Rank Normalizado promedio: {promedio:.4f}")
    resultados.append((nombre_config, promedio))

# === Graficar resultados ===
etiquetas = [r[0] for r in resultados]
valores = [r[1] for r in resultados]

plt.figure(figsize=(10, 5))
plt.bar(etiquetas, valores, color="skyblue")
plt.ylabel("Rank Normalizado Promedio")
plt.xlabel("Configuración HOG")
plt.title("Análisis de parámetros HOG")
plt.xticks(rotation=45)
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()

