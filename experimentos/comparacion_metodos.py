import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from extractores.hog_extractor import extraer_hog
from extractores.cnn_extractor import extraer_resnet
from utils.image_loader import cargar_imagen
from utils.evaluation import calcular_rank_normalizado
from utils.similarity import comparar_con_base

# === Configuración ===
dataset_path = Path("datasets/CBIR_50")
carpeta_vectores_hog = Path("vectores/hog/o9_ppc8_cpb2")  # Mejor configuración previa
carpeta_vectores_cnn = Path("vectores/resnet")
medida = "coseno"

# === Evaluar ambos métodos en las mismas 10 clases ===
rank_hog = []
rank_cnn = []
tiempos_hog = []
tiempos_cnn = []

for i, carpeta in enumerate(sorted(dataset_path.iterdir())):
    if not carpeta.is_dir():
        continue

    img_path = list(carpeta.glob("*.jpg"))[0]
    nombre = f"{carpeta.name}/{img_path.stem}"
    img = cargar_imagen(img_path)

    # --- HOG
    t0 = time.time()
    vec_hog = extraer_hog(np.array(img))
    nr_hog = calcular_rank_normalizado(nombre, vec_hog, carpeta_vectores_hog, medida)
    t1 = time.time()
    rank_hog.append(nr_hog)
    tiempos_hog.append(t1 - t0)

    # --- CNN (ResNet)
    t0 = time.time()
    vec_cnn = extraer_resnet(img)
    nr_cnn = calcular_rank_normalizado(nombre, vec_cnn, carpeta_vectores_cnn, medida)
    t1 = time.time()
    rank_cnn.append(nr_cnn)
    tiempos_cnn.append(t1 - t0)

    print(f"{nombre} | HOG NR={nr_hog:.4f} | CNN NR={nr_cnn:.4f}")

    if i == 9:  # Solo 10 clases
        break

# === Resumen ===
prom_hog = np.mean(rank_hog)
prom_cnn = np.mean(rank_cnn)
time_hog = np.mean(tiempos_hog)
time_cnn = np.mean(tiempos_cnn)

print("\n===== Tabla Resumen =====")
print(f"{'Método':<10} | {'Rank N. Prom':<14} | {'Tiempo promedio (s)'}")
print("-" * 42)
print(f"{'HOG':<10} | {prom_hog:<14.4f} | {time_hog:.4f}")
print(f"{'ResNet':<10} | {prom_cnn:<14.4f} | {time_cnn:.4f}")



from utils.evaluation import curva_precision_recall
import matplotlib.pyplot as plt

recalls, precisiones = curva_precision_recall(
    vector_consulta=vec_cnn,
    clase_real="manzana",
    carpeta_vectores="vectores/resnet",
    medida="coseno"
)

plt.plot(recalls, precisiones, marker='o')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Curva Precision–Recall (CNN - manzana)")
plt.grid(True)
plt.show()



from experimentos.visualizacion_caracteristicas import cargar_vectores_y_etiquetas, visualizar_2D

# HOG
X_hog, y_hog = cargar_vectores_y_etiquetas("vectores/hog/o9_ppc8_cpb2", n_clases=10, n_por_clase=10)
visualizar_2D(X_hog, y_hog, metodo="pca", titulo="HOG")
visualizar_2D(X_hog, y_hog, metodo="tsne", titulo="HOG")

# ResNet
X_cnn, y_cnn = cargar_vectores_y_etiquetas("vectores/resnet", n_clases=10, n_por_clase=10)
visualizar_2D(X_cnn, y_cnn, metodo="pca", titulo="ResNet")
visualizar_2D(X_cnn, y_cnn, metodo="tsne", titulo="ResNet")



from pathlib import Path
from experimentos.robustez_transformaciones import evaluar_robustez
from extractores.cnn_extractor import extraer_resnet

dataset = Path("datasets/CBIR_50")
vector_dir = "vectores/resnet"
imagenes = list(dataset.rglob("*.jpg"))

originales = []
rotadas = []
ruidosas = []
deformadas = []

for img_path in random.sample(imagenes, 10):
    orig, rot = evaluar_robustez(img_path, vector_dir, extraer_resnet, "rotar", "Rotación")
    _, ruido = evaluar_robustez(img_path, vector_dir, extraer_resnet, "ruido", "Ruido Gaussiano")
    _, defo = evaluar_robustez(img_path, vector_dir, extraer_resnet, "deformar", "Deformación Elástica")

    originales.append(orig)
    rotadas.append(rot)
    ruidosas.append(ruido)
    deformadas.append(defo)

# Graficar
import matplotlib.pyplot as plt
labels = ["Original", "Rotada", "Ruido", "Deformada"]
promedios = [np.mean(originales), np.mean(rotadas), np.mean(ruidosas), np.mean(deformadas)]

plt.bar(labels, promedios, color=["green", "orange", "red", "blue"])
plt.ylabel("Rank Normalizado Promedio")
plt.title("Impacto de Transformaciones (CNN)")
plt.grid(True)
plt.show()

