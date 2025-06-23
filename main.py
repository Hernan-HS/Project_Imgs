##############################
# Etapa 1: Extracción y almacenamiento de vectores de características.
##############################

from utils.image_loader import cargar_imagen
from utils.io_utils import guardar_vector
from extractores.hog_extractor import extraer_hog
from extractores.cnn_extractor import extraer_resnet

from pathlib import Path
import numpy as np

# === Configuración ===
dataset_path = Path("datasets/CBIR_50")
salida_hog = Path("vectores/hog")
salida_resnet = Path("vectores/resnet")

# === Procesar todas las imágenes en subcarpetas ===
for ruta_imagen in dataset_path.rglob("*.jpg"):
    clase = ruta_imagen.parent.name  # nombre de la carpeta = clase
    nombre_base = ruta_imagen.stem   # sin extensión

    imagen_pil = cargar_imagen(ruta_imagen)
    if imagen_pil is None:
        continue

    print(f"Procesando {clase}/{nombre_base}...")

    # HOG
    imagen_np = np.array(imagen_pil)
    vector_hog = extraer_hog(imagen_np)
    guardar_vector(vector_hog, nombre_base, salida_hog / clase)

    # ResNet
    vector_resnet = extraer_resnet(imagen_pil)
    guardar_vector(vector_resnet, nombre_base, salida_resnet / clase)

    print(f"✓ Guardado {clase}/{nombre_base}")










##############################
# Etapa 2: Medidas de similitud.
##############################

from PIL import Image
from extractores.cnn_extractor import extraer_resnet
from utils.similarity import comparar_con_base

# Cargar imagen de consulta
ruta = "datasets/CBIR_50/manzana/manzana_001.jpg"
imagen = Image.open(ruta).convert("RGB")
vector_consulta = extraer_resnet(imagen)

# Comparar con vectores ResNet
resultados = comparar_con_base(vector_consulta, "vectores/resnet", medida="coseno")

print("Top 5 similares:")
for nombre, sim in resultados[:5]:
    print(f"{nombre} → {sim:.4f}")










##############################
# Etapa 3: Ordenar resultados por relevancia y calcular el Rank Normalizado.
##############################

from utils.evaluacion import calcular_rank_normalizado
from extractores.cnn_extractor import extraer_resnet
from pathlib import Path
from utils.image_loader import cargar_imagen

# === Evaluar Rank Normalizado en subconjunto ===
carpeta_imgs = Path("datasets/CBIR_50")
vectores_resnet = "vectores/resnet"
medida = "coseno"
rankings = []

# Usar 1 imagen por clase (10 clases aleatorias)
for i, carpeta_clase in enumerate(sorted(carpeta_imgs.iterdir())):
    if not carpeta_clase.is_dir():
        continue

    primera_imagen = list(carpeta_clase.glob("*.jpg"))[0]
    nombre = f"{carpeta_clase.name}/{primera_imagen.stem}"

    imagen = cargar_imagen(primera_imagen)
    vector = extraer_resnet(imagen)

    nr = calcular_rank_normalizado(nombre, vector, vectores_resnet, medida)
    rankings.append(nr)

    print(f"{nombre} → Rank Normalizado: {nr:.4f}")

    if i == 9:  # probar solo 10 clases por ahora
        break

# Promedio
promedio = sum(rankings) / len(rankings)
print(f"\nRank Normalizado promedio (10 clases): {promedio:.4f}")










##############################
# Etapa 4: Visualización cualitativa de resultados.
##############################

from extractores.cnn_extractor import extraer_resnet
from utils.similarity import comparar_con_base
from utils.image_loader import cargar_imagen
from utils.visualizacion import mostrar_resultados

# Imagen de consulta
ruta = "datasets/CBIR_50/Apple/Apple_1.jpg"
imagen = cargar_imagen(ruta)
vector = extraer_resnet(imagen)

# Comparar
resultados = comparar_con_base(vector, "vectores/resnet", medida="coseno")

# Visualizar resultados
mostrar_resultados(ruta, resultados, "datasets/CBIR_50", top_k=10)










##############################
# Etapa 6: Comparación de resultados y análisis gráfico.
##############################
from utils.visualizacion import mostrar_resultados
recalls, precisiones = curva_precision_recall(vec_cnn, clase_real="manzana", carpeta_vectores="vectores/resnet")

plt.plot(recalls, precisiones, marker='o')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Curva Precision–Recall (CNN - manzana)")
plt.grid(True)
plt.show()










##############################
# Etapa 7: Visualización del espacio de características.
##############################
from experimentos.visualizacion_caracteristicas import cargar_vectores_y_etiquetas, visualizar_2D

# HOG
X_hog, y_hog = cargar_vectores_y_etiquetas("vectores/hog/o9_ppc8_cpb2", n_clases=10, n_por_clase=10)
visualizar_2D(X_hog, y_hog, metodo="pca", titulo="HOG")
visualizar_2D(X_hog, y_hog, metodo="tsne", titulo="HOG")

# ResNet
X_cnn, y_cnn = cargar_vectores_y_etiquetas("vectores/resnet", n_clases=10, n_por_clase=10)
visualizar_2D(X_cnn, y_cnn, metodo="pca", titulo="ResNet")
visualizar_2D(X_cnn, y_cnn, metodo="tsne", titulo="ResNet")









##############################
# Etapa 8: Evaluación de robustez ante perturbaciones.
##############################
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

