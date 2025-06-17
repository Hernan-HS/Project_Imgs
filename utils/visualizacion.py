import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import os

def mostrar_resultados(imagen_consulta_path, resultados, ruta_base_imagenes, top_k=10):
    """
    Muestra imagen de consulta y top-k resultados.

    imagen_consulta_path: ruta de la imagen original consultada (Path o str)
    resultados: lista [(nombre_relativo, similitud/distancia), ...]
    ruta_base_imagenes: carpeta raíz de imágenes, ej: datasets/CBIR_50
    """
    fig, axes = plt.subplots(1, top_k + 1, figsize=(3 * (top_k + 1), 4))
    fig.suptitle("Consulta vs Top-K resultados", fontsize=16)

    # Mostrar imagen de consulta
    img_consulta = Image.open(imagen_consulta_path).convert("RGB")
    axes[0].imshow(img_consulta)
    axes[0].set_title("Consulta")
    axes[0].axis("off")

    # Mostrar top-k resultados
    for i in range(top_k):
        nombre, score = resultados[i]
        clase, nombre_archivo = nombre.split("/")
        ruta = Path(ruta_base_imagenes) / clase / f"{nombre_archivo}.jpg"
        if not ruta.exists():
            axes[i+1].set_title("No encontrada")
            axes[i+1].axis("off")
            continue

        img = Image.open(ruta).convert("RGB")
        axes[i+1].imshow(img)
        axes[i+1].set_title(f"{clase}\n{score:.2f}")
        axes[i+1].axis("off")

    plt.tight_layout()
    plt.show()

