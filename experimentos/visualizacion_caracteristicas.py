import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from pathlib import Path
import random

def cargar_vectores_y_etiquetas(carpeta_vectores, n_clases=10, n_por_clase=10):
    """
    Carga vectores .npy y etiquetas de un subconjunto del dataset.
    """
    X = []
    y = []
    clases = sorted([d for d in Path(carpeta_vectores).iterdir() if d.is_dir()])
    clases = random.sample(clases, n_clases)

    for clase in clases:
        archivos = list(clase.glob("*.npy"))
        seleccion = random.sample(archivos, min(n_por_clase, len(archivos)))
        for archivo in seleccion:
            vector = np.load(archivo)
            X.append(vector)
            y.append(clase.name)

    return np.array(X), np.array(y)

def visualizar_2D(X, y, metodo="pca", titulo=""):
    """
    Reduce dimensiones y visualiza en 2D.
    """
    if metodo == "pca":
        reducer = PCA(n_components=2)
    elif metodo == "tsne":
        reducer = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    else:
        raise ValueError("Método no soportado")

    X_reducido = reducer.fit_transform(X)

    plt.figure(figsize=(10, 8))
    for clase in np.unique(y):
        idxs = y == clase
        plt.scatter(X_reducido[idxs, 0], X_reducido[idxs, 1], label=clase, alpha=0.7)

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(f"Visualización 2D ({metodo.upper()}) - {titulo}")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

