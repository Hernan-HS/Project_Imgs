from sklearn.decomposition import PCA
import numpy as np
import os
from pathlib import Path

def aplicar_pca_en_carpeta(carpeta_origen, carpeta_destino, n_dim=128):
    """
    Aplica PCA a todos los vectores de una carpeta y guarda los reducidos.
    """
    carpeta_origen = Path(carpeta_origen)
    carpeta_destino = Path(carpeta_destino)
    carpeta_destino.mkdir(parents=True, exist_ok=True)

    # Recolectar todos los vectores y nombres
    vectores = []
    rutas = []

    for archivo in carpeta_origen.rglob("*.npy"):
        vectores.append(np.load(archivo))
        rutas.append(archivo)

    X = np.array(vectores)
    print(f"Reduciendo {X.shape[0]} vectores de dimensión {X.shape[1]} a {n_dim}...")

    pca = PCA(n_components=n_dim)
    X_reducido = pca.fit_transform(X)

    # Guardar los nuevos vectores
    for vec, ruta in zip(X_reducido, rutas):
        salida = carpeta_destino / ruta.relative_to(carpeta_origen)
        salida.parent.mkdir(parents=True, exist_ok=True)
        np.save(salida.with_suffix(".npy"), vec.astype(np.float32))

    print("✔ PCA aplicado y vectores guardados.")

