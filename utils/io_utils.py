import numpy as np
from pathlib import Path

def guardar_vector(vector, nombre_salida, carpeta_destino):
    """
    Guarda un vector de características en formato .npy.

    Parámetros:
        vector: ndarray
        nombre_salida: nombre del archivo sin extensión
        carpeta_destino: carpeta donde se guardará el vector
    """
    Path(carpeta_destino).mkdir(parents=True, exist_ok=True)
    ruta = Path(carpeta_destino) / f"{nombre_salida}.npy"
    np.save(ruta, vector)
