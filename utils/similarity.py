import numpy as np
from numpy.linalg import norm
from pathlib import Path

def similitud_coseno(vec1, vec2):
    """
    Retorna la similitud del coseno entre dos vectores.
    """
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2) + 1e-10)

def distancia_euclidiana(vec1, vec2):
    """
    Retorna la distancia euclidiana entre dos vectores.
    """
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return norm(vec1 - vec2)

def comparar_con_base(vector_consulta, carpeta_vectores, medida="coseno"):
    """
    Compara un vector de consulta con todos los vectores .npy en subdirectorios.

    Retorna: [(nombre_completo, similitud/distancia), ...]
    """
    resultados = []
    carpeta_vectores = Path(carpeta_vectores)

    for archivo in carpeta_vectores.rglob("*.npy"):
        vector = np.load(archivo)
        nombre = f"{archivo.parent.name}/{archivo.stem}"

        if medida == "coseno":
            valor = similitud_coseno(vector_consulta, vector)
        elif medida == "euclidiana":
            valor = distancia_euclidiana(vector_consulta, vector)
        else:
            raise ValueError("Medida no soportada")

        resultados.append((nombre, valor))

    reverse = True if medida == "coseno" else False
    resultados.sort(key=lambda x: x[1], reverse=reverse)

    return resultados
