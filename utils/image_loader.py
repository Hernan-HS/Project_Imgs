from PIL import Image
from pathlib import Path
import os
import numpy as np
from pathlib import Path

def guardar_vector(vector, nombre_salida, carpeta_destino):
    Path(carpeta_destino).mkdir(parents=True, exist_ok=True)
    np.save(Path(carpeta_destino) / nombre_salida, vector)


def cargar_imagen(ruta):
    """
    Carga una imagen y la convierte a RGB.
    """
    try:
        return Image.open(ruta).convert("RGB")
    except Exception as e:
        print(f"Error cargando {ruta}: {e}")
        return None

def obtener_clase_desde_nombre(nombre_archivo):
    """
    Extrae el nombre de clase desde el nombre del archivo (formato: clase_id.jpg)
    """
    return nombre_archivo.split("_")[0]

