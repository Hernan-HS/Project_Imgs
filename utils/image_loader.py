from PIL import Image
from pathlib import Path
import os


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

