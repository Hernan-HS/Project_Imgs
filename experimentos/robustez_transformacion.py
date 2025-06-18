import numpy as np
import cv2
import random
from pathlib import Path
from extractores.cnn_extractor import extraer_resnet
from utils.image_loader import cargar_imagen
from utils.evaluation import calcular_rank_normalizado
import matplotlib.pyplot as plt

# === Transformaciones ===

def rotar_imagen(imagen_pil, angulo):
    return imagen_pil.rotate(angulo)

def agregar_ruido_gaussiano(imagen_np, sigma=25):
    ruido = np.random.normal(0, sigma, imagen_np.shape).astype(np.int16)
    img_ruidosa = np.clip(imagen_np.astype(np.int16) + ruido, 0, 255).astype(np.uint8)
    return img_ruidosa

def deformacion_elastica(imagen_np, alpha=34, sigma=4):
    random_state = np.random.RandomState(None)
    shape = imagen_np.shape[:2]
    dx = cv2.GaussianBlur((random_state.rand(*shape) * 2 - 1), (17, 17), sigma) * alpha
    dy = cv2.GaussianBlur((random_state.rand(*shape) * 2 - 1), (17, 17), sigma) * alpha

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)

    return cv2.remap(imagen_np, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

# === Evaluación ===

def evaluar_robustez(ruta_imagen, vector_base_dir, metodo_extractor, transform_func, nombre_transform):
    """
    Evalúa el Rank Normalizado antes y después de aplicar una transformación.
    """
    from PIL import Image
    clase = ruta_imagen.parent.name
    nombre = f"{clase}/{ruta_imagen.stem}"

    original = cargar_imagen(ruta_imagen)
    vec_original = metodo_extractor(original)
    nr_original = calcular_rank_normalizado(nombre, vec_original, vector_base_dir, "coseno")

    if transform_func == "rotar":
        pert = rotar_imagen(original, angulo=30)
    elif transform_func == "ruido":
        pert = agregar_ruido_gaussiano(np.array(original))
        pert = Image.fromarray(pert)
    elif transform_func == "deformar":
        pert = deformacion_elastica(np.array(original))
        pert = Image.fromarray(pert)
    else:
        raise ValueError("Transformación no soportada")

    vec_pert = metodo_extractor(pert)
    nr_pert = calcular_rank_normalizado(nombre, vec_pert, vector_base_dir, "coseno")

    print(f"{nombre} | Original: {nr_original:.4f} | {nombre_transform}: {nr_pert:.4f}")
    return nr_original, nr_pert

