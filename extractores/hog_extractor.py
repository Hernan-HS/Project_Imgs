from skimage.feature import hog
from skimage.color import rgb2gray
import numpy as np

def extraer_hog(imagen, parametros=None):
    """
    Extrae vector HOG de una imagen.

    imagen: ndarray RGB
    parametros: diccionario con opciones HOG

    Retorna: vector de características HOG (float32)
    """
    if parametros is None:
        parametros = {
            "orientations": 9,
            "pixels_per_cell": (8, 8),
            "cells_per_block": (2, 2),
            "block_norm": "L2-Hys"
        }

    gray = rgb2gray(imagen)
    features = hog(
        gray,
        orientations=parametros["orientations"],
        pixels_per_cell=parametros["pixels_per_cell"],
        cells_per_block=parametros["cells_per_block"],
        block_norm=parametros["block_norm"],
        transform_sqrt=True,
        feature_vector=True
    )

    return features.astype(np.float32)
