from utils.image_loader import cargar_imagen, obtener_clase_desde_nombre
from extractores.hog_extractor import extraer_hog
from extractores.cnn_extractor import extraer_resnet
from pathlib import Path
import numpy as np
import os

dataset_path = "datasets/CBIR_50"
carpeta_salida_hog = "vectores/hog"
carpeta_salida_resnet = "vectores/resnet"

for archivo in os.listdir(dataset_path):
    ruta = os.path.join(dataset_path, archivo)
    imagen_pil = cargar_imagen(ruta)

    if imagen_pil:
        # HOG
        from skimage import img_as_ubyte
        import cv2
        img_np = np.array(imagen_pil)
        hog_vec = extraer_hog(img_np)
        guardar_vector(hog_vec, archivo.replace(".jpg", ""), carpeta_salida_hog)

        # ResNet
        resnet_vec = extraer_resnet(imagen_pil)
        guardar_vector(resnet_vec, archivo.replace(".jpg", ""), carpeta_salida_resnet)

        print(f"Procesada: {archivo}")

