from extractores.cnn_extractor import extraer_resnet
from utils.similarity import comparar_con_base
from utils.image_loader import cargar_imagen
from utils.visualizacion import mostrar_resultados

# Imagen de consulta
ruta = "datasets/CBIR_50/manzana/manzana_0001.jpg"
imagen = cargar_imagen(ruta)
vector = extraer_resnet(imagen)

# Comparar
resultados = comparar_con_base(vector, "vectores/resnet", medida="coseno")

# Visualizar resultados
mostrar_resultados(ruta, resultados, "datasets/CBIR_50", top_k=10)
