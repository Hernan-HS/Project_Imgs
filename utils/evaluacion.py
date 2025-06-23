from utils.similarity import comparar_con_base
from utils.image_loader import obtener_clase_desde_nombre

def calcular_rank_normalizado(nombre_consulta, vector_consulta, carpeta_vectores, medida="coseno"):
    """
    Calcula el Rank Normalizado para una imagen de consulta.
    
    nombre_consulta: str → Ej. "manzana/manzana_0001"
    """
    clase_real = nombre_consulta.split("/")[0]
    resultados = comparar_con_base(vector_consulta, carpeta_vectores, medida)

    posiciones_relevantes = []
    for i, (nombre, _) in enumerate(resultados):
        clase_predicha = nombre.split("/")[0]
        if clase_predicha == clase_real and nombre != nombre_consulta:
            posiciones_relevantes.append(i + 1)  # R_i (1-based)

    N_rel = len(posiciones_relevantes)
    N = len(resultados)

    if N_rel == 0:
        return 1.0  # peor caso

    rank_normalizado = sum(posiciones_relevantes) / (N_rel * N)
    return rank_normalizado


def curva_precision_recall(vector_consulta, clase_real, carpeta_vectores, medida="coseno"):
    """
    Calcula los puntos de la curva Precision–Recall para una imagen de consulta.

    Retorna dos listas: recall[], precision[]
    """
    resultados = comparar_con_base(vector_consulta, carpeta_vectores, medida)
    total_relevantes = sum(1 for nombre, _ in resultados if nombre.split("/")[0] == clase_real)

    precisiones = []
    recalls = []
    tp = 0

    for i, (nombre, _) in enumerate(resultados, start=1):
        clase_pred = nombre.split("/")[0]
        if clase_pred == clase_real:
            tp += 1

        recall = tp / total_relevantes
        precision = tp / i
        recalls.append(recall)
        precisiones.append(precision)

    return recalls, precisiones

