from utils.similarity import comparar_con_base

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

