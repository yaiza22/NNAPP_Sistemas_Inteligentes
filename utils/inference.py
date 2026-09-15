import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

def run_inference(x, framework, data_type):
    if framework == "tensorflow":
        from models.tensorflow_models import get_model
        model = get_model(data_type)
        preds = model.predict(x)
    else:
        from models.pytorch_models import get_model
        model = get_model(data_type)
        import torch
        with torch.no_grad():
            #preds = model(x).cpu().numpy()
            # 1. Convertimos 'x' a Tensor. 
            # Usamos hasattr para que no falle si 'x' es un DataFrame de Pandas o un arreglo normal de NumPy
            if hasattr(x, 'values'):
                x_tensor = torch.tensor(x.values, dtype=torch.float32)
            else:
                x_tensor = torch.tensor(x, dtype=torch.float32)
                # 2. Hacemos la predicción con el Tensor
                preds = model(x_tensor).cpu().numpy()

    # Convertir probabilidades a clases
    if data_type == "tabular":
        if preds.shape[1] == 1:
            # Clasificación binaria (sigmoid)
            classes = (preds > 0.5).astype(int).flatten()
        else:
            # Multiclase (softmax) - ej: Iris con 3 clases
            classes = np.argmax(preds, axis=1)
        return classes.tolist()
    return preds.tolist()
