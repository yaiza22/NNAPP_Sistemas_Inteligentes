# NNAPP Sistemas Inteligentes: Clasificación de Cáncer de Mama (Wisconsin)

En esta actividad aplicamos cambios para entrenar y predecir diagnósticos de cáncer de mama utilizando el dataset de Wisconsin (`load_breast_cancer`). Hicimos ajustes en las redes neuronales tanto en **PyTorch** como en **TensorFlow**.

## Miembros del Equipo
* Yaiza Angelina Sanchez Dueñez - 2220232034
* Nicolas Gonzalez Ortiz - 2220261089

---

## Contexto y Problema Original
La aplicación base estaba configurada originalmente para trabajar con datasets genéricos multiclase o de imágenes. Al intentar ingerir los datos médicos del dataset de Wisconsin, encontramos varias incompatibilidades estructurales:

1. **Dimensiones Incorrectas:** La red esperaba 4 características, pero el dataset de Wisconsin cuenta exactamente con 30 variables biológicas. 
2. **Naturaleza del Problema:** El modelo estaba utilizando funciones de pérdida y salidas multiclase (Softmax / CrossEntropyLoss) para un problema que en realidad es de clasificación binaria (Maligno (0) o Benigno (1)).
3. **Formatos de Datos (Tensores):** El pipeline de inferencia no traducía correctamente los datos crudos (Pandas/NumPy) al formato tensorial requerido por PyTorch, causando caídas del servidor (Error 500).
4. **Errores de Guardado:** El preprocesamiento de Pandas guardaba índices como columnas extra (`Unnamed: 0`), arruinando las dimensiones de la red neuronal. Era un error de preprocesamiento que ocurría antes de generar los archivos en formato csv de la data dividida para entrenamiento y predicción (notebook: `load_dataset_cancer_train_predict.ipynb`)

---

## Mejoras y Modificaciones Implementadas

Para adaptar el sistema a la predicción de cáncer de mama, realizamos las siguientes intervenciones clave:

### 1. Preprocesamiento de Datos (Generación de CSV)
* **Eliminación del Índice:** Modificamos el script de extracción de `sklearn.datasets` añadiendo el parámetro `index=False` al exportar los archivos `.to_csv()`. Esto solucionó el error de _"size mismatch"_ (31 vs 30) asegurando que el modelo reciba estrictamente 30 características clínicas (en el notebook: `load_dataset_cancer_train_predict.ipynb`).

### 1.1. Nuevo notebook: 
* Se creó este notebook (`notebooks/load_dataset_cancer_train_predict.ipynb`) en colab para poder descargar los archivos csv con la data de entrenamiento y de predicción con el dataset de kaggle: load_breast_cancer de Wisconsin.

### 1.2. Datasets creados con el nuevo notebook:
* En la carpeta `datasets/tabular/` se agregó la data para entrenamiento, la data para predicción y los datos esperados de la predicción. Con los siguientes nombres, respectivamente: `breast_cancer_train.csv`, `breast_cancer_pred.csv`, `breast_cancer_results_y_pred.csv`

* Con el archivo `breast_cancer_results_y_pred` o en el mismo notebook `load_dataset_cancer...ipynb` se puede observar los datos esperados de la predicción para ser comparados con la predicción de los modelos pytorch y tensorflow.


### 2. Arquitectura de la Red Neuronal (`pytorch_arch.py` / `tensorflow_arch.py`)
En `pytorch_arch.py`:
* **Ajuste de Entradas y Salidas:** Se configuró el `input_dim` de la red tabular a 30 variables.
* **Activación de Salida:** Se reemplazó la salida de 2 neuronas por **1 sola neurona** (`n_classes=1`) equipada con una función de activación **Sigmoide**, ideal para predecir probabilidades entre 0 y 1.

En ambas arquitecturas (`pytorch_arch.py` / `tensorflow_arch.py`): 
* **Una capa oculta más:** En ambas arquitecturas solo habían 2 capas neuronales ocultas (de 32 y 16 neuronas), así que se revisó la guía de clase (presentación donde se explica la app) y se agregó otra capa de 64 neuronas.


### 3. Lógica de Entrenamiento (`trainer_pt.py`)
* **Función de Pérdida (Loss Function):** Se cambió la función `CrossEntropyLoss` (multiclase) por `BCELoss` (Binary Cross Entropy), permitiendo que la red aprenda correctamente la diferencia binaria entre tumores.
* **Manejo de Directorios:** Se añadió lógica al backend para crear automáticamente las carpetas de `uploads/` y `models/saved/` y evitar errores de `FileNotFoundError`.


### 4. Motor de Inferencia (`utils/inference.py   (línea 16 - 24)`)
* **Conversión de Tensores:** Se implementó una capa de seguridad matemática que convierte los DataFrames de Pandas (`x.values`) en tensores de PyTorch (`torch.tensor(..., dtype=torch.float32)`) antes de inyectarlos a la red, solucionando el error TypeError de NumPy.

---

## Conclusiones del Rendimiento

1. **Precisión Matemática:** Tras las modificaciones, la arquitectura feedforward logró clasificar con éxito 28 de 29 pacientes del set de prueba, obteniendo un Accuracy (Precisión) aproximado del **96.55%**.
2. **Equivalencia de Frameworks:** Se logró un comportamiento idéntico entre las arquitecturas de **PyTorch** y **TensorFlow**, demostrando una programación consistente y datos excelentemente balanceados/normalizados, ambas predicciones fueron iguales.
3. **Análisis del Error (Falso Negativo):** El único error del modelo clasificó un tumor maligno como benigno. Este hallazgo destaca la importancia de monitorear este tipo específico de fallos en despliegues médicos y abre la puerta a futuros ajustes (como penalizar los falsos negativos).

---





### 


