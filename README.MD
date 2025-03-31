# Comment Analyzer: Análisis de Sentimiento en Comentarios.

Este proyecto consistió en desarrollar un modelo de clasificación utilizando Machine Learning (ML) y Procesamiento del Lenguaje Natural (NLP), con el objetivo de detectar el sentimiento de compradores en un marketplace, luego de que estos, realicen una breve encuesta en donde comparten su experiencia sobre la compra y el producto recibido.

Estas devoluciones contienen información muya valiosa respecto a la reputación de una marca o el producto en el mercado y por ello merecen atención, por esa razón el modelo clasifica los comentarios en tres categorías:

- Negativo - Clase 0.
- Neutro - Clase 1.
- Positivo - Clase 2.

---

## Estructura del Proyecto:

```
market_sentiment_analysis
├── data
│ ├── dataset.csv # Datos originales del desafío
│ └── simulada
│ ├── test_comments.csv # Datos simulados para pruebas
│ └── predicciones
│ └── predicciones_test_comments.csv
├── enunciado
│ └── Challenge ML Engineer - Nubimetrics.pdf # Descripción del desafío
├── models
│ ├── comment_sentiment_analysis_model.pkl # Modelo entrenado
│ └── word_vectorizer
│ └── fit_count_vectorizer.pkl # Vectorizador entrenado
├── src
│ └── comment_analyzer.py # Clase para análisis y preprocesamiento
├── requirements.txt # Dependencias del proyecto
├── resolucion.ipynb # Notebook con la solución implementada
├── test.ipynb # Notebook de prueba
└── README.MD # Este archivo
```

---

## Creación de entorno de trabajo y ejecución de Comment Analyzer.

### 1. Crear Entorno Virtual.

Para asegurar la replicabilidad y consistencia del proyecto, crear un entorno virtual, puede ser usando `conda`, `virtualenv`, etc.

Ejemplo con conda:

```
conda create -n nubimetrics_env python=3.11
conda activate nubimetrics_env
```

### 2. Instalar Dependencias.

Desde la carpeta raiz del proyecto y ejecutar:

```
pip install -r requirements.txt
```

### 3. Ejecutar el entrenamiento del modelo y validación de la solución.

El archivo `nb_comment_analyzer_training.ipynb` contiene todo el proceso de desarrollo del modelo, incluyendo:

- Análisis exploratorio y preparación de los datos.
- Limpieza y preprocesamiento de texto.
- Codificación (encoding) de comentarios utilizando técnicas como Bag of Words.
- Entrenamiento y validación del modelo ganador.
- Optimización final del modelo.

Para ejecutar la notebook utilizar:

```
jupyter notebook nb_comment_analyzer_training.ipynb
```

### 4. Utilizar la clase Comment_Analyzer.

Pipeline integrado para el manejo de nuevos comentarios, facilita la carga, limpieza, preprocesamiento y predicción.

```
from src.comment_analyzer import Comment_Analyzer

# Inicializar clase
analyzer = Comment_Analyzer(
    path_to_comments='data/simulada/test_comments.csv',
    vectorizer_path='models/word_vectorizer/fit_count_vectorizer.pkl',
    model_path='models/comment_sentiment_analysis_model.pkl'
)

# Obtener predicciones
predictions = analyzer.predictions
```
