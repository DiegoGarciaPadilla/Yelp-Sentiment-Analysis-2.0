# Yelp Reviews Sentiment Analysis

**Autor:** Diego Antonio García Padilla

**Matrícula:** A01710777

## 1 Dataset

El dataset utilizado es el **[Yelp Academic Dataset](https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset)**, que contiene aproximadamente 7 millones de reseñas de negocios de la plataforma Yelp. Este dataset es ampliamente reconocido en la comunidad académica y ofrece:

- **Escala:** Suficientes datos para entrenar modelos complejos de deep learning
- **Diversidad:** Reviews de múltiples categorías de negocios y ubicaciones geográficas
- **Autenticidad:** Feedback real de usuarios con sus respectivas calificaciones
- **Riqueza lingüística:** Variedad en longitud, estilo y expresividad del texto

## 2 Preparación del entorno

El proyecto fue desarrollado originalmente en **Google Colab**, utilizando:

- Gemini integrado para la documentación
- Google Drive para persistencia de datos

**Ejecución Local (Opcional):**

Para ejecutar el proyecto en un ambiente local:

1. Crear entorno virtual

```bash
python -m venv venv
```

2. Activar entorno virtual

 - Linux/Mac
   
```bash
source venv/bin/activate
```

 - Windows

```bash
venv\Scripts\activate
```

3. Instalar dependencias

```bash
pip install -r requirements.txt
```

4. Remover celdas específicas de Colab

```python
from google.colab import drive
drive.mount('/content/drive')
```

## 3 ETL (Extract, Transform, Load)

El proceso ETL se documentó en el notebook `PortfolioETL.ipynb` y consistió en tres fases principales.

### 3.1 Extracción

La extracción del dataset se realizó mediante la librería `kagglehub`, que descarga automáticamente los archivos del Yelp Academic Dataset desde Kaggle:

```python
import kagglehub

# Download Yelp dataset

yelp_path = kagglehub.dataset_download("yelp-dataset/yelp-dataset")
print(f"Dataset downloaded to: {yelp_path}")
```

**Archivos obtenidos:**

- `yelp_academic_dataset_review.json` (5.09 GB) - **Archivo principal utilizado**
- `yelp_academic_dataset_business.json` (113 MB)
- `yelp_academic_dataset_user.json` (3.2 GB)
- `yelp_academic_dataset_checkin.json` (274 MB)
- `yelp_academic_dataset_tip.json` (172 MB)

Para este proyecto, únicamente se requirió el archivo de reviews, que contiene el texto y las calificaciones necesarias para el análisis de sentimientos.

### 3.2 Transformación

La transformación del dataset involucró seis pasos secuenciales de preprocesamiento:

#### **Paso 1: Selección de features**

Para optimizar el uso de memoria y enfocarse en las variables relevantes, se extrajeron únicamente dos columnas del dataset original: **text** y **stars**.

```python
def select_features(self, df: pd.DataFrame, features: list) -> pd.DataFrame:
  return df[features].copy()


df = self.select_features(df, ["text", "stars"])
```

#### **Paso 2: Eliminación de duplicados**

Los duplicados pueden introducir sesgo al modelo, haciendo que memorice reviews repetidas en lugar de aprender patrones generalizables. Su eliminación asegura que cada ejemplo de entrenamiento sea único.

```python
def drop_duplicates(self, df):
  return df.drop_duplicates()
```

#### **Paso 3: Creación de feature de sentimiento**

Dado que el dataset original utiliza calificaciones de 1-5 estrellas, se creó una variable categórica `sentiment` que agrupa las estrellas en tres clases:

```python
def create_sentiment_column(self, df: pd.DataFrame) -> pd.DataFrame:
    def _classify(stars):
      if stars in (1.0, 2.0): return "negative"
      if stars == 3.0:        return "neutral"
      if stars in (4.0, 5.0): return "positive"
      return None

    df = df.copy()
    df["sentiment"] = df["stars"].apply(_classify)

    return df[["text", "sentiment"]]
```

**Mapeo:**

- **Negativo:** 1-2 estrellas (experiencias insatisfactorias)
- **Neutral:** 3 estrellas (experiencias medianas o mixtas)
- **Positivo:** 4-5 estrellas (experiencias satisfactorias)

Este agrupamiento es estándar en análisis de sentimientos y refleja cómo los consumidores interpretan las calificaciones en la práctica. Una calificación de 3/5 raramente se considera "positiva" en contextos comerciales.

#### **Paso 4: Balanceo del dataset**

Dado que las reviews de Yelp presentan un desbalance natural, se implementó un downsample para igualar el número de ejemplos por clase:

- ~67% positivas (4-5 estrellas)
- ~23% negativas (1-2 estrellas)
- ~10% neutrales (3 estrellas)

```python
def balance_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
    min_count = df["sentiment"].value_counts().min()
    print(f"  Balancing to {min_count:,} reviews per class…")

    df_balanced = (
      df.groupby("sentiment", group_keys=False)
        .apply(lambda g: g.head(min_count))
        .reset_index(drop=True)
    )

    print(f"  Balanced size: {len(df_balanced):,} reviews")
    return df_balanced

```

Sin este balanceo, un modelo "ingenuo" podría alcanzar 67% de accuracy simplemente prediciendo "positivo" para todas las reviews, sin aprender realmente a distinguir sentimientos. El balanceo fuerza al modelo a aprender características discriminativas de cada clase.

Sin embargo, esto no es suficiente: elegir la métrica de evaluación adecuada, como el F1-score, es crucial para evaluar el rendimiento del modelo. Esto se verá más adelante.

#### Nota sobre el preprocesamiento

En el caso de los modelos basados en BERT, se omite deliberadamente el proceso estándar de limpieza de texto (eliminación de palabras vacías, lematización y tokenización personalizada).

El tokenizador de BERT se encarga internamente de todo el preprocesamiento necesario (conversión a minúsculas, división de subpalabras, relleno e inserción de tokens especiales). Lo único que hay que hacer es proporcionar la cadena de texto sin procesar de la reseña.

### 3.3 Carga

El dataset procesado se guardó en formato Parquet para persistencia eficiente,

El dataset final contiene las siguientes columnas:

- `text`: texto original
- `sentiment`: sentimiento (positivo, negativo, neutral)

El siguiente paso natural fue diseñar arquitecturas de deep learning capaces de extraer patrones significativos de estos millones de reviews. La estrategia consistió en comenzar con un modelo simple para establecer un baseline, e incrementar la complejidad mediante técnicas como el fine-tuning y la atención.

## 4 Modelos

### 4.1 Primer modelo: BERT + BiLSTM

Este primer modelo, propuesto por Nkhata et al. (2025), consiste en un modelo BERT pre-entrenado seguido de una capa BiLSTM y una capa de salida con activación softmax.

![Modelo BERT + BiLSTM](images/bert-bilstm.png)

#### 4.1.1 Arquitectura

La arquitectura del modelo es la siguiente:

| Layer (type)                         | Output Shape                                                                                                                                                                                      | Param #      | Connected to                                  |
| :-------------------------------------| :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------| :-------------| :----------------------------------------------|
| `input_ids`                          | `[(None, 256)]`                                                                                                                                                                                   | 0            | `[]`                                          |
| `attention_mask`                     | `[(None, 256)]`                                                                                                                                                                                   | 0            | `[]`                                          |
| `bert`                               | `TFBaseModelOutputWithPoolingAndCrossAttentions(last_hidden_state=(None, 256, 768), pooler_output=(None, 768), past_key_values=None, hidden_states=None, attentions=None, cross_attentions=None)` | 109,482,240 | `['input_ids[0][0]', 'attention_mask[0][0]']` |
| `getitem_1`                          | `(None, 768)`                                                                                                                                                                                     | 0            | `['bert[0][0]']`                              |
| `reshape_1` (Reshape)                | `(None, 1, 768)`                                                                                                                                                                                  | 0            | `['tf.__operators__.getitem_1[0][0]']`        |
| `bidirectional_lstm` (Bidirectional) | `(None, 128)`                                                                                                                                                                                     | 426,496     | `['reshape_1[0][0]']`                         |
| `output` (Dense)                     | `(None, 3)`                                                                                                                                                                                       | 387          | `['bidirectional_lstm[0][0]']`                |

**Hiperparámetros**

En esta primera iteración, se mantuvieron los hiperparámetros propuestos por los autores del artículo:

- ***Batch size:*** 64
- ***Learning rate:*** 1e-4
- ***Sequence length:*** 128
- ***Number of epochs:*** 15

Así mismo, se usó el optimizador Adam para el entrenamiento del modelo.

**Descripción de las capas**

1. ***BERT (Bidirectional Encoder Representations from Transformers):*** 
  
BERT es un modelo de deep learning basado en transformers que ha demostrado ser muy eficaz en tareas de procesamiento de lenguaje natural.

2. ***BiLSTM (Bidirectional Long Short-Term Memory):*** 

BiLSTM es una variante de LSTM que procesa la secuencia en ambas direcciones (adelante y atrás), permitiendo capturar dependencias a largo plazo, y evitando los problemas de gradiente desvanecido y explosivo de las redes recurrentes tradicionales.

Por ejemplo: En la frase "La película no fue buena, fue una obra maestra", una LSTM unidireccional, al leer "no fue buena", podría asignar prematuramente un sentimiento negativo. Sin embargo, la BiLSTM, al leer también desde el final hacia atrás ("obra maestra... fue"), entiende que "no fue buena" en realidad es el preámbulo de un elogio mayor, capturando el contexto completo y la ironía o énfasis de la frase.

3. ***Softmax:*** 
 
Proyecta el vector de contexto a 3 neuronas (una por clase) y aplica Softmax para convertir en distribución de probabilidad:

```none
Ejemplo de output: [0.05, 0.10, 0.85]
Interpretación: 5% negativo, 10% neutral, 85% positivo
```

#### 4.1.2 Resultados

El modelo fue entrenado durante 4 epochs con los siguientes resultados:

- **Épocas entrenadas:** 4/15
- **Mejor época:** 2 (Epoch 2)

| **Loss** | **Accuracy** | **Precision** | **Recall** |
| :---------| :-------------| :--------------| :-----------|
| 0.3634   | 0.8492       | 0.8528        | 0.8450     |


![BERT + BiLSTM Training History](images/bert-bilstm-training-history.png)

1. **Divergencia en loss (Ligero overfitting):**

La gráfica de loss muestra formación en "U" característica:

- **Train loss:** Decrece consistentemente 
- **Validation loss:** Decrece inicialmente, luego aumenta

Esta divergencia indica que el modelo está memorizando el conjunto de entrenamiento en lugar de aprender patrones generalizables.

2. **Estancamiento en accuracy:**

- Train accuracy: Continúa creciendo (82% → 91%)
- Validation accuracy: Se estanca alrededor de 83-85%

3. **Matriz de confusión:**

![BERT + BiLSTM Confusion Matrix](images/bert-bilstm-confusion-matrix.png)

**Insights de la matriz:**

- El modelo es capaz de clasificar correctamente, sin embargo, tiene dificultad con reviews de sentimiento neutral.

**Conclusión del baseline:**

El modelo alcanza ~85% de accuracy. Lo cual es una mejora sobre el primer acercamiento realizado en el repositorio (https://github.com/DiegoGarciaPadilla/Yelp-Sentiment-Analysis).

Sin embargo, se puede mejorar para obtener un mejor rendimiento explorando técnicas como el fine-tuning o la atención. 

## Referencias

- Nkhata, G., Gauch, S., Anjum, U., & Zhan, J. (2025). Fine-tuning BERT with Bidirectional LSTM for Fine-grained Movie Reviews Sentiment Analysis (Version 1). arXiv. https://doi.org/10.48550/ARXIV.2502.20682
