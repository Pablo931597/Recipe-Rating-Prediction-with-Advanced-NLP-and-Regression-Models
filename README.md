# Master_uc3m
Uc3m projects
Convocatoria Ordinaria: Proyecto realizado por Emilio Muñoz Álvarez y Pablo Ebohon Serna

Convocatoria Extraordinaria: Proyecto realizado por Pablo Ebohon Serna

## Introducción

Este proyecto tiene como objetivo aplicar técnicas avanzadas de procesamiento y análisis de datos textuales para abordar una tarea de aprendizaje automático centrada en documentos de recetas de cocina. Se tendrá que resolver un problema de regresión, utilizando diferentes representaciones vectoriales de texto y estrategias de aprendizaje automático, con el fin de predecir la puntuación de las recetas a partir de la información textual y los metadatos disponibles.

Para la realización de este proyecto se utilizará un fichero JSON, proporcionado en la asignatura. El conjunto de datos utilizado contiene más de 20,000 recetas, incluyendo información textual como instrucciones, descripciones, categorías y títulos, así como variables numéricas como valoraciones, grasas, proteínas, calorías, sodio, ingredientes y fecha de publicación. El desarrollo del proyecto implica el procesado de los textos, la representación vectorial de los documentos mediante técnicas como TF-IDF, Word2Vec y embeddings contextuales basados en Transformers, y el entrenamiento y evaluación de modelos de regresión utilizando redes neuronales.

Además, se realiza una comparación de los resultados obtenidos con un modelo preentrenado de Hugging Face para regresión, y se lleva a cabo un análisis crítico de los resultados apoyado en métricas y visualizaciones adecuadas para problemas de regresión.

## Análisis de las Categorías

Para comprender la influencia de las distintas categorías en la puntuación de las recetas, se ha realizado un análisis exhaustivo de la columna categories del conjunto de datos. En primer lugar, se ha asegurado que esta columna estuviera limpia y en un formato utilizable, desglosando las listas de categorías para que cada fila represente una única categoría asociada a una receta.

Posteriormente, se agruparon las recetas por categoría y se calculó la media de la puntuación (rating) para cada una de ellas. Este procedimiento permitió identificar las categorías con mayor valoración promedio, mostrando las diez categorías mejor puntuadas del conjunto de datos. Este análisis es útil para detectar tendencias y preferencias dentro de la comunidad de usuarios que valoran las recetas.

Además, se extrajeron todas las categorías presentes en el DataFrame y se ordenaron según su frecuencia de aparición. De este modo, se identificaron las categorías más representadas en el conjunto de datos, lo que permite enfocar el análisis en aquellas con mayor relevancia estadística. A partir de este listado, se seleccionaron las cinco categorías más frecuentes para analizar en detalle su puntuación media.

Finalmente, se filtraron las recetas correspondientes a estas cinco categorías principales y se calculó la media de sus puntuaciones. Este enfoque permite comparar de manera directa el desempeño de las categorías más populares y observar si existe alguna relación entre la frecuencia de aparición y la valoración recibida por los usuarios. Los resultados obtenidos ofrecen una visión clara sobre qué tipos de recetas tienden a recibir mejores valoraciones y cuáles son las preferencias predominantes en el conjunto de datos analizado.

A continuavión se muestran algunas de las gráficas obtenidas:

![alt text](https://github.com/Pablo931597/Master-UC3m/blob/main/Captura%20de%20pantalla%202025-06-18%20a%20las%2012.33.03.png)
![alt text2](https://github.com/Pablo931597/Master-UC3m/blob/main/Captura%20de%20pantalla%202025-06-18%20a%20las%2012.33.38.png)

## Procesamiento de los datos de las recetas

El preprocesamiento de los datos textuales es un paso fundamental para preparar la información de las recetas antes de aplicar técnicas de aprendizaje automático. En este proyecto, se ha realizado un pipeline de preprocesamiento que transforma el texto en un formato adecuado para su análisis y modelado posterior.

En primer lugar, se instalaron y configuraron las librerías necesarias, como NLTK, spaCy y Gensim, que proporcionan herramientas avanzadas para el procesamiento de lenguaje natural. Se descargaron los recursos lingüísticos requeridos, incluyendo modelos de idioma y listas de palabras vacías (stopwords).

El proceso de preprocesamiento comenzó con la unión de las columnas relevantes del conjunto de datos, concretamente las columnas `directions` y `desc`, para formar un único texto representativo de cada receta. Posteriormente, se aplicó una función de preprocesamiento que incluye varias etapas clave:

- **Tokenización:** El texto se divide en palabras individuales (tokens), facilitando su análisis posterior.
- **Limpieza:** Se eliminan caracteres especiales y signos de puntuación, conservando únicamente las palabras alfabéticas y normalizando todo a minúsculas.
- **Lematización:** Cada palabra se reduce a su forma base o lema, lo que ayuda a unificar diferentes variantes de una misma palabra.
- **Eliminación de stopwords:** Se eliminan las palabras vacías, es decir, aquellas que no aportan significado relevante al análisis, como preposiciones y artículos.

Una vez preprocesados los textos, se construyó un diccionario utilizando la librería Gensim, que asigna un identificador único a cada término del vocabulario. Para reducir la dimensionalidad y eliminar términos poco informativos, se filtraron las palabras que aparecen en menos de cuatro documentos o en más del 80% de ellos. Finalmente, cada documento se representó como un vector de ocurrencias de palabras, listo para ser utilizado en las siguientes etapas del pipeline de aprendizaje automático.

Este proceso garantiza que los datos textuales estén limpios, normalizados y estructurados de manera eficiente, facilitando la extracción de información relevante y mejorando el rendimiento de los modelos de regresión aplicados posteriormente.

### TF-IDF: 

Para transformar los textos de las recetas en una representación numérica adecuada para el modelado, se ha empleado la técnica TF-IDF, ampliamente utilizada en el procesamiento de lenguaje natural para ponderar la importancia de las palabras en un corpus de documentos. El objetivo de esta técnica es asignar un peso mayor a aquellas palabras que son relevantes en un documento pero poco frecuentes en el resto del corpus, permitiendo así capturar mejor la información distintiva de cada receta.

El proceso comenzó con la construcción de un modelo TF-IDF utilizando la librería Gensim, a partir del corpus preprocesado. Cada documento fue transformado en un vector de pesos TF-IDF, donde cada posición representa la importancia relativa de una palabra en ese documento. Para ilustrar la interpretación de estos vectores, se extrajeron y visualizaron los términos con mayor peso en el primer documento, lo que permite identificar rápidamente las palabras más representativas de cada receta.

Posteriormente, se utilizó la clase `TfidfVectorizer` de Scikit-learn para generar una matriz TF-IDF de mayor tamaño, limitando el vocabulario a las 2,000 palabras más frecuentes y relevantes del corpus. Esta matriz fue utilizada para analizar la distribución de los pesos TF-IDF en los documentos y para identificar las palabras más influyentes tanto a nivel individual como global. Se calcularon y graficaron las palabras con mayor peso promedio en todo el corpus, lo que proporciona una visión general de los términos más característicos de las recetas analizadas.

Para explorar la estructura interna de los datos, se aplicó una reducción de dimensionalidad mediante PCA, proyectando los documentos en un espacio bidimensional. Esta visualización facilita la identificación de agrupamientos y patrones en la distribución de las recetas según su contenido textual.

Finalmente, se calculó la similitud entre documentos utilizando la métrica de coseno sobre los vectores TF-IDF. Se seleccionó un subconjunto aleatorio de 500 recetas y se representó la matriz de similitud mediante un mapa de calor, lo que permite observar la existencia de grupos de recetas con alto grado de similitud semántica, así como la diversidad temática presente en el corpus.

En conjunto, la representación TF-IDF ha permitido transformar eficazmente los textos de las recetas en vectores numéricos interpretables, facilitando tanto el análisis como la posterior aplicación de modelos de aprendizaje automático para la predicción de la puntuación de las recetas.


### Word2Vec: 

Para capturar la semántica de las palabras presentes en las recetas, se ha utilizado la técnica Word2Vec, que permite transformar cada término en un vector denso de alta dimensión, reflejando relaciones de similitud y contexto en el corpus analizado. El modelo se entrenó sobre los textos preprocesados de las recetas, utilizando una dimensión de 500 para los vectores, una ventana de contexto de 5 palabras y considerando todas las palabras presentes en el conjunto de datos.

Una vez entrenado el modelo, es posible obtener el vector asociado a cualquier palabra del vocabulario, como por ejemplo `'turkey'`, lo que permite analizar su posición en el espacio semántico generado. Además, se pueden identificar las palabras más similares a un término dado, como `'place'`, mostrando las diez palabras que comparten mayor proximidad semántica según el modelo. Este análisis resulta útil para descubrir sinónimos, términos relacionados o agrupaciones temáticas dentro del corpus.

El vector correspondiente a la palabra `'place'` es un array de 500 dimensiones, que encapsula la información contextual aprendida a partir de su uso en las recetas. La forma y valores de este vector permiten su utilización en tareas posteriores de agrupamiento, reducción de dimensionalidad o visualización.

Por último, se puede calcular la similitud entre dos palabras, como `'place'` y `'stock'`, utilizando la métrica de coseno sobre sus vectores. Este valor cuantifica el grado de relación semántica entre ambos términos en el contexto de las recetas, facilitando la exploración de asociaciones y patrones lingüísticos en el conjunto de datos.

En resumen, la representación Word2Vec proporciona una forma potente y flexible de modelar el significado de las palabras en el corpus de recetas, permitiendo tanto el análisis exploratorio como la integración en modelos de aprendizaje automático más complejos.

### BERT:

Para obtener una representación contextual avanzada de los textos de las recetas, se ha empleado BERT, un modelo preentrenado de lenguaje natural que permite capturar relaciones semánticas profundas y dependencias contextuales entre palabras. El proceso se ha centrado en extraer los embeddings de los documentos a partir del token especial `[CLS]`, que resume la información global de cada texto y es ampliamente utilizado para tareas de clasificación y regresión en NLP.

El procedimiento comenzó cargando el tokenizador y el modelo preentrenado `bert-base-uncased`, configurado para devolver los estados ocultos de todas las capas. Se seleccionaron las columnas `directions` y `desc` de las primeras 100 recetas, concatenando su contenido para formar el texto de entrada de cada documento. Cada texto fue tokenizado, truncado a un máximo de 512 tokens y enriquecido con los tokens especiales `[CLS]` y `[SEP]`, siguiendo el formato requerido por BERT.

A continuación, los textos tokenizados se convirtieron en índices numéricos y se rellenaron con ceros para igualar la longitud máxima. Se generaron también las máscaras de segmento necesarias para el modelo. Los tensores resultantes se pasaron por BERT en modo evaluación, obteniendo como salida los vectores de estado oculto para cada token de cada documento.

De cada documento, se extrajo el embedding correspondiente al token `[CLS]`, que actúa como una representación densa y contextualizada del texto completo. Estos embeddings se almacenaron en una matriz de tamaño `(n_documentos, 768)`, donde cada fila representa un documento en el espacio semántico aprendido por BERT.

Esta representación permite capturar matices complejos del lenguaje y relaciones de alto nivel entre las recetas, superando las limitaciones de técnicas tradicionales como TF-IDF o Word2Vec. Los embeddings obtenidos pueden utilizarse directamente como entrada para modelos de regresión, clasificación o análisis exploratorio, facilitando la identificación de patrones y similitudes profundas en el corpus de recetas analizado.

## Entrenamiento y rendimiento de una red neuronal

Para abordar la predicción de la puntuación de las recetas, se ha implementado una red neuronal utilizando PyTorch, una de las librerías más versátiles y extendidas para el desarrollo de modelos de aprendizaje profundo. El proceso de entrenamiento y evaluación se ha estructurado siguiendo las mejores prácticas para problemas de regresión supervisada, asegurando la correcta generalización del modelo y la interpretación de sus resultados.

### Con TF-IDF

Para evaluar la capacidad predictiva de los textos procesados mediante la técnica TF-IDF, se utilizó la matriz de características generada como entrada para una red neuronal construida en PyTorch. El conjunto de datos se dividió en entrenamiento y prueba, asegurando la correcta validación del modelo. La arquitectura de la red incluyó varias capas densas con activaciones ReLU, finalizando en una única neurona para la predicción de la puntuación de las recetas. El entrenamiento se realizó utilizando la función de pérdida de error cuadrático medio y el optimizador AdamW, monitorizando la evolución de la pérdida para evitar el sobreajuste. El rendimiento se evaluó mediante métricas como el MSE y el coeficiente de determinación, permitiendo cuantificar la precisión de las predicciones sobre el conjunto de prueba.

### Con Word2Vec

Empleando los vectores generados por Word2Vec, cada receta fue representada por la media de los vectores de sus palabras, obteniendo así una representación densa y semánticamente informativa. Estos vectores se utilizaron como entrada para una red neuronal de arquitectura similar a la empleada con TF-IDF, adaptando el número de neuronas de entrada a la dimensión de los embeddings de Word2Vec. El proceso de entrenamiento y validación siguió la misma metodología, utilizando MSELoss y AdamW, y evaluando el rendimiento con las mismas métricas de regresión. Esta aproximación permitió comparar la capacidad de los embeddings semánticos frente a las representaciones basadas en frecuencia de términos.

### Con BERT

Se extrajeron los vectores correspondientes al token `[CLS]` de cada documento, que resumen la información global de la receta. Estos embeddings, de mayor dimensionalidad y capacidad expresiva, se emplearon como entrada para una red neuronal adaptada a este tipo de representación. El entrenamiento se realizó siguiendo el mismo pipeline, utilizando MSELoss y AdamW, y evaluando el rendimiento mediante MSE y R² sobre el conjunto de prueba. El uso de BERT permitió capturar relaciones complejas y matices lingüísticos que pueden mejorar la precisión en la predicción de la puntuación de las recetas.

En conjunto, la comparación de los resultados obtenidos con las tres representaciones, TF-IDF, Word2Vec y BERT permitió analizar el impacto de cada técnica en el rendimiento de la red neuronal y seleccionar la estrategia más adecuada para el problema de predicción abordado.

## Entrenamiento del modelo de regresión Ridge con Scikit-learn

Para evaluar el rendimiento de técnicas de vectorización textual en la predicción de la puntuación de recetas, se ha implementado un modelo de regresión Ridge utilizando la librería Scikit-learn. Este modelo se ha entrenado y validado sobre las representaciones TF-IDF, Word2Vec y BERT, permitiendo comparar la capacidad predictiva de cada enfoque en un entorno controlado y reproducible.

### Con TF-IDF

En primer lugar, se utilizó la matriz TF-IDF generada a partir de los textos preprocesados como entrada para el modelo Ridge. El conjunto de datos se dividió en entrenamiento y prueba (80/20), asegurando la aleatoriedad mediante una semilla fija. El modelo se entrenó con una determinada tolerancia y se evaluó utilizando métricas estándar de regresión: error absoluto medio, error cuadrático medio, coeficiente de determinación y el score propio del modelo. Estos indicadores permiten cuantificar la precisión y la capacidad explicativa del modelo sobre datos no vistos.

### Word2Vec

Para la representación Word2Vec, cada receta se transformó en un vector denso calculando el promedio de los embeddings de sus palabras. Se verificó que todos los vectores tuvieran la misma dimensión y se identificaron posibles documentos sin representación válida. Tras dividir los datos en entrenamiento y prueba, se entrenó el modelo Ridge bajo las mismas condiciones que en el caso TF-IDF. Las métricas obtenidas permiten comparar la eficacia de los embeddings semánticos frente a las representaciones basadas en frecuencia de términos.

### BERT

En el caso de BERT, se emplearon los embeddings del token `[CLS]` extraídos de los primeros 100 documentos, dada la limitación de recursos computacionales. Estos vectores, de mayor riqueza contextual, se utilizaron como entrada para el modelo Ridge, repitiendo el proceso de división, entrenamiento y evaluación. Las métricas obtenidas reflejan la capacidad de los embeddings contextuales para capturar matices complejos del lenguaje y su impacto en la predicción de la puntuación de las recetas.

### Evaluación y comparación

En cada uno de los tres enfoques, el modelo Ridge se evaluó mediante las siguientes métricas:
MAE: mide el error promedio absoluto entre las predicciones y los valores reales.
MSE: cuantifica el error cuadrático medio, penalizando más los errores grandes.
R²: indica la proporción de la varianza explicada por el modelo.
Score: valor propio del modelo.

La comparación de estos resultados permite analizar el impacto de cada técnica de vectorización en la capacidad predictiva del modelo Ridge y seleccionar la estrategia más adecuada para el problema de regresión sobre recetas de cocina.

## Fine-Tuning de un Modelo Transformer con Hugging Face

Se ha realizado un fine-tuning de BERT utilizando la librería Hugging Face Transformers. Este proceso permite adaptar un modelo generalista a la tarea específica de regresión sobre textos de recetas, optimizando su rendimiento para el problema planteado.

El procedimiento comenzó seleccionando un subconjunto de datos, concretamente las primeras diez recetas procesadas, y extrayendo tanto las instrucciones (`directions`) como las puntuaciones (`rating`). Los datos se dividieron en conjuntos de entrenamiento y prueba, asegurando la aleatoriedad y la correcta alineación de los índices para evitar inconsistencias en las etiquetas.

A continuación, se utilizó el tokenizador `BertTokenizer` para convertir los textos en secuencias de tokens compatibles con el modelo BERT. Este paso incluyó la truncación y el padding de los textos a una longitud máxima de 128 tokens, garantizando que todos los ejemplos tuvieran el mismo formato de entrada.

Para facilitar el entrenamiento, se definió una clase personalizada `RecipeDataset`, que estructura los datos y etiquetas en el formato requerido por PyTorch. Esta clase permite acceder de manera eficiente a los ejemplos durante el proceso de entrenamiento y evaluación.

El modelo seleccionado fue `BertForSequenceClassification` con una única salida, adaptado para tareas de regresión. Se configuraron los argumentos de entrenamiento mediante la clase `TrainingArguments`, especificando el número de épocas, el tamaño de los lotes, los pasos de calentamiento, el decaimiento del peso y la estrategia de evaluación.

El entrenamiento se llevó a cabo utilizando la clase `Trainer` de Hugging Face, que gestiona automáticamente el ciclo de entrenamiento, la evaluación periódica y el registro de métricas. Tras completar las épocas de entrenamiento, el modelo se evaluó sobre el conjunto de prueba para obtener una estimación objetiva de su rendimiento en datos no vistos.

## Extensiones

Se han implementado diversas técnicas avanzadas de procesamiento de lenguaje natural y generación de texto, ampliando el alcance del proyecto más allá de los requisitos básicos. Estas extensiones permiten profundizar en el análisis de los datos, enriquecer la interpretación de los resultados y experimentar con modelos generativos de última generación.

### 1. Resumen automático de instrucciones

Se ha implementado un sistema de resumen automático utilizando el pipeline de Hugging Face y el modelo preentrenado `facebook/bart-large-cnn`. Dado que las instrucciones de las recetas pueden ser extensas y contener pasos repetidos, el texto se divide en fragmentos de longitud adecuada y cada uno se resume individualmente. Los resúmenes parciales se combinan para obtener una versión condensada y coherente de las instrucciones originales, facilitando la comprensión rápida de los pasos principales de cada receta.

### 2. Análisis de bigramas en las instrucciones

Para identificar patrones frecuentes en la redacción de las recetas, se ha realizado un análisis de bigramas sobre el campo `directions`. Tras tokenizar y extraer todos los pares de palabras consecutivas, se calcularon las frecuencias y se identificaron los diez bigramas más comunes. Este análisis revela combinaciones de acciones y términos recurrentes en la cocina, proporcionando información útil sobre la estructura típica de las instrucciones culinarias.

### 3. Etiquetado gramatical (Part-of-Speech Tagging)

Se ha aplicado el etiquetado gramatical a las instrucciones de las recetas utilizando las herramientas de NLTK. Este proceso asigna una categoría gramatical (sustantivo, verbo, adjetivo, etc.) a cada palabra, permitiendo analizar la composición sintáctica de las instrucciones. El etiquetado se ha realizado tanto sobre ejemplos individuales como sobre múltiples instrucciones.

### 4. Búsqueda de sinónimos con tesauros

Se ha incorporado la búsqueda de sinónimos mediante el tesauro WordNet de NLTK. Esta funcionalidad permite identificar alternativas léxicas para palabras clave presentes en las recetas, lo que puede ser útil para tareas de normalización, expansión de vocabulario o generación de variantes de instrucciones. El sistema soporta tanto inglés como otros idiomas.

### 5. Generación de recetas con modelos tipo transformer

Se ha experimentado con la generación automática de recetas utilizando modelos de lenguaje generativo como GPT-2 y Llama. A partir de un prompt inicial, estos modelos son capaces de producir instrucciones completas y coherentes para nuevas recetas. El proceso incluye la codificación del prompt, la generación controlada del texto y la decodificación del resultado. Además, se ha comparado la generación con diferentes modelos, evaluando la creatividad y la calidad de las recetas producidas.

## Diferencias y mejoras entre la convocatoria ordinaria y la extraordinaria

Se han mejorado distintos aspectos en comparación con el trabajo entregado en la convocatoria ordinaria atendiendo a las indicaciones que los profesores hicieron durante la defensa del trabajo. Las diversas mejoras se detallan a continuación.

### 1. Estructura y organización

El README presenta una estructura mucho más elaborada y detallada, con secciones claramente diferenciadas para cada etapa del proyecto: introducción, análisis de categorías, procesamiento de datos, técnicas de vectorización, entrenamiento de modelos, evaluación y extensiones. 

### 2. Profundidad en la explicación de cada apartado

Se ha utilizado una redacción más clara, precisa y profesional, con explicaciones completas y un lenguaje técnico adecuado. Cada sección describe de manera precisa cómo se ha implementado cada técnica, incluyendo explicaciones sobre el preprocesamiento, la construcción de los modelos y la interpretación de los resultados. Se explican los pasos seguidos, las librerías utilizadas y los criterios de evaluación.

### 3. Inclusión de ejemplos y resultados

El README incorpora ejemplos concretos de resultados obtenidos, como la visualización de las categorías mejor valoradas, la representación de palabras relevantes mediante TF-IDF, la comparación de métricas de rendimiento entre modelos y la inclusión de gráficos y análisis visuales.

### 4. Ampliación de las extensiones

Además del resumen automático de instrucciones, se han añadido análisis de bigramas, etiquetado gramatical, búsqueda de sinónimos y generación automática de recetas con modelos generativos. Estas extensiones demuestran que se ha ampliado el alcance del proyecto más allá de los requisitos mínimos.

### 5. Evaluación crítica y comparativa

Se realiza un análisis crítico de los resultados, comparando el rendimiento de las distintas técnicas y modelos, y justificando la elección de las mejores estrategias. Se discuten las limitaciones y se proponen posibles mejoras.
