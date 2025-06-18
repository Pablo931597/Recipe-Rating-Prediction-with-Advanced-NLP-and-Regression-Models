# Master_uc3m
Uc3m projects
Proyecto realizado por Emilio Muñoz Álvarez y Pablo Ebohon Serna

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


## Procesamiento de los textos de las recetas
Para poder emplear redes neuronales y algoritmos de regresión, tenemos que procesar el texto de manera que se transformen en datos que dichos algoritmos puedan procesr.
Para ello haremos uso de 3 tipos distintos de NLP: TF-IDF, Word2Vec y BERT.

### TF-IDF: 
Técnica en la que, mediante tokens, se le asigna un id y la frecuencia de la palabra. En este caso hemos decidido quedarnos con un vocabulario de 2000 palabras. 

### Word2Vec: 
Con este método transformamos las distintas palabras en vectores. En este proyecto transformamos en vectores de 500 dimensiones todo el texto contenido en cada receta.

### BERT:
Transformer al que inicialmente le pasamos el texto sin preprocesar. nos convierte dicho texto procesado en "embeddings".

Una vez procesado el texto, ya podemos utilizarlo para entrenar nuestros modelos. Dicho conjunto de entrenamiento constará únicamente de los textos procesados, 
pues el rendiminento de los modelos con las características numéricas tales como grasa, sodio, etc, no fue significativamente superior, y debido a que usamos el entorno de programación de Google Colab, 
optamos por reducir el nº de características a las de los textos.

## Entrenamiento y rendimiento red neuronal:
Realizamos el entrenamiento de la red neuronal con los 3 tipos de NLP, con número de épocas = 50.
Las prestaciones obtenidas fueron similares con los 3 tipos de NLP, siendo la que mejor rendimiento tuvo la entrenada con el transformer BERT (un MSE de 0.82 y R2 de -0.37).

## Entrenamiento modelo de regresión Ridge:
Para el modelo de regresión nos basamos en el regresor lineal Ridge.
Las prestaciones obtenidas tambien fueron similares (sobretodo TF-IDF y W2V), siendo la mejor otra vez la obtenida en BERT, con MSE=0.88 y R2=-0.48.

La conclusión final fue que las mejores prestaciones fueron las de la red neuronal utilizando BERT,
aunque no es una mejoría muy significativa con respecto al Ridge utilizando el mismo conjunto de entrenamiento.

También se ha hecho uso del fine tuning del Hugging Face, comparando el rendimiento del BERT no usándolo. Éste realiza un ajuste de nuestro conjunto de texto (al ser un modelo preentrenado con un conjunto extenso de datos)
para nuestra tarea específica (en este caso, recetas de cocina). Las prestaciones fueron mejores



