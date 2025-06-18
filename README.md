# Master_uc3m
Uc3m projects
Proyecto realizado por Emilio Muñoz Álvarez y Pablo Ebohon Serna

El tema principal del proyecto trata sobre un análisis de recetas de cocina y posterior predicción de sus valoraciones con modelos entrenados basados en algoritmos de regresión y redes neuronales,
para comparar el rendimiento de cada una en base a sus resultados.

Profundizando más en dicho proyecto, el conjunto de datos que vamos a tratar es un fichero JSON que almacena 20130 recetas distintas,
en las que se detallaan una serie de caracterísiticas que son: título, ingredientes, categoría, receta, calorías, grasa, sodio y fecha de publicación.
Algunas de las características como título, ingredientes, receta y categoría, contienen texto tales como frases o palabras concretas, a diferencia del resto de características que son numéricas.
Es por ello que hay que realizar técnicas de procesamiento de lenguaje natural (NLP), para poder utilizarlo como conjunto de entrenamiento tanto en las redes neuronales como algoritmos de regresión.

## Análisis de las categorías
Realizamos una breve comparación de algunas de las categorías, donde vemos su valoración media obtenida (comparaciñon sobre 5 o 6 categorías distintas).

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



