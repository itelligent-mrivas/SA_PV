# SST (Stanford Sentiment Treebank)
**Fuente** https://nlp.stanford.edu/sentiment/

Datos etiquetados con análisis del sentitmiento.
## raw.xls
Contiene todos los datos del data set. Los campos contenidos son:
1. **sentence_index.** Identificador del elemento
2. **sentence.** Texto del comentario
3. **splitset_label.** Indica la partición del conjunto que puede ser: 1 = train, 2 = test y 	3 = dev

4. **sentiment values.** Valores para el analis del sentimiento. Se podría dividir en 5 conjutnos de la siguiente manera: 0, 0.2], (0.2, 0.4], (0.4, 0.6], (0.6, 0.8], (0.8, 1.0]
for very negative, negative, neutral, positive, very positive, respectively.

***
### NOTA

Estos datos no son usables directamente, se han utilizado para generar los datasets.
***

## Dataset (SST 2)
Corresponde al SST pero solo teniendo dos clases, cometarios positivos y negativos. Los comentarios negativos con aquello con un valor de sentimiento menor o igual que 0,4. Los comentarios positivos son aquellos con un valor de sentimeinto mayor que 0,6

Los datos se parten en tres conjunto: 
* train ([polaridad_binaria_train.csv](polaridad_binaria_train.csv))
* test ([Polaridad_binaria_test.csv](Polaridad_binaria_test.csv)) 
* validadción ([Polaridad_binaria_validación.csv](Polaridad_binaria_validación.csv)).

Los campos que tienen los ficheros son:
1. **sentence_index.** Identificador del elemento
2. **sentence.** Texto del comentario
3. **Polarida.** Polaridad del cometario, que peude tener los valores {"Negativo", "Positivo"}
