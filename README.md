# SA_PV
Proyecto para la optimización de los hyperparamteros de paragraph vector usando simulated annealing

## Intalación
### Prerequisitos
Para la ejecución de este proyecto es necesario python3.

Los siguientes paquetes son requeridos:
* NLTK. Biblioteca con funcionalidades comunes de procesameinto de lenguaje natural. `pip3 install nltk`
* Gensim. Bibloteca que implementa Doc2Vec. `pip3 install gensim`
* Sklearn. Biblioteca de aprendizaje automático. `pip3 install sklearn`
* Pandas. Bibloteca para el manejo de CSV. `pip3 install pandas`

### Obtener código
Obtener el proyecto desde git clonando el repositorio

`git clone https://github.com/itelligent-mrivas/SA_PV.git`

## Ejecución
Para ejecutar un experimento usando el proyecto lanzar el siguiente comando

`python3 run.py "./datasets/SST/Polaridad_binaria_train.csv" "./datasets/SST/Polaridad_binaria_test.csv" "../salidas_SA_PV/"`

Dentro de este comando el primer parametro es el path donde se encuentran los datos de entrenamiento, el segudno parametro donde se encuentra los datos de test y el tercero donde se alamcenaran los datos de salida
