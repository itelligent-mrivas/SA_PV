# -*- coding: utf-8 -*-
import pandas as pd
import re
import string
import multiprocessing
from nltk.tokenize import word_tokenize
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.linear_model import LogisticRegression


class Doc2VecBuilder:
    def __init__(self, hyperparametros):
        self.__hyperparametros = hyperparametros
    
    def setPathDatosTrain (self, strPath):
        self.__strPathDatosTrain = strPath   

    def setPathDatosCoste (self, strPath):
        self.__strPathDatosCoste = strPath
    
    def __cleanText(self, text):
        text = text.lower()
        text = re.sub(r'\d+', '', text)
        
        translator = str.maketrans('', '', string.punctuation)
        text = text.translate(translator)

        return text

    def train(self):
        print("Entrenando con los datos de " + self.__strPathDatosTrain)
        #Carga de los datos
        recursos = pd.read_csv(self.__strPathDatosTrain, sep=';')
        
        
        #Monto el tagged Documento con el que trabaja doc2ve. Pasandole los textos limpios y tokenizados
        # y como clase tags el valor de 
        train_tagged = recursos.apply(lambda r: TaggedDocument(words=word_tokenize(self.__cleanText(r['sentence'])), tags=r["polaridad"]), axis=1)

        #Miro el numero de procesadores de la maquina para usarlos en el entrenamiento
        cores = multiprocessing.cpu_count()
        print("Numero de procesadores detectados para el entrenamiento ", cores)
        
        self.__modelDoc2vev = Doc2Vec(vector_size= self.__hyperparametros.layerSize, window=self.__hyperparametros.widowsSize, min_count=self.__hyperparametros.minWordFrecueny, workers=cores, alpha=self.__hyperparametros.learningRate, min_alpha=self.__hyperparametros.minLearningRate, epochs= self.__hyperparametros.epochs)
        self.__modelDoc2vev.build_vocab([x for x in train_tagged.values])
        self.__modelDoc2vev.train(train_tagged, epochs=self.__modelDoc2vev.epochs, total_examples=self.__modelDoc2vev.corpus_count)
        
        #Una vez entrenado el Doc2Vec se entrena el clasificados. Para ello primero monto xTrain con 
        #los vectores de los documentos train, y yTrain con las clases de train
        yTrain = recursos["polaridad"]
        xTrain = []
        for i in range(len(self.__modelDoc2vev.docvecs)):
            xTrain.append(self.__modelDoc2vev.docvecs[i].tolist())
        
        #Entrenamiento del clasificador
        self.__clasificador = LogisticRegression(n_jobs=cores)
        self.__clasificador.fit(xTrain, yTrain)

    # def coste():
    #     #Entrenar el clasificador con los vectores de train y validarlo con los de test
        
    #     xTrain = []
    #     for i in len(self.__model.docvecs):
    #         xTrain.append(model.docvecs[i].tolist())

    #     cores = multiprocessing.cpu_count()
    #     clasificador = LogisticRegression(n_jobs=cores)
    #     clasificador.fit(xTrain)