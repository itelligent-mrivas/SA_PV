# -*- coding: utf-8 -*-
import pandas as pd
import re
import string
import multiprocessing
from nltk.tokenize import word_tokenize
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


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
        
        model = Doc2Vec(vector_size= self.__hyperparametros.layerSize, window=self.__hyperparametros.widowsSize, min_count=self.__hyperparametros.minWordFrecueny, workers=cores, alpha=self.__hyperparametros.learningRate, min_alpha=self.__hyperparametros.minLearningRate, epochs= self.__hyperparametros.epochs)
        model.build_vocab(train_tagged)
        model.train(train_tagged, epochs=model.epochs, total_examples=model.corpus_count)
        
        #TODO entrenar el clasificador