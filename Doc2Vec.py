# -*- coding: utf-8 -*-
import pandas as pd
import re
import string
from nltk.tokenize import word_tokenize
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


class Doc2Vec:
    def __init__(self, hyperparametros):
        self.__hyperparametros = hyperparametros
    
    def setPathDatosTrain (self, strPath):
        self.__strPathDatosTrain = strPath   

    def setPathDatosCoste (self, strPath):
        self.__strPathDatosCoste = strPath
    
    #TODO esta función no esta limpiaando bien
    def __cleanText(self, text):
        text = text.lower()
        text = re.sub(r'\d+', '', text)
        
        translator = str.maketrans('', '', string.punctuation)
        text = text.translate(translator)

        return text

    def train(self):
        print("Entrenando con los datos de " + self.__strPathDatosTrain)
        recursos = pd.read_csv(self.__strPathDatosTrain, sep=';')
        
        train_tagged = recursos.apply(lambda r: TaggedDocument(words=word_tokenize(self.__cleanText(r['sentence'])), tags=r.Polarida), axis=1)

        print (train_tagged.values[30])
        #TODO  y entrenar el doc2vec
        