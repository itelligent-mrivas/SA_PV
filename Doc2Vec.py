# -*- coding: utf-8 -*-

class Doc2Vec:
    def __init__(self, hyperparametros):
        self.__hyperparametros = hyperparametros
    
    def setPathDatosTrain (self, strPath):
        self.__strPathDatosTrain = strPath   

    def setPathDatosCoste (self, strPath):
        self.__strPathDatosCoste = strPath
    
    def train(self):
        print("Entrenando con los datos de " + self.__strPathDatosTrain)