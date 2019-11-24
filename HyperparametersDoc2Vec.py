# -*- coding: utf-8 -*-
from random import seed
from random import randint
from random import random

class HyperparametersDoc2Vec:
    __hsGenerados = set()
    
    #En la creación existe un otro parametro para limitar el número de palabras
    #del volcabulario, quizas peuda interesar optimizarlos.
    #Tambien existe parametro para modo ‘distributed memory’ o 'bag of words', en 
    #java esto no se tenia en cuenta y se usaba simepre 'distributed memory.
    #Existen mas parametros para la configuración del PV, como si se quiere media o suma
    #de lso vectores, reglas sobre el vocabulario,... esto no se tenai en cuneta en java
    #Link a la documentación: https://radimrehurek.com/gensim/models/doc2vec.html
    
    def __init__(self, epochs, layerSize, learningRate, minLearningRate, minWordFrecuency, windowsSize):
        self.epochs = epochs
        self.layerSize = layerSize
        self.learningRate = learningRate
        self.minLearningRate =minLearningRate
        self.minWordFrecueny = minWordFrecuency
        self.widowsSize = windowsSize
    
    def __key(self):
        return str(self.epochs) + "_" + str(self.layerSize) + "_" + str(self.learningRate) + "_" + str(self.minLearningRate) + "_" + str(self.minWordFrecueny) + "_" + str(self.widowsSize)
    
    @staticmethod
    def __generarAleatorioINT ( min, max):
        return randint(min, max);
    
    @staticmethod
    def __generarAleatorioFloat(min, max):
        return min + (random() * (max - min))
    
    @staticmethod
    def generarAleatorios (intEltos,permitirRepetidos):
        lstRes = []

        while (len(lstRes) < intEltos):
            epochs = HyperparametersDoc2Vec.__generarAleatorioINT(1, 500);
            #epochs = HyperparametersDoc2Vec.__generarAleatorio(1, 2);
            layerSize = HyperparametersDoc2Vec.__generarAleatorioINT(1, 500);
            learningRate = HyperparametersDoc2Vec.__generarAleatorioFloat(0.1e-6, 0.1);
            minLearningRate = HyperparametersDoc2Vec.__generarAleatorioFloat(0.1e-6, learningRate);
            minWordFrecuency = HyperparametersDoc2Vec.__generarAleatorioINT(1, 100);
            windowsSize = HyperparametersDoc2Vec.__generarAleatorioINT(1, 125);

            elto = HyperparametersDoc2Vec(epochs, layerSize,learningRate, minLearningRate, minWordFrecuency, windowsSize)
            

            if (permitirRepetidos == False):
                if ((elto.__key() in HyperparametersDoc2Vec.__hsGenerados) == False):
                    lstRes.append(elto);
                    HyperparametersDoc2Vec.__hsGenerados.add(elto.__key())
            else:
                lstRes.add(elto)

        return lstRes; 
    
    def toCSV (self):
        
        return str(self.epochs) + ";" + str(self.layerSize) + ";" + str(self.learningRate) + ";" + str(self.minLearningRate) + ";" + str(self.minWordFrecueny) + ";" + str(self.widowsSize)

    def toString (self):
        return "epochs=" + str(self.epochs) + ", layerSize=" + str(self.layerSize) + ", learningRate=" + str(self.learningRate) + ", minLearningRate=" + str(self.minLearningRate) + ", minWordFrecueny=" + str(self.minWordFrecueny) + ", widowsSize=" + str(self.widowsSize)
    
