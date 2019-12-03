# -*- coding: utf-8 -*-
from datetime import datetime
from HyperparametersDoc2Vec import HyperparametersDoc2Vec
from Doc2Vec import Doc2Vec

class SA:
    def __init__(self, T0, LT, enfriamiento, strPathTrain, strPathTest, strPathSalidad):
        self.__T0 = T0
        self.__LT = LT
        self.__enfriamiento = enfriamiento
        self.__strPathTrain = strPathTrain
        self.__strPathTest = strPathTest
        self.__strPathSalida = strPathSalidad
    
    def run(self):
        T = self.__T0
        
        print ('[', datetime.now().strftime("%d/%m/%Y %H:%M:%S"),'] SA.run: generarndo primer elemento aleatorio')
    
        lstConfiguraciones = HyperparametersDoc2Vec.generarAleatorios(1, False)
        print ('[', datetime.now().strftime("%d/%m/%Y %H:%M:%S"),'] SA.run: generado primer elemento aleatorio')
        print (lstConfiguraciones[0].toString())

        #CSe crea la intancia de doc2vec desde el candidato generado y se entrene
        S_act = Doc2Vec(lstConfiguraciones[0])
        S_act.setPathDatosTrain(self.__strPathTrain)
        S_act.setPathDatosCoste(self.__strPathTest)
        S_act.train()