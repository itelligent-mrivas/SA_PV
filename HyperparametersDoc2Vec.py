# -*- coding: utf-8 -*-
from random import seed
from random import randint
from random import random
import math

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
        return randint(min, max)
    
    @staticmethod
    def __generarAleatorioFloat(min, max):
        return min + (random() * (max - min))
    
    @staticmethod
    def generarAleatorios (intEltos,permitirRepetidos=False):
        lstRes = []

        while (len(lstRes) < intEltos):
            #epochs = HyperparametersDoc2Vec.__generarAleatorioINT(1, 500)
            epochs = HyperparametersDoc2Vec.__generarAleatorioINT(1, 2)
            layerSize = HyperparametersDoc2Vec.__generarAleatorioINT(1, 500)
            learningRate = HyperparametersDoc2Vec.__generarAleatorioFloat(0.1e-6, 0.1)
            minLearningRate = HyperparametersDoc2Vec.__generarAleatorioFloat(0.1e-6, learningRate)
            minWordFrecuency = HyperparametersDoc2Vec.__generarAleatorioINT(1, 100)
            windowsSize = HyperparametersDoc2Vec.__generarAleatorioINT(1, 125)

            elto = HyperparametersDoc2Vec(epochs, layerSize,learningRate, minLearningRate, minWordFrecuency, windowsSize)
            

            if (permitirRepetidos == False):
                if ((elto.__key() in HyperparametersDoc2Vec.__hsGenerados) == False):
                    lstRes.append(elto)
                    HyperparametersDoc2Vec.__hsGenerados.add(elto.__key())
            else:
                lstRes.append(elto)

        return lstRes
    @staticmethod
    def generarUniforme(input, intEltos, permitirRepetidos=False):
        lstRes = []

        while (len(lstRes) < intEltos) :
            elto = input.__mutacionUniforme()
            if (permitirRepetidos == False):
                #Si no se ha pasado ya por este elemento se tiene en cuenta, sino se decarta
                if ((elto.key() in HyperparametersDoc2Vec.__hsGenerados) == False):
                    lstRes.append(elto)
                    HyperparametersDoc2Vec.__hsGenerados.add(elto.__key())
            else :
                lstRes.append(elto)

        return lstRes
    
    def __mutacionUniforme(self):
        

        intCampoMutar = HyperparametersDoc2Vec.__generarAleatorioINT(0, 5)
        
        mutado = HyperparametersDoc2Vec(self.epochs, self.layerSize, self.learningRate, self.minLearningRate, self.minWordFrecueny, self.widowsSize)

        if intCampoMutar == 0:
            mutado.epochs = HyperparametersDoc2Vec.__generarAleatorioINT(1, 500)
            #mutado.epochs = hyperparametersDoc2Vec.generarAleatorio(1, 2);
        elif intCampoMutar == 1:
            mutado.layerSize = HyperparametersDoc2Vec.__generarAleatorioINT(1, 500)
        elif intCampoMutar == 2:
            mutado.learningRate = HyperparametersDoc2Vec.__generarAleatorioFloat(0.1e-6, 0.1)
        elif intCampoMutar == 3:
            mutado.minLearningRate = HyperparametersDoc2Vec.__generarAleatorioFloat(0.1e-6, mutado.learningRate)
        elif intCampoMutar == 4:
            mutado.minWordFrecueny = HyperparametersDoc2Vec.__generarAleatorioINT(1, 100)
        elif intCampoMutar == 5:
            mutado.widowsSize = HyperparametersDoc2Vec.__generarAleatorioINT(1, 125)
        
        return mutado

    @staticmethod
    def generarNoUniforme (input, T, intEltos, permitirRepetidos=False):
        lstRes = []

        while (len(lstRes) < intEltos):
            elto = input.__mutacionNoUniforme(T)
            
            if (permitirRepetidos == False):
                #Si no se ha pasado ya por este elemento se tiene en cuenta, sino se decarta
                if ((elto.key() in HyperparametersDoc2Vec.__hsGenerados) == False):
                    lstRes.append(elto)
                    HyperparametersDoc2Vec.__hsGenerados.add(elto.__key())
            else :
                lstRes.append(elto)

        return lstRes
    
    def __mutacionNoUniforme (self, T) :
        intCampoMutar = HyperparametersDoc2Vec.__generarAleatorioINT(0,5)

        mutado = HyperparametersDoc2Vec(self.epochs, self.layerSize, self.learningRate, self.minLearningRate, self.minWordFrecueny, self.widowsSize)

        if intCampoMutar == 0:
            mutado.epochs = self.__generaNoUniformeINT(self.epochs, 1, 500, T)
            #mutado.epochs = self.__generaNoUniformeINT(mutado.epochs, 1, 2, T)
        elif intCampoMutar == 1:
            mutado.layerSize = self.__generaNoUniformeINT(mutado.layerSize, 1, 500, T)
        elif intCampoMutar == 2:
            mutado.learningRate = self.__generaNoUniformeFLOAT(mutado.learningRate, 0.1e-6, 0.1, T)
        elif intCampoMutar == 3:
            mutado.minLearningRate = self.__generaNoUniformeFLOAT(mutado.minLearningRate, 0.1e-6, mutado.learningRate, T)
        elif intCampoMutar == 4:
            mutado.minWordFrecueny = self.__generaNoUniformeINT(mutado.minWordFrecueny, 1, 100, T)
        elif intCampoMutar == 5:
            mutado.widowsSize = self.__generaNoUniformeINT(mutado.widowsSize, 1, 125, T)

        return mutado
    
    def __generaNoUniformeINT(self, intElto, LI, LS, T):
        intRes=None

        intRuleta = None
        if (intElto == LS): # Si ya estamso en el Limite supersiso forzamso que reste
            intRuleta = 10000
        elif (intElto == LI): # Si ya estamos en el limite inferior, forzamso que sume
            intRuleta = 0
        else :
            intRuleta = HyperparametersDoc2Vec.__generarAleatorioINT(0, 10000)

        if (intRuleta < 5000) :
            intIncremento = 0
            intVuelta = 0
            while True :
                intIncremento = self.__incrementoINT(T, LS - intElto)

                if (intIncremento == 0 and (LS - intElto) == 1):
                    intIncremento = 1
                
                if (intVuelta > 10):
                    print("1 DEMASIADAS VUELTAS!!!!")
                intVuelta = intVuelta+1
                if intIncremento != 0:
                    break

            intRes = intElto + intIncremento

            if (intRes > LS):
                intRes = LS

        else:
            intIncremento = 0
            intVuelta = 0
            while True:
                intIncremento = self.__incrementoINT(T, intElto - LI)

                if (intIncremento == 0 and (intElto - LI) == 1):
                    intIncremento = 1
                if (intVuelta > 10):
                    print("2 DEMASIADAS VUELTAS!!!!")
                
                intVuelta = intVuelta+1
                if(intIncremento != 0):
                    break

            intRes = intElto - intIncremento
            if (intRes < LI):
                intRes = LI

        return intRes

    def __incrementoINT (self, t, y) :
        exponente = math.pow(math.pow(math.e, -1 / t), 5)
        aleatorio = HyperparametersDoc2Vec.__generarAleatorioFloat(0.0, 1.0)
        res = (int) (y * (1 - math.pow(aleatorio, exponente)))

        return res

    def __generaNoUniformeFLOAT (self, intElto,  LI,  LS,  T):
        intRuleta = None
        dblRes = None

        if (intElto == LS) : # Si ya estamso en el Limite supersiso forzamos que reste
            intRuleta = 10000
        elif (intElto == LI) : # Si ya estamos en el limite inferior, forzamso que sume
            intRuleta = 0
        else :
            intRuleta = HyperparametersDoc2Vec.__generarAleatorioINT(0, 10000)

        if (intRuleta < 5000):
            dblIncremento = 0
            intVuelta = 0
            
            while True :
                dblIncremento = self.__incrementoFLOAT(T, LS - intElto)

                if (intVuelta > 10) :
                    print("3 DEMASIADAS VUELTAS!!!!")
                
                intVuelta = intVuelta + 1
                if dblIncremento != 0:
                    break

            dblRes = intElto + dblIncremento

            if (dblRes > LS) :
                dblRes = LS

        else :
            dblIncremento = 0
            intVuelta = 0
            while True:
                dblIncremento = self.__incrementoFLOAT(T, intElto - LI)

                if (intVuelta > 10) :
                    print("4 DEMASIADAS VUELTAS!!!!")
                
                intVuelta = intVuelta +1
                if (dblIncremento != 0):
                    break

            dblRes = intElto - dblIncremento
            if (dblRes < LI) :
                dblRes = LI

        return dblRes
    
    def __incrementoFLOAT (self, t, y) :
        exponente = math.pow(math.pow(math.e, -1 / t), 5)
        aleatorio = HyperparametersDoc2Vec.__generarAleatorioFloat(0.0, 1.0)
        res = y * (1 - math.pow(aleatorio, exponente))

        return res

    def toCSV (self):
        
        return str(self.epochs) + ";" + str(self.layerSize) + ";" + str(self.learningRate) + ";" + str(self.minLearningRate) + ";" + str(self.minWordFrecueny) + ";" + str(self.widowsSize)

    def toString (self):
        return "epochs=" + str(self.epochs) + ", layerSize=" + str(self.layerSize) + ", learningRate=" + str(self.learningRate) + ", minLearningRate=" + str(self.minLearningRate) + ", minWordFrecueny=" + str(self.minWordFrecueny) + ", widowsSize=" + str(self.widowsSize)
    
