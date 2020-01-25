# -*- coding: utf-8 -*-
from datetime import datetime
from HyperparametersDoc2Vec import HyperparametersDoc2Vec
from Doc2VecBuilder import Doc2VecBuilder
from random import random
import math
import os

class SA:
    def __init__(self, T0, LT, enfriamiento, strPathTrain, strPathTest, strPathSalidad, modificador):
        self.__T0 = T0
        self.__LT = LT
        self.__enfriamiento = enfriamiento
        self.__strPathTrain = strPathTrain
        self.__strPathTest = strPathTest
        self.__strPathSalida = strPathSalidad

        modificador = modificador.lower()
        if(modificador=="aleatorio" or modificador == "uniforme" or modificador == "no_uniforme"):
            self.__modificador = modificador
        else:
            raise Exception("Modifcado " + modificador + " invalido. Debe ser: aleatorio, uniforme o no_uniforme")

        self.__colaCostes = []
        self.__id_solucion = 0
    
    def __inicializa_fichero(self):
        #Creo la carpeta global
        strPath = self.__strPathSalida + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        os.mkdir(strPath)
        
        #Creo la carpeta para modelos y guardo su path para luego poder guardar los modelos
        self.__pathModelos = strPath + "/modelos/"
        os.mkdir(self.__pathModelos)

        #inicializo el fichero de resultados
        self.__file = open(strPath+ '/resultados.csv', 'w')
        self.__file.write("id solucion;epochs;layerSize;learningRate;minLearningRate;minWordFrecueny;widowsSize;coste\n")
        self.__file.flush()
    
    def __escribe_solucion(self, id, hyperparametros, coste):
        self.__file.write(str(id) + ";"+ hyperparametros.toCSV()+";"+str(coste)+ "\n")
        self.__file.flush()
    
    def __generarId(self):
        self.__id_solucion = self.__id_solucion +1
        return self.__id_solucion
    
    def run(self):
        T = self.__T0
        
        self.__inicializa_fichero()

        print ('[', datetime.now().strftime("%d/%m/%Y %H:%M:%S"),'] SA.run: generarndo primer elemento aleatorio')
    
        lstConfiguraciones = HyperparametersDoc2Vec.generarAleatorios(1, False)
        print ('[', datetime.now().strftime("%d/%m/%Y %H:%M:%S"),'] SA.run: generado primer elemento aleatorio')
        print (lstConfiguraciones[0].toString())

        #CSe crea la intancia de doc2vec desde el candidato generado y se entrene
        S_act = Doc2VecBuilder(lstConfiguraciones[0])
        S_act.setPathDatosTrain(self.__strPathTrain)
        S_act.setPathDatosCoste(self.__strPathTest)
        S_act.train()

        #obtengo el coste de la solucion, este al ser el primero sera el mejor coste y al mejor
        #solución hasta el momento
        print ('[', datetime.now().strftime("%d/%m/%Y %H:%M:%S"),'] SA.run: coste de la primera solución aleatoria')

        dblCoste_act =S_act.coste()
        dblCosteMejor = dblCoste_act
        mejorSolucion = S_act

        # guardo el modelo y la slucion
        id_solucion = self.__generarId()
        self.__escribe_solucion(id_solucion, lstConfiguraciones[0], dblCoste_act)
        S_act.guardarModelo(self.__pathModelos, id_solucion)

        blnConitnuar = True
        while(blnConitnuar):
            print ('[', datetime.now().strftime("%d/%m/%Y %H:%M:%S"),'] Temperatura ', T)

            configuracion = None
            if self.__modificador == "aleatorio":
                configuracion = HyperparametersDoc2Vec.generarAleatorios(self.__LT, False)
            elif self.__modificador == "uniforme":
                configuracion = HyperparametersDoc2Vec.generarUniforme(S_act.getParametros(), self.__LT, False)
            elif self.__modificador == "no_uniforme":
                configuracion = HyperparametersDoc2Vec.generarNoUniforme(S_act.getParametros(), T, self.__LT, False)

            S_Cand = Doc2VecBuilder(configuracion[0])
            S_Cand.setPathDatosTrain(self.__strPathTrain)
            S_Cand.setPathDatosCoste(self.__strPathTest)
            
            S_Cand.train()
            dblCoste_Cand = S_Cand.coste()

            print ('[', datetime.now().strftime("%d/%m/%Y %H:%M:%S"),'] Modelo ', S_Cand.getParametros().toCSV(), " Coste ", dblCoste_Cand)

            #guardo solución y modelos
            id_solucion = self.__generarId()
            self.__escribe_solucion(id_solucion, S_Cand.getParametros(), dblCoste_Cand)
            S_Cand.guardarModelo(self.__pathModelos, id_solucion)

            difCoste = dblCoste_Cand - dblCoste_act
            porcentajeMejora = 0

            #Si el candidato es mejor que el actual nos quedamso con el
            if (difCoste < 0) :
                porcentajeMejora = abs(difCoste)/dblCoste_act

                S_act = S_Cand
                dblCoste_act = dblCoste_Cand

                # Miro si mejora la mejor solucion hasta el momento
                if (dblCoste_act < dblCosteMejor) :
                    dblCosteMejor = dblCoste_act
                    mejorSolucion = S_act

                    print ('[', datetime.now().strftime("%d/%m/%Y %H:%M:%S"),'] \tCandaidto anceptado, coste: ', dblCoste_Cand)
            else :
                #Si no es mejor calculamso la probabilidad de aceptación
                dblPropabilidadAceptacion = math.pow(math.e, - difCoste / T)
                if (random() < dblPropabilidadAceptacion) :
                    S_act = S_Cand
                    dblCoste_act = dblCoste_Cand

                    print ('[', datetime.now().strftime("%d/%m/%Y %H:%M:%S"),'] \tCandaidto anceptado por porb, coste: ', dblCoste_act)

            # Actualizo la temperatura
            T = self.__enfriamiento * T

            #Compruebo criterio de parada
            blnConitnuar = self.paradaKirkpatric(porcentajeMejora)
        
        #FIND DE LA OPTIMIZACIÓN, GUARDAR RESULTADOS FINALES
        print('\n\n--------------------------------------------------------------------\n')
        print('********* Fin de la optimización ********************************')
        print('Mejor solución: ' + mejorSolucion.getParametros().toCSV())
        print('Coste de la mejor soluciín ', dblCosteMejor)

    def paradaKirkpatric(self, coste):
        if len(self.__colaCostes) < 4 :
            self.__colaCostes.append(coste)
            return True
        
        self.__colaCostes.pop(0)
        self.__colaCostes.append(coste)

        blnContinuar = False
        i = 0
        while(i < len(self.__colaCostes) and blnContinuar == False):
            valor = self.__colaCostes[i]
            if(valor >0.02):
                blnContinuar=True
            i = i+1
        return blnContinuar
        


