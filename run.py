# -*- coding: utf-8 -*-
import sys
from SA import SA

if (len(sys.argv) != 3):
    print ('Se requieren tres parameros, se recibe, ', len(sys.argv), ';', str(sys.argv))
    #sys.exit(1)


strPathTrain = str(sys.argv[1])
strPathTest = str(sys.argv[2])
strPathSalida = str(sys.argv[3])


print("Path train: " + strPathTrain)
print("Path test: " + strPathTest)
print("Salida: " + strPathSalida)

experimento =SA(3, 20, 0.9, strPathTrain, strPathTest, strPathSalida)
experimento.run()