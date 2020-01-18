# -*- coding: utf-8 -*-
import sys
from SA import SA

if (len(sys.argv) != 4):
    print ('Se requieren tres parameros, se recibe, ', len(sys.argv), ';', str(sys.argv))
    sys.exit(1)


strPathTrain = str(sys.argv[1])
strPathTest = str(sys.argv[2])
strPathSalida = str(sys.argv[3])


print("Path train: " + strPathTrain)
print("Path test: " + strPathTest)
print("Salida: " + strPathSalida)

experimento =SA(3, 1, 0.9, strPathTrain, strPathTest, strPathSalida, "no_uniforme")
experimento.run()