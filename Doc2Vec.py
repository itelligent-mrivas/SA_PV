# -*- coding: utf-8 -*-
import pandas as pd
from bs4 import BeautifulSoup
import re


class Doc2Vec:
    def __init__(self, hyperparametros):
        self.__hyperparametros = hyperparametros
    
    def setPathDatosTrain (self, strPath):
        self.__strPathDatosTrain = strPath   

    def setPathDatosCoste (self, strPath):
        self.__strPathDatosCoste = strPath
    
    #TODO esta función no esta limpiaando bien
    def __cleanText(self, text):
        text = BeautifulSoup(text, "lxml").text
        text = re.sub(r'\|\|\|', r' ', text) 
        text = re.sub(r'http\S+', r'<URL>', text)
        text = text.lower()
        text = text.replace('x', '')
        return text

    def train(self):
        print("Entrenando con los datos de " + self.__strPathDatosTrain)
        recursos = pd.read_csv(self.__strPathDatosTrain, sep=';')
        print(recursos.iloc[0][1])
        print("------------------------")
        print(self.__cleanText(recursos.iloc[0][1]))