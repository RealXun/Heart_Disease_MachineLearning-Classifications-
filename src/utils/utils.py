# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd
import os
import sys
from utils import utils

# visualization
import matplotlib.pyplot as plt
import seaborn as sns

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB,BernoulliNB, MultinomialNB


## scikit modeling libraries
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

## Load metrics for predictive modeling
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc

## Warnings and other tools
import warnings
warnings.filterwarnings("ignore")

#Preparación de las variables de carpeta
os.chdir(os.path.dirname(sys.path[0])) # Este comando convierte el cuaderno en la ruta principal y puede trabajar en cascada.
carpeta_principal = sys.path[0]
carpeta_datos = (carpeta_principal + "\data")



'''
Gaussian Naive Bayes
'''
def gaussian_naive_bayes(df,size):
  X = df.drop(["target"],axis=1)
  y = df["target"]
  X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=size,random_state=0)
  # llamamos al modelo
  heart_Gaussian = GaussianNB()
  #hacemos el fit
  heart_Gaussian.fit(X_train, y_train)
  # Creamos la predicción
  y_pred_heart_Gaussian = heart_Gaussian.predict(X_test)
  # print(classification_report(y_true=y_test,y_pred=y_pred_heart_Gaussian))
  #Obtenemos los datos de accuracy_score
  return gaussian_naive_bayes.__name__,accuracy_score(y_test, y_pred_heart_Gaussian),heart_Gaussian.get_params(deep=True).items()


'''
Bernoulli Naive Bayes
'''
def bernoulli(df,size):
  X = df.drop(["target"],axis=1)
  y = df["target"]
  X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=size,random_state=0)
  # llamamos al modelo
  heart_Bernoulli = BernoulliNB()
  #hacemos el fit
  heart_Bernoulli.fit(X_train, y_train)
  # Creamos la predicción
  y_pred_heart_Bernoulli = heart_Bernoulli.predict(X_test)
  # print(classification_report(y_true=y_test,y_pred=y_pred_heart_Bernoulli))
  #Obtenemos los datos de accuracy_score
  return bernoulli.__name__, accuracy_score(y_test, y_pred_heart_Bernoulli),heart_Bernoulli.get_params(deep=True).items()


'''
Multinomial Naive Bayes
'''
def multinomial(df,size):
  X = df.drop(["target"],axis=1)
  y = df["target"]
  X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=size,random_state=0)
  # llamamos al modelo
  heart_Multinomial = MultinomialNB()
  #hacemos el fit
  heart_Multinomial.fit(X_train, y_train)
  # Creamos la predicción
  y_pred_heart_Multinomial = heart_Multinomial.predict(X_test)
  # print(classification_report(y_true=y_test,y_pred=y_pred_heart_Multinomial))
  #Obtenemos los datos de accuracy_score
  return multinomial.__name__, accuracy_score(y_test,y_pred_heart_Multinomial),heart_Multinomial.get_params(deep=True).items()



'''
Random Forest
'''
def RandomForest(df,size):
  X = df.drop(["target"],axis=1)
  y = df["target"]
  X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=size,random_state=0)
  # llamamos al modelo
  heart_RandomForestClassifier = RandomForestClassifier()
  #hacemos el fit
  heart_RandomForestClassifier.fit(X_train, y_train)
  # Creamos la predicción
  y_pred_heart_RandomForestClassifier = heart_RandomForestClassifier.predict(X_test)
  # print(classification_report(y_true=y_test,y_pred=y_pred_heart_RandomForestClassifier))
  #Obtenemos los datos de accuracy_score
  return RandomForest.__name__, accuracy_score(y_test,y_pred_heart_RandomForestClassifier),heart_RandomForestClassifier.get_params(deep=True).items()



'''
Decision Tree Classifier
'''
def DecisionTree(df,size):
  X = df.drop(["target"],axis=1)
  y = df["target"]
  X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=size,random_state=0)
  # llamamos al modelo
  heart_decisiontree = DecisionTreeClassifier()
  #hacemos el fit
  heart_decisiontree.fit(X_train, y_train)
  # Creamos la predicción
  y_pred_heart_decisiontree = heart_decisiontree.predict(X_test)
  # print(classification_report(y_true=y_test,y_pred=y_pred_heart_decisiontree))
  #Obtenemos los datos de accuracy_score
  return DecisionTree.__name__, accuracy_score(y_test,y_pred_heart_decisiontree),heart_decisiontree.get_params(deep=True).items()



'''
C-Support Vector Classification
'''
def SVCmodel(df,size):
  X = df.drop(["target"],axis=1)
  y = df["target"]
  X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=size,random_state=0)
  # llamamos al modelo
  heart_svm = SVC()
  #hacemos el fit
  heart_svm.fit(X_train, y_train)
  # Creamos la predicción
  y_pred_svm = heart_svm.predict(X_test)
  #print(classification_report(y_true=y_test,y_pred=y_pred_svm))
  #Obtenemos los datos de accuracy_score
  return SVCmodel.__name__, accuracy_score(y_test,y_pred_svm),heart_svm.get_params(deep=True).items()

'''
Logistic Regression
'''
def LogisticRegr(df,size):
  X = df.drop(["target"],axis=1)
  y = df["target"]
  X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=size,random_state=0)
  # llamamos al modelo
  heart_LogisticRegression = LogisticRegression(solver='lbfgs', max_iter=10000)
  #hacemos el fit
  heart_LogisticRegression.fit(X_train, y_train)
  # Creamos la predicción
  y_pred_LogisticRegression = heart_LogisticRegression.predict(X_test)
  # print(classification_report(y_true=y_test,y_pred=y_pred_svm))
  #Obtenemos los datos de accuracy_score
  return LogisticRegr.__name__, accuracy_score(y_test,y_pred_LogisticRegression),heart_LogisticRegression.get_params(deep=True).items()


'''
K-Nearest Neighbour
'''
def knearest_find_k(df,size):
  puntuaciones = [] 
  parametros = []
  X = df.drop(["target"],axis=1)
  y = df["target"]
  for j in range(1,20):
      X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=size,random_state=0)
      # llamamos al modelo
      heart_knn = KNeighborsClassifier(n_neighbors = j, weights = 'uniform',algorithm = 'brute',metric = 'manhattan')  # n_neighbors means k
      # hacemos el fit
      heart_knn.fit(X_train, y_train)
      # Creamos la predicción
      y_pred_knn = heart_knn.predict(X_test)
      # Obtenemos los resultados del accuracy_score haciendo append a la lista de puntuaciones
      puntuaciones.append(accuracy_score(y_test, y_pred_knn))
      parametros.append(heart_knn.get_params(deep=True).items())
      
  #buscamos la mejor puntuacion    
  maxscore = max(puntuaciones)
  position = puntuaciones.index(max(puntuaciones))
  #print("Maximum KNN Score is",(maxscore), "with",puntuaciones.index(max(puntuaciones)),"n_neighbors and test_size:",i)
  return knearest_find_k.__name__, maxscore,parametros[position]


'''
Modelos con dummies
'''
def models_con(df):
  lista_size = []
  lista_results = []
  lista_modelo = []
  lista_parameters = []
  for i in range(1,9):
    for fn in [gaussian_naive_bayes(df,i/10),bernoulli(df,i/10), multinomial(df,i/10),RandomForest(df,i/10),DecisionTree(df,i/10),LogisticRegr(df,i/10),knearest_find_k(df,i/10),SVCmodel(df,i/10)]:
      dict = {"Test_Size": lista_size.append(i/10),
              "Modelo": lista_modelo.append(fn[0]),
              "Result":lista_results.append(fn[1]),
              "Hyperparameters":lista_parameters.append(fn[2])
              }
  d = {'Test_Size':lista_size,'Modelo':lista_modelo,"Result":lista_results,"Hyperparameters":lista_parameters}
  return d


'''
Modelos sin dummies
'''
def models_sin(df):
  lista_size = []
  lista_results = []
  lista_modelo = []
  lista_parameters = []

  for i in range(1,9):
    for fn in [gaussian_naive_bayes(df,i/10),bernoulli(df,i/10), multinomial(df,i/10),RandomForest(df,i/10),DecisionTree(df,i/10),LogisticRegr(df,i/10),knearest_find_k(df,i/10),SVCmodel(df,i/10)]:
      dict = {"Test_Size": lista_size.append(i/10),
              "Modelo": lista_modelo.append(fn[0]),
              "Result":lista_results.append(fn[1]),
              "Hyperparameters":lista_parameters.append(fn[2])
              }
  g = {'Test_Size':lista_size,'Modelo':lista_modelo,"Result":lista_results,"Hyperparameters":lista_parameters}
  return g