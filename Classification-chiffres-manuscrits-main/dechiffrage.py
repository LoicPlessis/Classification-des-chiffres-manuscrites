import numpy as np
import pandas as pd
# import os
import cv2
# from sklearn.model_selection import train_test_split

import tensorflow as tf
#from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

filepath = './modelCNN'
liste_label = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
nom = input('Nom du fichier (*.jpg) :')

def afficher():
    pass


def dechifrer(nom_fichier):
    # nom_fichier = input('Nom du fichier (*.jpg) :')
    #path = f'../image1/{nom_fichier}.jpg'
    image = cv2.imread(f'./images/{nom_fichier}.jpg')

    plt.figure(figsize=(2,2))
    plt.subplot()
    plt.xticks([])
    plt.yticks([])
    plt.imshow(image)
    plt.show()

    #resultat = pd.DataFrame({})
    img_ar = np.array(image)/255.0
    img_exp = np.expand_dims(img_ar, axis=0)
    
    loaded_model = tf.keras.models.load_model(filepath)
    prediction = loaded_model.predict(img_exp)

    resultat = liste_label[np.argmax(prediction)]
    print('Prédiction :',resultat)
    return(resultat)

dechifrer(nom)