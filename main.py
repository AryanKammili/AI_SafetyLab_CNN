import pandas as pd
import numpy as np
import sklearn
from sklearn import preprocessing
from sklearn import model_selection
import matplotlib.pyplot as plt
import cv2
import os

os.environ['TF_CPP_MIN_LOG_LEVpip install opencv-pythonEL'] = '2'

import tensorflow as tf

path = 'C:/Users/aryan/OneDrive/Desktop/vehicle_classification'
objects = ["Bicycle", "Bus", "Car", "Motorcycle", "NonVehicles", "Taxi", "Truck", "Van"]

for object in objects:
    path = os.path.join(path, object)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
        plt.show(img_array, cmap="gray")
        plt.show()
        break


