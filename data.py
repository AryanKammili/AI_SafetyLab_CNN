import pandas as pd
import numpy as np
import sklearn
from sklearn import preprocessing
from sklearn import model_selection
import matplotlib.pyplot as plt
import cv2
import os
import random
import pickle

os.environ['TF_CPP_MIN_LOG_LEVpip install opencv-pythonEL'] = '2'

import tensorflow as tf

path = 'C:/Users/aryan/OneDrive/Desktop/vehicle_classification'
objects = ["Bicycle", "Bus", "Car", "Motorcycle", "NonVehicles", "Taxi", "Truck", "Van"]
data = []

def create_data():
    for object in objects:

        new_path = os.path.join(path, object)
        class_num = objects.index(object)

        for img in os.listdir(new_path):
            # We want the image to be in gray scale because it won't make a difference #
            # The different objects come in all colors so that's why we set it to grayscale #
            img_array = cv2.imread(os.path.join(new_path, img), cv2.IMREAD_GRAYSCALE)
            new_array = cv2.resize(img_array, (64, 64))
            data.append([new_array, class_num])
            plt.show()


create_data()
x = []
y = []

random.shuffle(data)

# Reshape tensors for neural network input #
for pixels, label in data:
    # This allows us to split the data into inputs and outputs #
    x.append(pixels)
    y.append(label)

# We do -1 because it encompasses all the instances of data #
# 64 by 64 because of the count of pixels #
# We finally do 1 because we are in gray scale #
x = np.array(x).reshape(-1, 64, 64, 1)
y = np.array(y)

x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2, random_state=42)

test_loader = [x_test, y_test]
train_loader = [x_train, y_train]

# All the possible different classes outputs, helpful for softmax # 
classes = ["Bicycle", "Bus", "Car", "Motorcycle", "NonVehicles", "Taxi", "Truck", "Van"]

# Export Data to a new file this way we don't need to reload and create the data every time # 
pickle_out = open("TrainData.pickle", "wb")
pickle.dump(train_loader, pickle_out)
pickle_out.close()

pickle_out = open("TestData.pickle", "wb")
pickle.dump(test_loader, pickle_out)
pickle_out.close()