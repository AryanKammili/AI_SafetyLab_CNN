from functools import partial
import tensorflow as tf
import pickle

# Load in the data from other file, now it is super fast to open up data :) # 
pickle_in = open("TestData.pickle", "rb")
x_test, y_test = pickle.load(pickle_in)

pickle_in = open("TrainData.pickle", "rb")
x_train, y_train = pickle.load(pickle_in)

print(x_train.shape)
print(y_train.shape)

Conv2D = partial(tf.keras.layers.Conv2D, kernel_size=3, padding="same", activation="relu")
Pool2D = tf.keras.layers.MaxPool2D()

model = tf.keras.Sequential([
    Conv2D(filters=64, kernel_size=3, input_shape=[64, 64, 1]),
    Pool2D,
    Conv2D(filters=128),
    Conv2D(filters=128),
    Pool2D,
    Conv2D(filters=256),
    Conv2D(filters=256),
    Pool2D,
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=128, activation="relu", kernel_initializer="he_normal"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(units=64, activation="relu", kernel_initializer="he_normal"),
    tf.keras.layers.Dropout(0.5),    
    tf.keras.layers.Dense(units=8, activation="softmax")
])

model.compile(
    loss = "categorical_crossentropy",
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01),
    metrics = ["accuracy"]
)

model.fit(x_train, y_train, epochs=12)




print("Done")