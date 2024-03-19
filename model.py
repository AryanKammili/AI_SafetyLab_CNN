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

# These layers are typed out a lot so doing this makes it easier # 
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
    # We flatten the layer so that its 1-D # 
    tf.keras.layers.Flatten(),
    # Create fully Connected Layers at end # 
    tf.keras.layers.Dense(units=128, activation="relu", kernel_initializer="he_normal"),
    # Drouput deactivates and ignores neutrons that have a weight of 0, making computation faster #
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(units=64, activation="relu", kernel_initializer="he_normal"),
    tf.keras.layers.Dropout(0.5),    
    # Since we have 8 possible outcomes we have 8 possible weights, the highest weight is the predicted object # 
    tf.keras.layers.Dense(units=8, activation="softmax")
])

model.compile(
    # Our Data has multiple possiblites and not two, so we use categorical cross entropy # 
    loss = "categorical_crossentropy",
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01),
    metrics = ["accuracy"]
)

model.fit(x_train, y_train, epochs=12)




print("Done")