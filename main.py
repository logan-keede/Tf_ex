import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28*28).astype('float32')/255.0

# y_train = y_train.reshape(-1, 28*28)
x_test = x_test.reshape(-1, 28*28).astype('float32')/255.0
# y_test = y_test.reshape(-1, 28*28)
model = keras.Sequential([
    keras.Input(shape = (28*28)),
    layers.Dense(183*4,activation = "relu"),
    layers.Dense(183,activation = "relu"),
    layers.Dense(43,activation = "relu"),
    layers.Dense(10),

])
model.compile(
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits = True),
    optimizer = keras.optimizers.Adam(learning_rate=0.0001) ,
    metrics = ["accuracy"],
)
model.fit(x_train, y_train, batch_size = 32, epochs = 8, verbose = 2)
model.evaluate(x_test, y_test, batch_size = 32, verbose = 2)