from tensorflow import keras
from keras import models
from keras import layers
import matplotlib.pyplot as plt
import numpy as np

class BostonHousing:

    """A class to handle regression tasks using the California Housing dataset."""

    def __init__(self):
        self.NUMERICAL_FEATURES = 13
        self.NUM_EPOCHS = 100
        self.BATCH_SIZE = 16
        self.VALIDATION_SPLIT = 0.2
        self.PATIENCE = 5
        self.model = None
    
    def prepare_data(self, data):
        mean = data.mean(axis=0)
        data -= mean
        std = data.std(axis=0)
        data /= std
        # data -= mean
        # data /= std
        return data

    def build_model(self):
        self.model = models.Sequential()
        self.model.add(layers.Dense(64, activation='relu', input_shape=(self.NUMERICAL_FEATURES,)))
        self.model.add(layers.Dense(64, activation='relu'))
        self.model.add(layers.Dense(1))
        self.model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
        return self.model
    
    def train(self,x_train,y_train):
        callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.PATIENCE)
        history = self.model.fit(x_train, y_train, epochs=self.NUM_EPOCHS, batch_size=self.BATCH_SIZE, validation_split=self.VALIDATION_SPLIT, callbacks=[callback])
        return history
    
    def plot_loss(self, history):
        history_dict = history.history
        loss_values = history_dict['loss']
        val_loss_values = history_dict['val_loss']
        epochs = range(1, len(loss_values) + 1)
        plt.plot(epochs, loss_values, 'bo', label='Training loss')
        plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
        plt.title('Training vs. Validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def evaluate(self, x_test, y_test):
        return self.model.evaluate(x_test, y_test)

    def predict(self, x_test):
        return self.model.predict(x_test)
