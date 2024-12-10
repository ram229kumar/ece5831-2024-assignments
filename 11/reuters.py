import numpy as np
import tensorflow as tf
from keras import models
from keras import layers
import matplotlib.pyplot as plt
from keras.utils import to_categorical

class Reuters:

    """A class to handle multiclass classification tasks using the Reuters dataset."""

    def __init__(self):
        self.DIMENSIONS = 10000
        self.NUM_CLASSES = 46
        self.NUM_EPOCHS=20
        self.BATCH_SIZE=512
        self.VALIDATION_SPLIT=0.2
        self.PATIENCE=2
        self.model = None
    
    def prepare_data(self, sequences):
        results = np.zeros((len(sequences),self.DIMENSIONS))
        for i, sequence in enumerate(sequences):
            results[i, sequence] = 1.
        return results
    
    def build_model(self):
        self.model = models.Sequential()
        self.model.add(layers.Dense(64,activation='relu', input_shape=(self.DIMENSIONS,)))
        self.model.add(layers.Dense(64,activation='relu'))
        self.model.add(layers.Dense(46,activation='softmax'))
        self.model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=["accuracy"])
        return self.model
    
    def train(self, x_train, y_train):
        y_train_one_hot = to_categorical(y_train, num_classes=self.NUM_CLASSES)
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.PATIENCE)
        history = self.model.fit(x_train, y_train_one_hot, epochs=self.NUM_EPOCHS, batch_size=self.BATCH_SIZE, validation_split=self.VALIDATION_SPLIT, callbacks=[callback])
        return history

    def plot_loss(self,history):
        history_dict = history.history
        history_dict.keys()
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
    
    def plot_accuracy(self,history):
        history_dict = history.history
        history_dict.keys()
        plt.clf()
        acc = history_dict["accuracy"]
        val_acc_values = history_dict["val_accuracy"]
        epochs = range(1, len(val_acc_values) + 1)
        plt.plot(epochs, acc, "bo", label="Training acc")
        plt.plot(epochs, val_acc_values, "b", label="Validation acc")
        plt.title("Training vs Validation accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.show()

    def evaluate(self, x_test, y_test):
        return self.model.evaluate(x_test, y_test)


    def predict(self, x_test, y_test):
        x_test = np.expand_dims(x_test, axis=0)
        prediction = self.model.predict(x_test)
        predicted_class = np.argmax(prediction[0])
        true_class = y_test
        print(f"True Class: {true_class}, Predicted Class: {predicted_class}")

