import numpy as np
import tensorflow as tf
from keras import models
from keras import layers
import matplotlib.pyplot as plt

class Imdb:
    def __init__(self):
        self.DIMENSIONS = 10000
        self.NUM_EPOCHS=20
        self.BATCH_SIZE=512
        self.VALIDATION_SPLIT=0.2
        self.PATIENCE=2
        self.model = None

    def prepare_data(self,sequences):
        results = np.zeros((len(sequences),self.DIMENSIONS))
        for i, sequence in enumerate(sequences):
            results[i, sequence] = 1.
        return results
    
    def build_model(self):
        self.model = models.Sequential()
        self.model.add(layers.Dense(16, activation='relu', input_shape=(self.DIMENSIONS,)))
        self.model.add(layers.Dense(16, activation='relu'))
        self.model.add(layers.Dense(1, activation="sigmoid"))
        self.model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
        return self.model
    
    def train(self, x_train, y_train):
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.PATIENCE)
        history = self.model.fit(x_train, y_train, epochs=self.NUM_EPOCHS, batch_size=self.BATCH_SIZE, validation_split=self.VALIDATION_SPLIT, callbacks=[callback])
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

    def evaluate(self,x_test,y_test):
        return self.model.evaluate(x_test,y_test)
    
    def predict(self, model, x_test, y_test):
        x_test = np.expand_dims(x_test, axis=0)
        prediction = model.predict(x_test)  
        predicted_class = "negative" if prediction[0][0] < 0.5 else "positive"  
        true_class = "negative" if y_test < 0.5 else "positive"
        print("True Class: ", true_class, " Predicted Class: ", predicted_class)
