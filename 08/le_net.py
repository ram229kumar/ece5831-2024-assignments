from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np


class LeNet:
    def __init__(self, batch_size=32, epochs=20):
        """
        Initialize the LeNet class.
        """
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = None
        self._create_lenet()
        self._compile()

    def _create_lenet(self):
        """
        Create the LeNet model.
        """
        self.model = Sequential([
            Conv2D(filters=6, kernel_size=(5, 5), activation='sigmoid', 
                   input_shape=(28, 28, 1), padding='same'),
            AveragePooling2D(pool_size=(2, 2), strides=2),

            Conv2D(filters=16, kernel_size=(5, 5), activation='sigmoid', padding='same'),
            AveragePooling2D(pool_size=(2, 2), strides=2),

            Flatten(),
            Dense(120, activation='sigmoid'),
            Dense(84, activation='sigmoid'),
            Dense(10, activation='softmax')
        ])

    def _compile(self):
        """
        Compile the model.
        """
        self.model.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

    def _preprocess(self):
        """
        Preprocess the MNIST dataset.
        """
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        x_train = x_train / 255.0
        x_test = x_test / 255.0

        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

        y_train = to_categorical(y_train, 10)
        y_test = to_categorical(y_test, 10)

        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test

    def train(self):
        """
        Train the LeNet model on the MNIST dataset.
        """
        self._preprocess()
        self.model.fit(self.x_train, self.y_train,
                       batch_size=self.batch_size,
                       epochs=self.epochs)

    def save(self, filepath):
        """
        Save the trained model to a file.
        """
        self.model.save(filepath)
        print(f"Model saved to {filepath}")

    def load(self, filepath):
        """
        Load a model from a file.
        """
        self.model = load_model(filepath)
        print(f"Model loaded from {filepath}")

    def predict(self, image):
        """
        Predict the class of a given image.
        """
        if len(image.shape) == 2:
            image = image.reshape(1, 28, 28, 1)
        elif len(image.shape) == 3:
            image = image.reshape(1, *image.shape)
        
        image = image / 255.0
        predictions = self.model.predict(image)
        return np.argmax(predictions, axis=1)[0]


if __name__ == "__main__":
    lenet = LeNet(batch_size=64, epochs=10)
    lenet.train()
    lenet.save("ippili_cnn_model.keras")

    lenet.load("ippili_cnn_model.keras")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    sample_image = x_test[9]
    plt.imshow(sample_image)
    predicted_label = lenet.predict(sample_image)
    print(f"Predicted Label: {predicted_label}")
    plt.show()
