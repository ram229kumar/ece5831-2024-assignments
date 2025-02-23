import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from PIL import Image

class Mnist():

    """
    A class to load and predict digits using a trained neural network.
    """

    image_dimensions = (28,28)
    model_loc_name = 'model/sample_weight.pkl'
    image_files={
        'test_0_img':'Custom MNIST Samples/Digit 0',
        'test_1_img':'Custom MNIST Samples/Digit 1',        
        'test_2_img':'Custom MNIST Samples/Digit 2',
        'test_3_img':'Custom MNIST Samples/Digit 3',
        'test_4_img':'Custom MNIST Samples/Digit 4',
        'test_5_img':'Custom MNIST Samples/Digit 5',
        'test_6_img':'Custom MNIST Samples/Digit 6',
        'test_7_img':'Custom MNIST Samples/Digit 7',
        'test_8_img':'Custom MNIST Samples/Digit 8',
        'test_9_img':'Custom MNIST Samples/Digit 9',        
    }

    def __init__(self):
        """
        Initialize an empty dictionary to store the model parameters.
        """
        self.params = {}


    def sigmoid(self, x):
        """
        To compute the sigmoid activation function.
        """
        return 1/(1 + np.exp(-x))


    def softmax(self, a):
        """
        To compute the softmax function for an array.
        """
        c = np.max(a)
        exp_a = np.exp(a - c)
        return exp_a/np.sum(exp_a)
    
    
    def init_network(self):
        """
        Loads the pretrained model parameters from the file.
        """
        with open(self.model_loc_name, 'rb') as f:
            self.params = pickle.load(f)
        return self.params

    def load_images(self, fileNames):
        """
        Loads and preprocess images from the directory.
        """
        images = []
        for img in os.listdir(fileNames):
            image = Image.open(fileNames+'/'+img).convert('L')
            image = image.resize(self.image_dimensions)
            image = np.array(image)
            images.append(image)

        images = np.array(images)
        images = images.reshape(len(images),-1)
        return images

    def predict(self, x):
        """
        Make a prediction for the given input using the neural network.
        """
        w1, w2, w3 = self.params['W1'], self.params['W2'], self.params['W3']
        b1, b2, b3 = self.params['b1'], self.params['b2'], self.params['b3']

        a1 = np.dot(x, w1) + b1
        z1 = self.sigmoid(a1)

        a2 = np.dot(z1, w2) + b2
        z2 = self.sigmoid(a2)

        a3 = np.dot(z2, w3) + b3
        y = self.softmax(a3)

        return y    