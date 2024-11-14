import mnist_data as mn
import sys
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from two_layer_net_with_back_prop import TwoLayerNetWithBackProp
from PIL import Image

if __name__ == "__main__":

    """
    Main script to test an image using the mnist class for digit recognition.

    This takes two command-line arguments:
    1. arg1(sys.argv[1]): Filename of the image to be tested ex: '0_1.jpg'
    2. arg2(sys.argv[2]): Expected digit label for the image ex: '0','1'
    """

    arg1 = sys.argv[1]
    arg2 = sys.argv[2]
     
    mnist = mn.MnistData()
    network = TwoLayerNetWithBackProp(784, 100, 10)
    with open("ippili_mnist_model.pkl", "rb") as f:
        network.params = pickle.load(f)
    network.update_layers()

    for file in mnist.image_files:
        x_test = mnist.load_images(mnist.image_files[file])


        index = 0
        found = False
        for img in  os.listdir(mnist.image_files[file]):
            if img==arg1:
                found = True
                break
            else:
                index+=1
        if found == True:
            break
x = x_test[index]
y = network.predict(x)
y_hat = np.argmax(y)
if y_hat == int(arg2):
    print(f'Success: Image {arg1} is for digit {arg2} is recognized as {y_hat}.')
else:
    print(f'Fail: Image {arg1} is for digit {arg2} but the inference result is {y_hat}.')
imagePath = os.path.join(mnist.image_files[file],img)
imgOpen = Image.open(imagePath).convert('L')
plt.imshow(imgOpen,cmap="gray")
plt.title(f"Expected: {arg2}, Predicted: {y_hat}")
plt.show()