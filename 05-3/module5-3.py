import mnist as mn
import sys
import os
import numpy as np


if __name__ == "__main__":

    """
    Main script to test an image using the mnist class for digit recognition.

    This takes two command-line arguments:
    1. arg1(sys.argv[1]): Filename of the image to be tested ex: '0_1.jpg'
    2. arg2(sys.argv[2]): Expected digit label for the image ex: '0','1'
    """

    arg1 = sys.argv[1]
    arg2 = sys.argv[2]
     
    mnist = mn.Mnist()
    mnist.init_network()

    for file in mnist.image_files:
        x_test = mnist.load_images(mnist.image_files[file])
        x_test = x_test/255.0
        

        index = 0
        found = False
        for img in  os.listdir(mnist.image_files[file]):
            if img==arg1:
                found = True
                break
            else:
                index+=1
        if found == True:
            x = x_test[index]
            y = mnist.predict(x)
            y_hat = np.argmax(y)

            if y_hat == int(arg2):
                print(f'Success: Image {arg1} is for digit {arg2} is recognized as {y_hat}.')
            else:
                print(f'Fail: Image {arg1} is for digit {arg2} but the inference result is {y_hat}.')