import mnist as mn
import sys
import os
import numpy as np


if __name__ == "__main__":

    arg1 = sys.argv[1]
    arg2 = sys.argv[2]

     # Initializing network and loading images
     
    mnist = mn.Mnist()
    mnist.init_network()

    for file in mnist.image_files:
        print(file)

    x_test = mnist.load_images(mnist.image_files['test_0_img'])
    x_test = x_test/255.0
    
    # To get testing image (i.e., x) from x_test using file name of the image

    index=0
    for img in  os.listdir(mnist.image_files['test_0_img']):
        if img==arg1:
            break
        else:
            index+=1
    x = x_test[index]

   # Predicting the output class

    y = mnist.predict(x)
    y_hat = np.argmax(y)

    if y_hat == int(arg2):
        print(f'Success: Image {arg1} is for digit {arg2} is recognized as {y_hat}.')
    else:
        print(f'Fail: Image {arg1} is for digit {arg2} but the inference result is {y_hat}.')


# Note: Take .jpg image as input