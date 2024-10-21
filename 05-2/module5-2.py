import sys
import matplotlib.pyplot as plt
import numpy as np
from mnist_data import MnistData

def show_image(image,label):
    """
    Displays the given image and prints the corresponding label.
    """
    image = image.reshape(28,28)
    plt.imshow(image)
    plt.show()
    print('Label: ',np.argmax(label))



if(__name__=='__main__'):
    if(len(sys.argv)!=3):
        print("Please follow this input method for output : python module5-2.py train/test index")

    train_or_test = sys.argv[1].lower()
    index = int(sys.argv[2])

    mnist_data = MnistData()
    (train_images,train_labels), (test_images,test_labels) = mnist_data.load()

    if(train_or_test=='train'):
        if(index>=0 and index<len(train_images)):
            show_image(train_images[index],train_labels[index])
        else:
            print('Index out of range.')
    elif(train_or_test=='test'):
        if(index>=0 and index<len(test_images)):
            show_image(test_images[index],test_labels[index])
        else:
            print('Index out of range.')
    else:
        print('Please enter either train or test.')

