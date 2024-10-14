# %%
from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np
import sys

# %%
def loadImage(filePath):
    """
    Loads the image from the filePath provided in the arguments (or if you uncomment the sample path and comment out the argument it will always take the 29.jpg) 
    and converts it into a RGB Format
    """
    image = Image.open(filePath).convert("RGB")
    return image

# %%
def init():
    """
    Initializes the environment for the program.
    """
    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)

# %%
def loadModel():

    """
    Load the pre-trained keras model (from Teachable Machines google) and the class labels from the respective files.
    """
    # Load the model
    model = load_model("keras_Model.h5", compile=False)

    # Load the labels
    class_names = open("labels.txt", "r").readlines()

    return model, class_names

# %%
def prepareInput(image):

    """
    Preprocess the input image to be compatible with the keras model.
    Like sizing, normalizing and reshaping.
    """
    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)


    # Replace this with the path to your image
    # image = Image.open(imageLocation).convert("RGB")

    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    return data

# %%
def predict(model, class_names, data):

    """
    Perform a prediction on the preprocessed image data using the model
    and print the predicted class along with the confidence score.
    """
    # Predicts the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    print("Class:", class_name[2:], end="")
    print("Confidence Score:", confidence_score)
    


# %%

"""
You can uncomment the below code to test a 29.jpg image which is a scissors1 but you have to comment where we are reading the `filePath = sys.argv[1]`.
If not please check the next docstring.
"""
# sys.argv[1] = "29.jpg"


# %%
if __name__ =="__main__":

    """
    The scripts input is an image file path as a command-line argument (unless the above code is uncommented) and processes the image and predicts its class with the
    pre trained model.
    """
    filePath = sys.argv[1]

    init()
    image = loadImage(filePath)
    model, class_names = loadModel()
    data = prepareInput(image)
    predict(model,class_names,data)
    image.show()
