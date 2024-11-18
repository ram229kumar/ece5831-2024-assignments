import sys
from le_net import LeNet
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array


def classify_image(image_path, expected_label):
    """
    Classify a single image using the trained LeNet model.
    """
    lenet = LeNet()
    lenet.load("ippili_cnn_model.keras")

    image_files={
        'test_0_img':'Custom KERAS Samples/Digit 0',
        'test_1_img':'Custom KERAS Samples/Digit 1',        
        'test_2_img':'Custom KERAS Samples/Digit 2',
        'test_3_img':'Custom KERAS Samples/Digit 3',
        'test_4_img':'Custom KERAS Samples/Digit 4',
        'test_5_img':'Custom KERAS Samples/Digit 5',
        'test_6_img':'Custom KERAS Samples/Digit 6',
        'test_7_img':'Custom KERAS Samples/Digit 7',
        'test_8_img':'Custom KERAS Samples/Digit 8',
        'test_9_img':'Custom KERAS Samples/Digit 9',        
    }

    for file in image_files:
        index = 0
        found = False
        for img in  os.listdir(image_files[file]):
            if img==image_filename:
                found = True
                break
            else:
                index+=1
        if found == True:
            break

    image_path = os.path.join(image_files[file],image_path)

    try:
        image = load_img(image_path, color_mode='grayscale', target_size=(28, 28))
        image = img_to_array(image)
    except Exception as e:
        print(f"Error loading image: {e}")
        sys.exit(1)

    predicted_label = lenet.predict(image)

    # Output result
    if predicted_label == expected_label:
        print(f"Success: Predicted {predicted_label}, Expected {expected_label}")
    else:
        print(f"Failure: Predicted {predicted_label}, Expected {expected_label}")


if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("Usage: python module8.py <image_filename> <expected_label>")
        sys.exit(1)

    image_filename = sys.argv[1]
    expected_digit = int(sys.argv[2])

    model_filename = "ippili_cnn_model.keras"
    classify_image(image_filename, expected_digit)
