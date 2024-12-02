import pathlib
import numpy as np
import os
import shutil
import tensorflow as tf
import matplotlib.pyplot as plt

class DogsVsCats:

    CLASS_NAMES = ['dog', 'cat']
    IMAGE_SHAPE = (180, 180, 3)
    BATCH_SIZE = 32
    BASE_DIR = pathlib.Path('dogs-vs-cats')
    SRC_DIR = pathlib.Path('dogs-vs-cats-original/train')
    EPOCHS=20


    def __init__(self):
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None
        self.model = None
    
    def make_dataset_folders(self, subset_name, start_index, end_index):
        for category in ("dog", "cat"):
            dir = self.BASE_DIR / subset_name / category
            #print(dir)
            if os.path.exists(dir) is False:
                os.makedirs(dir)
            files = [f'{category}.{i}.jpg' for i in range(start_index, end_index)]
            #print(files)
            for i, file in enumerate(files):
                shutil.copyfile(src=self.SRC_DIR / file, dst=dir / file)
                if i % 500 == 0: # show only once every 500
                    print(f'src:{self.SRC_DIR / file} => dst:{dir / file}')
    
    def _make_dataset(self, subset_name):
        dataset = tf.keras.utils.image_dataset_from_directory(self.BASE_DIR/subset_name,image_size = self.IMAGE_SHAPE[:2],batch_size = self.BATCH_SIZE,label_mode="int")
        return dataset
    
    def make_dataset(self):
        self.train_dataset = self._make_dataset('train')
        self.valid_dataset = self._make_dataset('valid')
        self.test_dataset = self._make_dataset('test')
    
    def build_network(self, augmentation=True):
        """
        Build and compile a convolutional neural network with optional data augmentation.
        """
        try:
            if augmentation:
                data_augmentation = tf.keras.Sequential([
                    tf.keras.layers.RandomFlip('horizontal'),
                    tf.keras.layers.RandomRotation(0.1),
                    tf.keras.layers.RandomZoom(0.2)
                ])
            else:
                data_augmentation = None

            self.model = tf.keras.models.Sequential()
            if data_augmentation:
                self.model.add(data_augmentation)

            # Add Rescaling and CNN tf.keras.layers
            inputs = tf.keras.layers.Input(shape = self.IMAGE_SHAPE)
            x = tf.keras.layers.Rescaling(1./255)(inputs)
            x = tf.keras.layers.Conv2D(filters=16, kernel_size=3, activation="relu")(x)
            x = tf.keras.layers.AveragePooling2D(pool_size=2)(x)
            x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu")(x)
            x = tf.keras.layers.AveragePooling2D(pool_size=2)(x)
            x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
            x = tf.keras.layers.AveragePooling2D(pool_size=2)(x)
            x = tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)
            x = tf.keras.layers.AveragePooling2D(pool_size=2)(x)
            x = tf.keras.layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)
            x = tf.keras.layers.AveragePooling2D(pool_size=2)(x)

            x = tf.keras.layers.Flatten()(x)
            outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
            self.model = tf.keras.Model(inputs=inputs, outputs=outputs)

            # Compile the model
            self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            print("Model built successfully!!")  # Debug log
        except Exception as e:
            print(f"Error in build_network: {e}")  # Debug log

    def train(self, model_name):
        callbacks=[tf.keras.callbacks.ModelCheckpoint(model_name, save_best_only=True)]
        return self.model.fit(self.train_dataset,validation_data = self.valid_dataset,epochs=self.EPOCHS,callbacks=callbacks)

    def load_model(self, model_name):
        self.model = tf.keras.models.load_model(model_name)

    def predict(self, image_file):
        img = tf.keras.utils.load_img(image_file, target_size=self.IMAGE_SHAPE[:2])
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)

        predictions = self.model.predict(img_array)
        predicted_class = self.CLASS_NAMES[np.argmax(predictions)]
        plt.imshow(img)
        plt.title(f"Prediction: {predicted_class}")
        plt.show()