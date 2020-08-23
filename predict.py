import cv2
import tensorflow
from tensorflow.keras import models
from tensorflow.keras.applications import EfficientNetB1
import sys
import numpy as np


def get_model_architecture():
    efficient_net = EfficientNetB1(
        weights=None, include_top=False, input_shape=(64, 64, 3))
    model = models.Sequential()
    model.add(efficient_net)
    model.add(tensorflow.keras.layers.Flatten())
    model.add(tensorflow.keras.layers.Dense(1024, activation="relu"))
    model.add(tensorflow.keras.layers.Dropout(0.5))
    model.add(tensorflow.keras.layers.Dense(10, activation="softmax"))
    model.summary()
    return model


def get_image_from_path(path):
    return cv2.imread(path)


def load_weights(model, model_weights):
    model.load_weights(model_weights)


def predict(model, image):
    y_pred = model.predict(np.expand_dims(image, axis=0))
    class_names = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway',
                   'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']
    return class_names[int(np.argmax(y_pred[0]))]


if __name__ == "__main__":
    img_path = sys.argv[1]
    model_path = sys.argv[2]
    img = get_image_from_path(img_path)
    model = get_model_architecture()
    load_weights(
        model, model_path)
    print(predict(model, img))
