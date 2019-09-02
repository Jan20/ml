import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as plt_img
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

def extract_features(conv_base, dir: str, sample_count: int):

    datagen = ImageDataGenerator(rescale=1./255)
    batch_size: int = 20

    features = np.zeros(shape=(sample_count, 4, 4, 512))
    labels = np.zeros(shape=(sample_count))

    generator = datagen.flow_from_directory(
        dir, 
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='binary'
    )

    i: int = 0

    for inputs_batch, labels_batch in generator:

        features_batch = conv_base.predict(inputs_batch)

        features[i * batch_size : (i + 1) * batch_size] = features_batch
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch

        i += 1

        if i * batch_size >= sample_count: 

            break

        return features, labels