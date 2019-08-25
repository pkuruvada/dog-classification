import os
import numpy as numpy
import keras.backend as K
import tensorflow as tf
import argparse

from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Input, Dropout

from keras.applications.vgg19 import VGG19
from keras.preprocessing.image import ImageDataGenerator

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

images_dir = os.path.join('Images')
annotations_dir = os.path.join('Annotation')
weights_file_name = 'dogs.h5'

class_names = os.listdir(images_dir)
class_count = len(class_names)


def get_generator():
    gen = ImageDataGenerator(
        vertical_flip=False,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        rotation_range=20,
        validation_split=0.2
    )

    generator = gen.flow_from_directory(
        images_dir,
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=64
    )

    return generator


def get_model():

    base_model = VGG19(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3)
    )

    for layer in base_model.layers:
        layer.trainable = False

    output = base_model.output
    x = Conv2D(64, (1, 1), activation='relu')(output)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.2)(x)

    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(class_count, activation='sigmoid')(x)

    model = Model(base_model.input, x)
    model.summary()
    return model


model = get_model()
generator = get_generator()
model.compile(optimizer='adam',
              loss='categorical_crossentropy', metrics=['acc'])

parser = argparse.ArgumentParser(description="Parse")
parser.add_argument('--train', action='store_true')
parser.add_argument('--load', action='store_true')
args = parser.parse_args()

print('Passed args ', args)


if args.load:
    model.load_weights(weights_file_name)

if args.train:
    hostory = model.fit_generator(generator, epochs=2, steps_per_epoch=200)
    model.save_weights(weights_file_name)
