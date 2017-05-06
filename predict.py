# -*- coding: utf-8 -*-
import argparse
import keras.preprocessing.image
import numpy

parser = argparse.ArgumentParser()
parser.add_argument('model')
parser.add_argument('image')
args = parser.parse_args()

image = keras.preprocessing.image.load_img(
    args.image,
    target_size=(224, 224)
)
x = numpy.expand_dims(
    keras.preprocessing.image.img_to_array(image),
    axis=0
)

model = keras.models.load_model(args.model)

predicted = model.predict(x)[0,0]
if predicted == 0:
    print("anime")
else:
    print("others")
