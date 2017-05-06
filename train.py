# -*- coding: utf-8 -*-
import argparse
import keras
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
import keras.preprocessing.image
import os

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

parser = argparse.ArgumentParser()
# parser.add_argument('image_directories', nargs="+")
parser.add_argument('image_directory')
parser.add_argument('--optimizer', required=True)
parser.add_argument('--freeze_vgg', action="store_true")
args = parser.parse_args()

batchsize = 10

model = Sequential()
classes = len(os.listdir(args.image_directory))
base_model = keras.applications.VGG16(input_shape=(224, 224, 3), weights='imagenet', include_top=False, classes=2)
if args.freeze_vgg:
    for layer in base_model.layers:
        layer.trainable = False
model.add(base_model)

print(base_model.output_shape)
# top_model = Sequential()
# # top_model.add(Flatten(input_shape=model.output_shape[1:]))
# top_model.add(Flatten(input_shape=(7, 7, 512)))
# top_model.add(Dense(256, input_dim=25088, activation='relu'))
# top_model.add(Dropout(0.5))
# top_model.add(Dense(1, activation='sigmoid'))
# model.add(top_model)

model.add(Flatten(input_shape=base_model.output_shape[1:]))
model.add(Dense(256, input_dim=25088, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
# # note that it is necessary to start with a fully-trained
# # classifier, including the top classifier,
# # in order to successfully do fine-tuning
# top_model.load_weights(top_model_weights_path)
#
# # add the model on top of the convolutional base

fpath = 'weights.{epoch:02d}-{loss:.2f}-{acc:.2f}.hdf5'
save_checkpoint = keras.callbacks.ModelCheckpoint(filepath=fpath, monitor='acc', verbose=1, save_best_only=True, mode='auto')

dataset_generator = keras.preprocessing.image.ImageDataGenerator(rotation_range=40,
                                                                 width_shift_range=0.2,
                                                                 height_shift_range=0.2,
                                                                 shear_range=0.2,
                                                                 zoom_range=0.2,
                                                                 horizontal_flip=True,
                                                                 fill_mode='constant',
                                                                 # fill_mode='reflect'
                                                                 )

train_dataset = dataset_generator.flow_from_directory(
    directory=args.image_directory,
    target_size=(224, 224),
    class_mode="binary",
    shuffle=True,
    batch_size=batchsize
)


# i = 0
# for batch in dataset_generator.flow_from_directory(
#     directory=args.image_directory,
#     target_size=(224, 224),
#     class_mode="binary",
#     shuffle=True,
#     save_to_dir="preview",
#     batch_size=batchsize
# ):
#     i += 1
#     if i > 20:
#         exit(0)

# print(train_dataset.next()[0].shape)

# model.compile(optimizer='adam',
#               loss='binary_crossentropy',
#               metrics=['accuracy'])
model.compile(optimizer=args.optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy'])

from keras.utils.vis_utils import plot_model
plot_model(model, to_file='model.png', show_shapes=True)

model.fit_generator(
    train_dataset,
    steps_per_epoch=100,
    epochs=10000,
    callbacks=[save_checkpoint]
)
