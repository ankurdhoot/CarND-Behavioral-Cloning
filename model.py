from keras.applications.vgg16 import preprocess_input, VGG16
from keras.layers import Conv2D, Dense, Input, Dropout, Flatten, Lambda
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.backend import tf

from scipy import ndimage
import csv
import numpy as np

# NOTES
# I used the NVIDIA deep net to predict steering angles.
# I noticed the model had a hard time predicting larger steering angles.
# Adding a tanh activation function at the end seemed to help with that.
# My initial guess when adding the tanh was that the slope increases
# sharply outside a small interval, so maybe that'll help it produce larger
# predictions when necessary. Haven't validated this hypothesis though.
# Dropout seemed to help at .25, but not .5.
# Normalizing the input had a surprisingly helpful effect, even though 
# we are using an Adam optimizer.
lines = []
with open('data/driving_log.csv') as csv_file: 
    reader = csv.reader(csv_file)
    # read the header first
    header = next(reader)
    print(header)
    for line in reader:
        lines.append(line)

# Stack the center, left, and right image depth wise (along the channel dimension).
# This didn't work because drive.py needs to be modified to input 3 images.
stacked_images = []
measurements = []
for line in lines:
    # center, left, right image path
    image_sources = [line[0], line[1], line[2]]
    # the images live in the data folder
    full_paths = ['data/' + img_src.strip() for img_src in image_sources] 
    
    # center_img, left_img, right_img = ndimage.imread(full_paths[0]), ndimage.imread(full_paths[1]), ndimage.imread(full_paths[2])
    center_img = ndimage.imread(full_paths[0])
    center_img_flipped = np.fliplr(center_img)
    # stacked_images.append(np.dstack((center_img, left_img, right_img)))
    stacked_images.append(center_img)
    stacked_images.append(center_img_flipped)
    measurement = float(line[3])
    measurements.append(measurement)
    measurements.append(-measurement)
   

X_train = np.array(stacked_images)
print(X_train.shape)
y_train = np.array(measurements)
print(y_train.shape)

#simulator_input = Input(shape=(160, 320, 9))
#resized_input = Lambda(lambda image : preprocess_input(tf.image.resize_images(image, (224, 224))))(simulator_input)

#base_model = VGG16(weights='imagenet', include_top=False)
#for layer in base_model.layers:
 #   layer.trainable = False
#vgg = base_model(resized_input)
#x = Flatten()(vgg)
#x = Dense(100, activation='relu')(x)
#x = Dense(50, activation='relu')(x)
#predictions = Dense(1)(x)

#model = Model(inputs=simulator_input, outputs=predictions)
#model.compile(loss='mse', optimizer='adam')
#model.summary()
# model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=3)

# model.save('model.h5')

# Create the Nvidia Model
# Input(shape=(160, 320, 3) -> Conv2D(filters=24, kernel=(5, 5), strides=(2,2), activation='relu') -> Conv2D(filters=36, kernel=(5,5), strides=(2,2), activation='relu') -> Conv2D(filters=48, kernel=(5,5), strides=(2,2), activation='relu') -> Conv2D(filters=64, kernel=(3,3), activation='relu') -> Conv2D(filters=64, kernel=(3,3), activation='relu') -> Dense(100, activation='relu') -> Dense(50, activation='relu') -> Dense(10, activation='relu') -> Dense(1)
nvidia = Sequential()
nvidia.add(Lambda(lambda x : x / 255.0 - .5, input_shape=(160, 320, 3)))
nvidia.add(Conv2D(filters=24, kernel_size=(5,5), strides=(2,2), activation='relu'))
nvidia.add(Conv2D(filters=36, kernel_size=(5,5), strides=(2,2), activation='relu'))
nvidia.add(Conv2D(filters=48, kernel_size=(5,5), strides=(2,2), activation='relu'))
nvidia.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
nvidia.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
nvidia.add(Flatten())
nvidia.add(Dense(100, activation='relu'))
nvidia.add(Dropout(rate=.25))
# TODO : maybe try cropping photo?
nvidia.add(Dense(50, activation='relu'))
nvidia.add(Dropout(rate=.25))
nvidia.add(Dense(10, activation='relu'))
nvidia.add(Dense(1, activation='tanh'))

nvidia.compile(loss='mse', optimizer='adam')
nvidia.summary()
nvidia.fit(X_train, y_train, epochs=5, validation_split=.2, shuffle=True)

nvidia.save('model.h5')