#!/usr/bin/env python3
import os

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, Flatten

def light_network(input_shape = (224, 224, 3), classes = 4):
   """A lightweight starting point network for training."""
   input = Input(input_shape)

   net = Conv2D(96, kernel_size = (7, 7), padding = 'same', strides = (2, 2))(input)
   net = BatchNormalization()(net)
   net = Activation('relu')(net)
   net = MaxPooling2D(pool_size = (3, 3))(net)

   net = Conv2D(256, kernel_size = (5, 5), padding = 'same', strides = (1, 1))(net)
   net = BatchNormalization()(net)
   net = Activation('relu')(net)
   net = MaxPooling2D(pool_size = (2, 2))(net)

   for i in range(3):
      net = Conv2D(512, kernel_size=(3, 3), padding='same', strides=(1, 1))(net)
      net = BatchNormalization()(net)
      net = Activation('relu')(net)
   net = MaxPooling2D(pool_size = (3, 3))(net)

   net = Flatten()(net)
   net = Dense(4048)(net)
   net = Dropout(0.2)(net)
   net = Dense(4048)(net)
   net = Dropout(0.2)(net)
   net = Dense(classes)(net)
   net = Activation('softmax')(net)

   return Model(input, net)



