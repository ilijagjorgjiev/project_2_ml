import tensorflow as tf
import numpy as np

from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


#Ref, idea and how to set up the unet come from her https://www.depends-on-the-definition.com/unet-keras-segmenting-images/
def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True, dropout=0.2):
    #param input_tensor => everything that have been created up until now
    #param n_filter => number of filters to be applied
    #param kernel_size => size of the kernel
    #param batchnorm => Boolean: Default: True, whether or not use batchnorm, we found it very helpful
    #param dropout = 0.2 => We use dropout to avoid overfitting

    #Create 2 Convolution layers, with padding = same and kernel_initializer="he_nromal"
    #Activation function used is RELU
    #Use BatchNormalization if True
    #Use Dropout to avoid overfitting

    #first layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(dropout)(x)

    # second layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

#Create the unet as found in the paper https://arxiv.org/pdf/1505.04597.pdf
def get_unet(input_img = Input((400, 400, 3), name='img'), n_filters=32, dropout=0.2, batchnorm=True):
    #param input_img => is basically of type Input and we said the input_dimesion of the img, by default set to (400, 400, 3)
    #param n_filters = 32 => number of filters to start with
    #param dropout => dropout factor
    #param batchnorm => Boolean: Default: True, whether or not use batchnorm, we found it very helpful
    
    # contracting path
    c1 = conv2d_block(input_img, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm, dropout=dropout)
    p1 = MaxPooling2D((2, 2)) (c1)

    c2 = conv2d_block(p1, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm, dropout=dropout)
    p2 = MaxPooling2D((2, 2)) (c2)

    c3 = conv2d_block(p2, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm, dropout=dropout)
    p3 = MaxPooling2D((2, 2)) (c3)

    c4 = conv2d_block(p3, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm, dropout=dropout)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

    c5 = conv2d_block(p4, n_filters=n_filters*16, kernel_size=3, batchnorm=batchnorm, dropout=dropout)

     # expansive path
    u6 = Conv2DTranspose(n_filters*8, (2, 2), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    c6 = conv2d_block(u6, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm, dropout=dropout)

    u7 = Conv2DTranspose(n_filters*4, (2, 2), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    c7 = conv2d_block(u7, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm, dropout=dropout)

    u8 = Conv2DTranspose(n_filters*2, (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    c8 = conv2d_block(u8, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm, dropout=dropout)

    u9 = Conv2DTranspose(n_filters*1, (2, 2), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = conv2d_block(u9, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm, dropout=dropout)

    #Output function the sigmoid
    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)

    #Create the Model
    model = Model(inputs=[input_img], outputs=[outputs])

    #Return the model ready for training
    return model
