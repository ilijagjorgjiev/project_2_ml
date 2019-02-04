import os
import random
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")


from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split

import tensorflow as tf

from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from helpers_submission import *
from read_augment_data import *
from unet import *
from loss_and_metrics import *
import keras as ks


#Set Image Width and Height
im_width = 400
im_height = 400

#Set Paths for reading from the training images, groundtruth and test_data
path_train = '../data/training_set/'
path_mask = "../data/training_set/"
path_test = '../data/testing_set/'
path_to_save_images = "../predictions/"

#Set the seed of 1, which is used for getting our best model
seed = 1
random.seed = seed
np.random.seed(seed=seed)

#This variable can be used to immediatelly use and predict with our best model so far if set to False
#If set to true, it will start training a new model
trainModel = True

if(trainModel):
    #Values used to get our best model, which are also default of the read_data_and_augmentation function
    train_generator = read_data_and_augmentation(path_images = path_train,
                                    path_mask = path_mask,
                                    class_img="images",
                                    class_mask="groundtruth",
                                    img_size = (400, 400),
                                    mask_size = (400, 400),
                                    batch_size = 8,
                                    seed = 1,
                                    image_color_mode = "rgb",
                                    mask_color_mode = "grayscale",
                                    rotation_range = 270,
                                    shear_range = 0.2,
                                    zoom_range = 0.2,
                                    horizontal_flip = True,
                                    vertical_flip = True,
                                    fill_mode = "nearest")


    #Create the unet model, with the same parameters the
    model = get_unet(input_img = Input((400, 400, 3), name='img'), n_filters=32, dropout=0.2, batchnorm=True)

    #Initialize global variables like BatchNormalization
    import keras as ks
    ks.backend.get_session().run(tf.global_variables_initializer())

    #Compile the model with the Adam Optimizer and set loss = f1_loss function, which would optimize for getting a better f1
    model.compile(optimizer= Adam(lr = 1e-3), loss=f1_loss, metrics=["accuracy", f1])
    model.summary()

    #This will save you only the best models under the name(best_model.h5), with respect to the f1_loss function
    callbacks = [ModelCheckpoint('../model/new_model.h5', verbose=1, monitor = "loss", save_best_only=True, save_weights_only=False)]

    #We run 15000 epochs, with 5 steps_per_epoch on a GPU which is described in the README, for getting our best model.
    #Since we are using a custom loss function u may not get the exact same result:(f1: 0.8313),
    #but you will certainly get an f1 that is bigger or smaller which will varry in from 0.82 to 0.835
    model.fit_generator(train_generator,
                        callbacks = callbacks,
                        steps_per_epoch=5,
                        epochs=12500)

    #Load the best model that was saved under ../model/new_model.h5
    model = ks.models.load_model('../model/new_model.h5', custom_objects={'f1_loss': f1_loss, "f1" : f1})

    #Read, adjust, predict and save the testing data
    files = build_save_test_data(path_test, path_to_save_images, model)

    #Create the submission
    masks_to_submission("../submissions/best_submission.csv", *files)

else:
    #OR use our best model
    best_model = ks.models.load_model('../model/best_model.h5', custom_objects={'f1_loss': f1_loss, "f1" : f1})
    files = build_save_test_data(path_test, path_to_save_images, best_model)
    masks_to_submission("../submissions/best_submission.csv", *files)

print("Done running")
