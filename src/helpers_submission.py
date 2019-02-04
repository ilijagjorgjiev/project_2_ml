#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.image as mpimg
import re
from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


foreground_threshold = 0.25 # percentage of pixels > 1 required to assign a foreground label to a patch

# assign a label to a patch
def patch_to_label(patch):
    df = np.mean(patch)
    if df > foreground_threshold:
        return 1
    else:
        return 0


def mask_to_submission_strings(image_filename):
    """Reads a single image and outputs the strings that should go into the submission file"""
    img_number = int(re.search(r"\d+", image_filename).group(0))
    im = np.load(image_filename)
    patch_size = 16
    for j in range(0, im.shape[1], patch_size):
        for i in range(0, im.shape[0], patch_size):
            patch = im[i:i + patch_size, j:j + patch_size]
            label = patch_to_label(patch)
            yield("{:03d}_{}_{},{}".format(img_number, j, i, label))


def masks_to_submission(submission_filename, *image_filenames):
    """Converts images into a submission file"""
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for fn in image_filenames[0:]:
            f.writelines('{}\n'.format(s) for s in mask_to_submission_strings(fn))



def build_save_test_data(path_test, path_to_save_images, model):

    #param path_test => path to the testing set
    #param path_to_save_images => path on where to save the predicted images(numpy arrays)
    #param model => the model that was trained, that will be used for predicting

    #Load the test data and save it into an numpy array and adjust it the same way the input data was adjusted
    test_data = np.zeros((50,608,608,3))
    for idx in range(1,51,1):
        test_data[idx-1] = load_img(path_test+"test_"+str(idx)+".png")
    test_data = test_data / 255

    #Predict for each image, we consider the image
    for i in range(0, 50):
        predict(test_data[i], path_to_save_images+ "test_"+str((i+1)), model)

    files = []
    #Get the paths of each npy array, such that we can feed it to the helpers method for submission that was provided to us.
    for i in range(1,51):
        img = path_to_save_images + "test_"+str((i)) + ".npy"
        files.append(img)

    return files

#Since our modified U-Net model is trained to accept
#400x400 sized images, we use this function to generate predictions for the test images which are of size 608 x 6008.
#The functions tiles 608x608 testing image with 4 partially overlapping
#400x400 tiles. Pixels within the overlaps have their different
#output values averaged.
def predict(input_img, filename, model):
    #input_img => numpy array of adjusted pixels
    #filename => path and filename where to be saved.
    #model => model to be used for predictions

    output_A = model.predict(input_img[    :400,   :400].reshape((1,400,400,3)))
    output_B = model.predict(input_img[-400:   ,   :400].reshape((1,400,400,3)))
    output_C = model.predict(input_img[-400:   ,-400:  ].reshape((1,400,400,3)))
    output_D = model.predict(input_img[    :400,-400:  ].reshape((1,400,400,3)))

    output = np.zeros((608,608,1))
    output[    :400,   :400] = output[    :400,   :400] + output_A
    output[-400:   ,   :400] = output[-400:   ,   :400] +output_B
    output[-400:   ,-400:  ] = output[-400:   ,-400:  ] +output_C
    output[    :400,-400:  ] = output[    :400,-400:  ] +output_D

    output[-400:400,:]  = output[-400:400,:]/2
    output[:,-400:400]  = output[:,-400:400]/2

    np.save(filename, output)
