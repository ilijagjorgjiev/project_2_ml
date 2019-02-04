from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
from itertools import chain

#adjusts each pair of img and mask
def adjustData(img, mask):
    #param img => the corresponding image
    #param mask => the corresponding mask/groundtruth

    #Divide each pixel by 255, making all pixels have vlalues between 0 & 1
    if(np.max(img) > 1):
        img = img / 255
        mask = mask /255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
        return (img, mask)

#Apply the adjustData function to each pair of image and mask/groundtruth
def trainGenerator(train_generator):
    #param trainGenerator => an interator that has zipped corresponding images and masks/groundtrut
    for (img,mask) in train_generator:
        img,mask = adjustData(img,mask)
        yield (img,mask)

def read_data_and_augmentation(path_images = "../data/training_set/",
                                path_mask = "../data/training_set/",
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
                                fill_mode = "nearest"):
        #param path_images => path to the images folder, in our case: ../data/training_set/
        #param path_mask => path to the mask folder, in our case: ../data/training_set/
        #param class_img => class where to find the images, in our case: "images"
        #param class_mask => class where to find the mask/groundtruth, in our case: "groundtruth"
        #param img_size => size of the image, must be same of the mask
        #param mask_size => size of the mask/groundtruth, must be the same of the image
        #param batch_size => size of batches
        #param seed => seed
        #param image_color_mode => color_mode: one of "grayscale", "rbg". Default: "rgb". Whether the images will be converted to have 1 or 3 color channels.
        #param mask_color_mode => default "grayscale"
        #param rotation_range => Int: Degree range for random rotations.
        #param shear_range =>  Float: Shear Intensity (Shear angle in counter-clockwise direction in degrees)
        #param zoom_range => Float: Zoom Range for each image
        #param fill_mode: One of {"constant", "nearest", "reflect" or "wrap"}.
        #Example of what fill_mode does: 'constant': kkkkkkkk|abcd|kkkkkkkk (cval=k), 'nearest': aaaaaaaa|abcd|dddddddd , 'reflect': abcddcba|abcd|dcbaabcd , 'wrap': abcdabcd|abcd|abcdabcd
        #param horizontal_flip => Boolean: Randomly flip input horizontally.
        #param vertical_flip => Boolean: Randomly flip input vertically.

        #Creates a dictionary with all the parameters, which will be applied to the data wiith real-time data augmentation.
        data_gen_args = dict(rotation_range = rotation_range,
        shear_range = shear_range,
        zoom_range = zoom_range,
        horizontal_flip = horizontal_flip,
        vertical_flip = vertical_flip,
        fill_mode= fill_mode )

        #Definition of ImageDataGenerator: Generate batches of tensor image data with real-time data augmentation. The data will be looped over (in batches).


        #Create an ImageDataGenerator both for the images and the groundtruth with the same
        image_datagen = ImageDataGenerator(**data_gen_args)
        mask_datagen = ImageDataGenerator(**data_gen_args)

        #Read the images from the specified directory, apply the data_gen_args to each img
        image_generator = image_datagen.flow_from_directory(
        path_images,
        classes = [class_img],
        class_mode=None,
        target_size = img_size,
        color_mode = image_color_mode,
        batch_size = batch_size,
        seed=seed)

        #Read the images from the specified directory, apply data_gen_args to each mask
        mask_generator = mask_datagen.flow_from_directory(
        path_mask,
        classes = [class_mask],
        target_size = mask_size,
        color_mode = mask_color_mode,
        batch_size = batch_size,
        class_mode=None,
        seed=seed)

        # combine generators into one which yields image and masks
        train_generator = zip(image_generator, mask_generator)

        #Adjust each corresponding pair of img, mask with adjustData function
        train_generator = trainGenerator(train_generator)

        #return it ready for training
        return train_generator
