from PIL import Image
import numpy as np
import tensorflow as tf


def get_image(i):
    directory_name='crop_part1/'
    im = Image.open(directory_name + i)
    im=im.resize((160,160))
    im = np.asarray(im)
    return im

def add_samples(X,Y,n_samples,y_l,y_u,image_names):
    # adding samples from y_l to y_u
    for i in image_names:
        if len(X) >= n_samples:
            break;
        age = int(i.split('_')[0])
        if (age >= y_l and age <= y_u):
            image = get_image(i)
            X.append(image)
            Y.append(age)

def normalize(data):
    mean=data.mean()
    stddev=data.std()
    data=(data-mean)/stddev
    return data


def process(path, img_size=(160, 160), batch_size=2000):
    """
    Extracts and processes the images from the given path
    :param path: String that denotes the path of the directory where Images are stored
    :return: images,attributes.   a numpy array of the shape [n_images,128,128,3],atrribute vector of shape [n_images,number of attributes]
    """

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=normalize, )

    generator = datagen.flow_from_directory(
        path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary')
    (x, y) = next(generator)
    # normalizing Y vector
    y = y * 0.8 + 0.1
    return x, y