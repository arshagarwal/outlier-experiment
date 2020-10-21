import cv2
import scipy.io
import os
import pandas as pd
from PIL import Image
import numpy as np
import random
import argparse
from models.model import model
import time


def get_images(batch_size, add, img_size=(160, 160), path=''):
    """

    :param batch_size:
    :param add: array of names
    :param img_size:
    :return:
    """
    rand = random.sample(range(0, len(add)), batch_size)
    X = []
    y = []
    for i in rand:
        """
        image = cv2.imread(add[i])
        curr_img = cv2.resize(image, img_size, interpolation=cv2.INTER_CUBIC)
        curr_img = curr_img.astype('float64')
        mean = curr_img.mean()
        stddev = curr_img.stddev()
        curr_img = (curr_img - mean)/stddev
        """
        image = Image.open(path + '/' + add[i])
        image = image.resize(img_size)
        curr_img = np.asarray(image)
        curr_img = curr_img / 127.5
        curr_img = curr_img - 1
        curr_age = (int)(add[i].split('_')[0])
        X.append(curr_img)
        y.append(curr_age)

    return X, y


def train(args):
    # Pre-processing
    Address = os.listdir(args.img_dir)

    # Training
    if args.pre_trained == 'facenet':
        from models.Face_recognition import FR_model
        FR = FR_model()
        Model = model()
        Model.compile(loss='mean_absolute_error', optimizer='adam')
        # training loop
        length = len(Address)
        print("length are {}".format(length))
        assert length > 0
        batch_size = args.batch_size
        n_batches = length // batch_size
        epochs = args.epochs
        iters = (int)(epochs * n_batches)
        assert iters > 0
        print("iters are {}".format(iters))
        start_time = time.time()
        for i in range(iters):
            X, Y = get_images(batch_size, Address, (args.img_size, args.img_size), args.img_dir)
            X = np.array(X)
            Y = np.array(Y)
            X = FR(X)

            assert X.shape == (batch_size, 128), 'expected shape {} O/p shape {}'.format((batch_size, 128), X.shape)
            history = Model.fit(X, Y, batch_size, 1, verbose=0)
            time_taken = (int)(time.time() - start_time)
            if (i + 1) % args.log_step == 0:
                print("Time: {}s Iters [{}/{}] Loss {} Batch size {}   ".format(time_taken, i + 1, iters, history.history['loss'],
                                                                      args.batch_size))

        Model.save(args.save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--pre_trained', type=str, default='facenet', help='pre-trained model to be usedx')
    parser.add_argument('--img_size', type=int, default=160, help='size of image to be fed to the model')
    parser.add_argument('--batch_size', type=int, default=50, help='batch s9ize to be used')
    parser.add_argument('--epochs', type=float, default=2, help='number of epochs to be used')
    parser.add_argument('--log_step', type=int, default=50, help='number of steps to be taken before logging')
    parser.add_argument('--save_path', type=str, default='Model_checkpoint',
                        help='path of dir where model is to be saved')
    parser.add_argument('--img_dir', type=str, default='UTK', help='path of training img directory')

    args = parser.parse_args()
    train(args)
