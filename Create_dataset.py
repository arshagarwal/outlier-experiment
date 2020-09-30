import tensorflow as tf
import utils
from models.Face_recognition import FR_model
from PIL import Image
import os
import argparse
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--path',
                            help="path to the dataset"))
parser.add_argument('--length',type=int,default=70000,help='length of dataset')
parser.add_argument('--l',type=float,default=0.5,help="lower limit")
parser.add_argument('--u',type=float,default=0.5,help="lower limit")
opt=parser.parse_args()

sampling_size=100
gen=utils.process2(opt.path, batch_size=sampling_size)
model=tf.keras.models.load_model('checkpoint2')

FR=FR_model()

os.mkdir("Dataset")
os.mkdir("Dataset/Fat")
os.mkdir("Dataset/Thin")

subdirs = [name for name in os.listdir(".") if os.path.isdir(name)]

for subdir in subdirs:
    
    listing = os.listdir("./"+subdir)

    for file in listing:

        img = Image.read(subdir+file)

        img_1 = img.copy()
        img_1.resize((128, 128))

        mean = img_1.mean()
        std = img_1.std()

        img_1 = (img_1 - mean)/std
        img_1 = img_1.reshape((1, 128, 128, 3))
        embedding = FR(img_1)
        pred = model.predict(embedding)

        if pred[0] < l:
            im.save("Dataset/Fat"+filename)
        elif pred[0] > r:
            im.save("Dataset/Thin"+filename)






