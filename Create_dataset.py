"""
loads the previous checkpoints and tests the model
"""
import tensorflow as tf
import utils
from models.Face_recognition import FR_model
from PIL import Image
import os
import argparse

# command line options
parser = argparse.ArgumentParser()
parser.add_argument('--path',
                            help="path to the dataset")
opt=parser.parse_args()

model=tf.keras.models.load_model('checkpoint2')
Images_un= utils.process2(opt.path)
mean=Images_un.mean()
std=Images_un.std()

Images=(Images_un-mean)/std
FR=FR_model()

Embeddings= FR(Images)

preds = model.predict(Embeddings)

os.mkdir("Class_Flicker")
os.mkdir('Class_Flicker/Fat')
os.mkdir('Class_Flicker/Thin')

count=0;
for i in range(len(preds)):
    curr_img=Images_un[i]
    curr_img=curr_img.astype('uint8')
    curr_img=Image.fromarray(curr_img)

    if(preds[i]<0.3):
        curr_img.save('Class_Flicker/Fat/'+str(i)+'.png')
        count+=1
    elif(preds[i]>=0.5):
        curr_img.save('Class_Flicker/Thin/' + str(i) + '.png')
        count+=1

print("{} no. of images were saved ".format(count))





