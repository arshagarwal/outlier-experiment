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
parser.add_argument('--l',type=int,default=0.5,help="lower limit")
parser.add_argument('--u',type=int,default=0.5,help="lower limit")
opt=parser.parse_args()

model=tf.keras.models.load_model('checkpoint2')
Images_un= utils.process2(opt.path,batch_size=80000)
mean=Images_un.mean()
std=Images_un.std()

Images=(Images_un-mean)/std
FR=FR_model()

Embeddings= FR(Images)
#assert len(Embeddings) == 70000, "only {} images were found".format(len(Embeddings))

preds = model.predict(Embeddings)
#assert len(preds) == 70000, "only {} images were found".format(len(preds))

os.mkdir("Class_Flicker")
os.mkdir('Class_Flicker/Fat')
os.mkdir('Class_Flicker/Thin')

count=0;
for i in range(len(preds)):
    curr_img=Images_un[i]
    curr_img=curr_img.astype('uint8')
    curr_img=Image.fromarray(curr_img)

    if(preds[i]<=opt.l):
        curr_img.save('Class_Flicker/Fat/'+str(i)+'.png')
        count+=1
    elif(preds[i]>=opt.u):
        curr_img.save('Class_Flicker/Thin/' + str(i) + '.png')
        count+=1

print("{} no. of images were saved ".format(count))





