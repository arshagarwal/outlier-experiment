"""
Code to tarin on the slim_dataset using outliers
"""
from models.Face_recognition import FR_model
import os
from PIL import Image
import numpy as np
from Options.slim_train_options import options
import tensorflow as tf
from models.model import model
import utils

opt=options().parser
s_e_freq=opt.s_e_freq

path='slim_dataset/Train_dataset'
train_set,y_train=utils.process(path)


FR=FR_model()
model=model()

# getting corresponding embeddings
train_set=FR(train_set)

optimizer=tf.keras.optimizers.Adam()
batch_size=opt.b_size
n_batches = int(len(train_set) / opt.b_size)

checkpoint=tf.train.Checkpoint(model=model,opt=optimizer)
manager=tf.train.CheckpointManager(checkpoint,directory='Checkpoints',max_to_keep=3)

if  opt.loss== 'mae':
    model.compile(optimizer=tf.keras.optimizers.Adam(),loss=tf.keras.losses.MeanAbsoluteError())
    model.fit(train_set,y_train,opt.b_size,epochs=opt.epochs,validation_split=0.1)
elif opt.loss== 'bce':
     y_train=y_train-0.1
     y_train=y_train/0.8
     model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.BinaryCrossentropy(),metrics=['accuracy'])
     model.fit(train_set, y_train, opt.b_size, epochs=opt.epochs, validation_split=0.1)



model.save('checkpoint2')