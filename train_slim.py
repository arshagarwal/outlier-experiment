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


for i in range(1,opt.epochs+1):
    loss_t=0
    it=0
    for j in range(n_batches):
        with tf.GradientTape() as tape:
            tape.watch(model.trainable_variables)
            curr=train_set[it:it+batch_size]
            forward=model(curr)
            if opt.loss=='mse':
                loss=tf.keras.losses.MeanSquaredError()(y_train[it:it+batch_size],forward)
            elif opt.loss=='mae':
                loss=tf.keras.losses.MeanAbsoluteError()(y_train[it:it+batch_size],forward)
            else:
                raise ValueError("Loss function not recognized ")
            loss_t += loss

        grads=tape.gradient(loss,model.trainable_variables)
        optimizer.apply_gradients(zip(grads,model.trainable_variables))
        it+=batch_size

    loss_t=loss_t.numpy()
    loss_t /= n_batches
    print("Epoch : {} Loss: {}".format(i, round(loss_t,4)))
    # saving checkpoints
    if i%s_e_freq == 0 or i == (opt.epochs-1):
        manager.save(checkpoint_number=i)





