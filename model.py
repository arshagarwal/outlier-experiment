import tensorflow as tf
class model(tf.keras.Model):

    def __init__(self):
        super(model, self).__init__();
        input=tf.keras.Input(shape=(128,))
        x=tf.keras.layers.Dense(64,activation='relu')(input)
        x=tf.keras.layers.BatchNormalization()(x)
        x=tf.keras.layers.Dense(32,activation='relu')(x)
        x=tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(1, activation='linear')(x)
        self.c_model=tf.keras.Model(input,x)