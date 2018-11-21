import tensorflow as tf

'''
The Qnetwork class implement the deep-q-network
We use a 2 fully-connected layer for simplicity
TODO: Change to 4 layers if time permits
'''
class QNetwork(tf.keras.Model):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.dense_1 = tf.keras.layers.Dense(256, activation='relu')
        self.dense_2 = tf.keras.layers.Dense(256, activation='relu')
        self.dense_3 = tf.keras.layers.Dense(12, activation='softmax')

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.dense_1(x)
        x = self.dense_2(x)
        return self.dense_3(x)