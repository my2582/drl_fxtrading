import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import Model

class SimpleModel(Model):
    def __init__(self):
        super(SimpleModel, self).__init__('simple_net')
        self.fc1 = Dense(units=128, activation='relu')
        self.dropout = Dropout(0.2)
        self.fc2 = Dense(units=32, activation='relu')
        self.value = Dense(units=1, activation='tanh', name='value')
        
        self.saved_log_probs = []
        self.rewards = 0
        
    def call(self, inputs):
        # `inputs` is a numpy array. We convert to Tensor.
        x = tf.convert_to_tensor(inputs)

        # Our batch size is just 1 and x's shape is (observation size,)
        # We expand this shape to (1, observation size)
        x = tf.expand_dims(x, 0)
        
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.value(x)
        
        return x
