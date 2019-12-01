import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras import Model

class SimpleModel(Model):
    def __init__(self):
        super(SimpleModel, self).__init__('simeple_net')
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

class DiscreteTradingModel(Model):
    def __init__(self, action_set):
        super(DiscreteTradingModel, self).__init__('discrete_trading_net')
        # action_set = [-1,0,1]
        # -1 is short
        # 0 is neutral (no trade)
        # 1 is long
        self.action_set = action_set
        self.fc1 = Dense(units=128, activation='relu')
        self.batchnorm1 = BatchNormalization()
        self.dropout1 = Dropout(0.2)
        self.fc2 = Dense(units=64, activation='relu')
        self.batchnorm2 = BatchNormalization()
        self.dropout2 = Dropout(0.3)
        self.fc3 = Dense(units=32, activation='relu')
        self.Q = Dense(units=len(self.action_set), name='Q')
        self.saved_log_probs = []
        self.rewards = 0
        
    def call(self, inputs):
        # `inputs` is a numpy array. We convert to Tensor.
        x = tf.convert_to_tensor(inputs)

        # Our batch size is just 1 and x's shape is (observation size,)
        # We expand this shape to (1, observation size)
        x = tf.expand_dims(x, 0)
        
        x = self.fc1(x)
        # x = self.batchnorm1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        # x = self.batchnorm2(x)
        # x = self.dropout2(x)
        # x = self.fc3(x)
        x = self.Q(x)
        
        return x
