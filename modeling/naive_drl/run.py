from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd

# utils
import time

# TensorFlow
import tensorflow as tf
import tensorflow.keras.optimizers as kop
import tensorflow.keras.losses as kls
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model

# logging
import logging 

from features import draw_episode


logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)



config = {}
config['lag'] = 20
config['split_sz'] = 100       # split size
config['batch_sz'] = config['split_sz'] - config['lag']
config['epi_sz'] = 30          # the number of splits
config['init_lr'] = 0.005
config['momentum'] = 0.9
#config['eval_sz'] = 30        # The number of valida
config['target_currency'] = 'audusd'
config['epochs'] = 1
config['obs_sz'] = 360  # It's features.shape[1] from "_, _, features = draw_episode(..)"
# config['timespan'] = 1440    # one full day (1440 minutes)

tf.keras.backend.set_floatx('float32')

X_train = pd.read_csv('./dataset/toy_X_train.csv', sep=',')
assert config['epi_sz']*config['split_sz'] <= np.min([X_train[key].shape[0] for key in X_train.keys()]), "Training set has less data points than # of episodes * # of splits."

class SimpleModel(Model):
    def __init__(self):
        super(SimpleModel, self).__init__('simple_net')
        self.fc1 = Dense(units=256, activation='relu')
        self.fc2 = Dense(units=64, activation='relu')
        self.value = Dense(1, activation='tanh', name='value')
        
        self.saved_log_probs = []
        self.rewards = 0
        
    def call(self, inputs):
        # `inputs` is a numpy array. We convert to Tensor.
        x = tf.convert_to_tensor(inputs)

        # Our batch size is just 1 and x's shape is (observation size,)
        # We expand this shape to (1, observation size)
        x = tf.expand_dims(x, 0)
        
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.value(x)
        
        return x

class SimpleDRLAgent:
    def __init__(self, model, config):
        self.config = config
        self.config['batch_sz'] = self.config['epi_sz'] - self.config['lag']
        self.model = model
        self.model.compile(
                optimizer=kop.SGD(
                    learning_rate=self.config['init_lr'],
                    momentum=self.config['momentum'],
                    name='SGD'),
                # deifne a loss function
                loss = 'mean_absolute_error'
            )
    
    # def _compute_loss(self, ep_rewards):
    #     return tf.reduce_sum(ep_rewards)

    def train(self):
        # We store an array of actions, rewards and obervations for each batch
        actions = np.empty((self.config['batch_sz'],))
        rewards = np.empty((self.config['batch_sz'],))
        observations = np.empty((self.config['batch_sz'],) + (self.config['obs_sz'],))
        
        # Accumucated rewards per episode
        epi_rewards = []

        # Average rewards per epoch
        epochs_avg_rewards = []

        # Training loop
        for epoch in range(self.config['epochs']):
            for epi_idx in range(self.config['epi_sz']):
                epi_rewards.append(0.0)
                prev_action = 0.
                
                # n is `n-th` episode
                # ep_sz is a length of one episode.
                # in draw_episode(), X[n:n+epi_sz+1] will be used.
                
                # For a lag of 20,
                # feature_batch[0] := [20 target's log return(bid prices),
                #                      20 target's log return(ask prices),
                #                     8*20 others' log return(bid prices),
                #                     8*20 others' log return(ask prices)]
                #     at the oldest time stamp.
                # (feature_batch[-1] has features at the latest)
                # Features are nomalized within draw_episode()
                target_bid, target_ask, feature_batch = \
                    draw_episode(X=X_train, cur=self.config['target_currency'],
                                n=epi_idx, epi_size=self.config['epi_sz'],
                                lag=self.config['lag'])
                assert feature_batch.shape[-1] == self.config['obs_sz'], "Found wrong config['obs_sz'], which must be the same as len(feature_batch[0])"
                assert len(feature_batch) == self.config['batch_sz'], "Found wrong config['batch_sz'], which must be the same as len(feature_batch)"
                bid, ask = target_bid[self.config['lag']:] * 1e3, \
                                target_ask[self.config['lag']:]*1e3
                
                ### start of batch ###
                # We loop through time stamps in order within an episode.
                for t, obs in enumerate(feature_batch):                    
                    # The neural networks returns an action-like value
                    # `state` is a numpy array.
                    action_by_nets = self.model(obs)
                    
                    # An actual action to take should effectively cause a trade.
                    # Therefore, we subtract `prev_action_by_nets`
                    # from `action_by_nets` and take it as our `action`
                    actions[t] = action_by_nets - prev_action
                    price = 0.
                    if actions[t] > 0.:
                        price = ask[t]  # We buy
                    elif actions[t] < 0.:
                        price = bid[t]  # We sell
                    
                    # This is an immediate reward caused by taking an action
                    rewards[t] = -1.0*actions[t]*price
                    
                    # Accumulated rewards gained in a batch and
                    # store at at the end of epi_rewards[]
                    epi_rewards[-1] += rewards[t]
                    
                    prev_action = action_by_nets
                ### end of all batches ###
                losses = self.model.train_on_batch(feature_batch, rewards)

                print('Epoch:{} Episode:{}. The training loss is {}'.format(epoch+1, epi_idx+1, losses))
            
            ### enf of one epoch ##
            epochs_avg_rewards.append(np.average(np.array(epi_rewards[-1])))
        
        return epi_rewards, epochs_avg_rewards

model = SimpleModel()
simple_agent = SimpleDRLAgent(model, config)
epi_rewards, epochs_avg_rewards = simple_agent.train()
print(epi_rewards, epochs_avg_rewards)

