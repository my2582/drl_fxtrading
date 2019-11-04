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
config['epi_sz'] = 1440
config['split_sz'] = config['epi_sz'] + config['lag']
config['total_no_epi'] = 21
config['init_lr'] = 0.005
config['momentum'] = 0.9
#config['eval_sz'] = 30        
config['target_currency'] = 'audusd'
config['epochs'] = 1
config['obs_sz'] = 360  # It's features.shape[1] from "_, _, features = draw_episode(..)"
# config['timespan'] = 1440    # one full day (1440 minutes)

tf.keras.backend.set_floatx('float32')

X_train = pd.read_csv('./dataset/toy_X_train.csv', sep=',')
assert config['total_no_epi']*config['split_sz'] <= np.min([X_train[key].shape[0] for key in X_train.keys()]), "Training set has less data points than # of episodes * # of splits."

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

class Environment:
    def __init__(self, X, config, num_agents=1):
        '''
        Parameters:
            X(pd.DataFrame): one of X_train, X_val, or X_test
            config(dict): a dictionary of parameters
            num_agents(int): the total number of agents
        '''
        self.X = X
        self.ts = np.sort(self.X['timestamp'].unique()) # all timestamps
        self.ccy = self.X['ccy'].unique() # all currency pairs
        self.num_agents = num_agents
        self.config = config
        self.feature_span = None    # A matrix of features (episode size, feature size)
        self.step_counter = None    # the current step index within an episode
        self.epi_idx = None
        self.bid = None
        self.ask = None
        self.cur_wt = None          # The current investment weight in the target currency.
        self.rec_wt = None
        self.delta_wt = None
        self.state = None           # A tuple (observation, cur_wt). observation == a row of `feature_span`
        self.price_records = None    # A tuple (bid, ask, next_bid, next_ask)

        # A part of self.X, which is used to generate an episode
        # where ccy == the target currency 
        self.X_episode = None   

    def get_features_within_epi(self, step_counter):
        '''
        Parameters:
            step_counter(int): the current step index within an episode
        '''
        return self.bid[step_counter], self.ask[step_counter], self.feature_span[step_counter]


    def reset(self, epi_idx):
        '''
        Parameters:
            epi_idx(int): an episode index
        
        Returns:
            A tuple: (step_counter, state, price_record, if_finished)
        '''
        self.epi_idx = epi_idx

        # n is `n-th` episode
        # ep_sz is a length of one episode.
        # Within draw_episode(),
        # X[n:n+epi_sz] will be used to draw an episode.
        # Features are nomalized log returns with a 'lag'.
        self.bid, self.ask, self.feature_span, self.X_episode = \
            draw_episode(X=self.X, cur=self.config['target_currency'],
                        n=self.epi_idx, split_sz=self.config['split_sz'],
                        lag=self.config['lag'], ts=self.ts, ccy=self.ccy)
        self.step_counter = 0
        obs = self.feature_span[self.step_counter]
        self.cur_wt = 0.0
        # self.state = np.append(obs, self.cur_wt)

        self.step_counter += 1
        next_bid, next_ask, _ = self.get_features_within_epi(self.step_counter)

        self.price_records = (self.bid[0], self.ask[0], next_bid, next_ask)

        return self.step_counter, obs, 0, self.price_records, False
    
    def step(self, action):
        '''
        Parameters:
            action(float): a recommended investment weight in the target currency.
        
        Returns:
            A tuple: (step_counter, state, price_record, is_finished
        '''
        self.step_counter += 1

        # `is_finished` is true if it arrives at the end of the current episode.
        is_finished = (self.step_counter+2 == self.config['epi_sz'])
        
        self.rec_wt = action     # We take `action` as a recommended investment weight in the target currency.

        # delta_wt will be used when an agent trades. 
        self.delta_wt = self.rec_wt - self.cur_wt

        bid, ask, obs = self.get_features_within_epi(self.step_counter)
        next_bid, next_ask, _ = self.get_features_within_epi(self.step_counter + 1)

        # self.state = np.append(obs, self.rec_wt)
        self.price_records = (bid, ask, next_bid, next_ask)

        return self.step_counter, obs, self.delta_wt, self.price_records, is_finished




class SimpleDRLAgent:
    def __init__(self, model, config, init_capital = 10000):
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
        self.init_capital = 10000
        self.state = None
    
    # def _compute_loss(self, ep_rewards):
    #     return tf.reduce_sum(ep_rewards)

    def train(self):
        # Within a batch, for time stamp t we have:
        #  - rec_wt[t]: the recommended investment weight by nn
        #  - delta_wt[t]: a difference in weights between the recommended investment weight by nn
        #                 and current weight in the target currency.
        #  - rewards[t]: rewards from the latest recommended trade (caued by the recommended weight)
        rec_wt = np.empty((self.config['epi_sz'],))
        delta_wt = np.empty((self.config['epi_sz'],))
        rewards = np.empty((self.config['epi_sz'],))
        # observations = np.empty((self.config['batch_sz'],) + (self.config['obs_sz'],))
        
        # Accumucated rewards per episode
        epi_rewards = []

        # Average rewards per epoch
        epochs_avg_rewards = []

        # Training loop
        for epoch in range(self.config['epochs']):
            for epi_idx in range(self.config['epi_sz']):
                epi_rewards.append(0.0)

                # We start with 0% weight in the target currency.
                cur_wt = 0.
                
                t, self.state, delta, price_tuple, is_finished = env.reset(epi_idx)
                delta_wt[t] = delta

                ### start of batch ###
                # We loop through time stamps within an episode.
                while not is_finished:
                    # The neural networks take an observation, which is a set of features,
                    # and they return an action-like scalar [-1,1].
                    # This scalar is a recommended investment weight (rec_wt) in the target currency.
                    # `state` is a numpy array with (features, rec_wt)
                    action = self.model(self.state)
                    rec_wt[t] = action
                    t, self.state, delta, price_tuple, is_finished = env.step(action)
                    delta_wt[t] = delta

                    ### Do trading
                    # 

                    ### Get an immedidate reward
                    rewards[t] = 1

                    ### Accumulate rewards within this episode.
                    # epi_rewards[-1] += rewards[t]
                ### end of all batches ###
                losses = self.model.train_on_batch(self.state, rewards)

                print('Epoch:{} Episode:{}. The training loss is {}'.format(epoch+1, epi_idx+1, losses))
            
            ### enf of one epoch ##
            epochs_avg_rewards.append(np.average(np.array(epi_rewards[-1])))
        
        return epi_rewards, epochs_avg_rewards

model = SimpleModel()
env = Environment(X_train, config, num_agents=1)
simple_agent = SimpleDRLAgent(model, config)
epi_rewards, epochs_avg_rewards = simple_agent.train()
print(epi_rewards, epochs_avg_rewards)

