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

# User packages
from model import SimpleModel
from environment import Environment
from agent import SimpleDRLAgent


logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


config = {}
config['lag'] = 10
config['epi_sz'] = 180
config['split_sz'] = config['epi_sz'] + config['lag']
config['total_no_epi'] = 50  # floor(91057/1460), 91057 == # rows of any currency.
config['lr_rate'] = 0.001
config['momentum'] = 0.9
config['target_currency'] = 'usdjpy'
config['epochs'] = 1
# config['features_sz'] = 2*config['epi_sz']  # It's features.shape[1] from "_, _, features = draw_episode(..)"
config['buffer_sz'] = 50000
config['batch_sz'] = 64
config['num_batches'] = 100
config['start_learning'] = 5000
config['M'] = 9     # The number of currency pairs to invest in.

# First '1' is a space for 'timestamp'
# Last '2' is a spcae for [rec_wt, delta_wt]
config['obs_sz'] = 1 + config['epi_sz'] + 2
tf.keras.backend.set_floatx('float32')

X_train = pd.read_csv('./dataset/toy_X_train.csv', sep=',')
X_val = pd.read_csv('./dataset/toy_X_val.csv', sep=',')
assert config['total_no_epi']*config['split_sz'] <= np.min([X_train[key].shape[0] for key in X_train.keys()]), "Training set has less data points than # of episodes * # of splits."




# action_set
# 0  short
# 1  neutral
# 2  long
action_set = [0,1,2]
model = SimpleModel(action_set)
env = Environment(X_train, config, num_agents=1)
simple_agent = SimpleDRLAgent(model, env, config, verbose=False)
epi_rewards, epi_rewards_mean_rev = simple_agent.train()
print(epi_rewards, epi_rewards_mean_rev)

env_val = Environment(X_val, config, num_agents=1)
simple_agent_val = SimpleDRLAgent(model, env_val, config, verbose=False)
epi_rewards_val, epi_rewards_mean_rev_val = simple_agent_val.val()
print(epi_rewards_val, epi_rewards_mean_rev_val)