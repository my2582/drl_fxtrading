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
from agents.agent import DQNAgent
from environments.environment_old import Environment


logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


config = {}
## setting for a toy dataset
# config['lag'] = 5
# config['epochs'] = 10
# config['epi_sz'] = 16   # epi_sz is equal to the number of possible time steps in one epidoe.
# config['total_no_epi'] = 20  # floor(91057/1460), 91057 == # rows of any currency.
# config['num_batches'] = 5
# config['batch_sz'] = 8
#######################
## setting for a real dataset
config['lag'] = 10
config['epochs'] = 20
config['epi_sz'] = 30   # epi_sz is equal to the number of possible time steps in one epidoe.
config['total_no_epi'] = 30  # floor(91057/1460), 91057 == # rows of any currency.
config['num_batches'] = 10
config['batch_sz'] = 16
#######################
config['state_sz'] = 1 + 9*config['lag']   # state = [time step, 9*lag log returns]
config['split_sz'] = config['epi_sz'] + config['lag']
config['lr_rate'] = 0.001
config['momentum'] = 0.9

config['bins'] = np.concatenate([[-10], np.arange(-1e-4, 1.1e-4, 1e-5), [10]])
config['channels'] = 30
config['buffer_sz'] = 50000

assert config['epi_sz'] > config['batch_sz']

# config['start_learning'] = 5000
config['action_set'] = [-1, 0, 1]
config['M'] = 1     # The number of the target currency pairs to invest in.
# config['N'] = 23    # The number of bins in a market-image matrix
# config['C'] = 30    # The number of channels in a market-image matrix
config['obs_sz'] = 183

X_train = pd.read_csv('../dataset/toy_X_train_close.csv', sep=',')
X_val = pd.read_csv('../dataset/toy_X_val_close.csv', sep=',')
assert config['total_no_epi']*config['split_sz'] <= np.min([X_train[key].shape[0] for key in X_train.keys()]), "Training set has less data points than # of episodes * # of splits."

# ccy_list = ['gbpusd', 'eurusd', 'chfusd', 'nokusd', 'sekusd', 'cadusd', 'audusd', 'nzdusd', 'jpyusd']
ccy_list = ['eurusd']
result_path = './results/milestone/'
for ccy in ccy_list:
    config['target_currency'] = ccy
    env = Environment(X_train, config, num_agents=1)
    dqn_agent = DQNAgent(env, config, verbose=True)
    epi_rewards = dqn_agent.train()
    # drl_model = pd.DataFrame(epi_rewards)
    # drl_model.to_csv(result_path+'train_drl_'+ccy+'.csv', index=False, index_label=False, header=False)
    print('epi_rewards:', epi_rewards)


    # config['target_currency'] = ccy
    # model = SimpleModel()
    # env = Environment(X_train, config, num_agents=1)
    # simple_agent = SimpleDRLAgent(model, env, config, verbose=False)
    # epi_rewards, epi_rewards_mean_rev, losses = simple_agent.train()
    # drl_model = pd.DataFrame(epi_rewards)
    # mean_rev_model = pd.DataFrame(epi_rewards_mean_rev)
    
    # losses_drl = pd.DataFrame(losses)
    # drl_model.to_csv(result_path+'train_drl_'+ccy+'.csv', index=False, index_label=False, header=False)
    # mean_rev_model.to_csv(result_path+'train_mean_rev_'+ccy+'.csv', index=False, index_label=False, header=False)
    # losses_drl.to_csv(result_path+'drl_losses_'+ccy+'.csv', index=False, index_label=False, header=False)
    # print(epi_rewards, epi_rewards_mean_rev)

    # env_val = Environment(X_val, config, num
    # _agents=1)
    # simple_agent_val = SimpleDRLAgent(model, env_val, config, verbose=False)
    # epi_rewards_val, epi_rewards_mean_rev_val = simple_agent_val.val()
    # drl_model = pd.DataFrame(epi_rewards_val)
    # mean_rev_model = pd.DataFrame(epi_rewards_mean_rev_val)
    # drl_model.to_csv(result_path+'test_drl_'+ccy+'.csv', index=False, index_label=False, header=False)
    # mean_rev_model.to_csv(result_path+'test_mean_rev_'+ccy+'.csv', index=False, index_label=False, header=False)
    # print(epi_rewards_val, epi_rewards_mean_rev_val)