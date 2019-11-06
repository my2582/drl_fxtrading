import numpy as np

import tensorflow as tf
import tensorflow.keras.optimizers as kop
import tensorflow.keras.losses as kls

class PortfolioManagement:
    def __init__(self, M=9, P0=10000):
        '''
        Parameters:
            M(int): the number of assets (excluding cash)
            P0(int): an initial portfolio value (=usually in cash)
        '''

        self.pf_wt = np.zeros((1, M+1))
        self.pf_wt[0][0] = 1    # cash 100% at time 0

        

class SimpleDRLAgent:
    def __init__(self, model, env, config, action_set = [-1, 0, 1], init_wealth = 10000, mode='train', verbose=False, test_model_path = None):
        self.config = config
        self.env = env

        if mode == 'train':
            self.mode = 'train'
            self.lr_rate = config['lr_rate']
            self.buffer_sz = config['buffer_sz']
            self.epsilon = 0.97
            self.discount = 0.99
            # self.batch_sz = config['batch_sz']
            # self.num_batches = config['num_batches']
            # self.start_learning = config['start_learning']

        self.model = model
        self.model.compile(
                optimizer=kop.Adam(learning_rate=self.lr_rate),
                # deifne a loss function
                # loss = 'mean_absolute_error'
                loss=[self._compute_loss],
                # metrics=['accuracy']
            )
        self.action_set = action_set
        self.wealth = init_wealth
        self.state = None
        self.verbose = verbose

    def _compute_loss(self, obs_history, immed_rewards):
        return -tf.reduce_sum(immed_rewards)


    def _epsilon_greedy_action(self, action, epsilon):
        if tf.random.uniform(shape=(1,))[0] > epsilon:
            # Take a random action
            return np.random.randint(-100,100)/100
        
        return action
    
    def act(self, obs_history, action_history):
        # obs_history[-1,] == the latest state (timestamp, features, rec_wt, delta_wt)
        state = obs_history[-1,:]

        action = self.model(state)
        # arg: 0 short, 1 neutral, 2 long 100%
        # choose_action = tf.argmax(action, axis=-1)

        return action

    def get_immed_reward(self, buysell, cost_bid, cost_ask, trade_bid, trade_ask):
        # cost_mid_price = (cost_bid + cost_ask)/2
        # trade_mid_price = (trade_bid + trade_ask)/2
        
        if trade_bid > cost_bid:
            # Price is up
            if buysell == 'B':
                # our action is correct in terms of betting direction
                reward = 1
            else:
                reward = -1
        elif trade_bid < cost_bid:
            # Price is down
            if buysell == 'B':
                # wrong in betting direction
                reward = -1
            else:
                reward = 1
        else:
            if buysell == 'N':
                reward = 1
            else:
                reward = -1

        return reward

    def train(self):
        # We keep `inv_wt` within this agent class as an attribute.
        self.inv_wt = np.zeros((self.config['epi_sz'],))

        # Within a batch, for time stamp t we have:
        rec_wt = np.zeros((self.config['epi_sz'],))
        delta_wt = np.zeros((self.config['epi_sz'],))
        immed_rewards = np.zeros((self.config['epi_sz'],))
        accum_rewards = np.zeros((self.config['epi_sz'],))

        # action_history = np.empty((self.config['epi_sz'],))
        obs_history = np.zeros((self.config['epi_sz'],) + (self.config['obs_sz'],))
        
        # Accumucated rewards per episode
        # epi_rewards = []

        # Average rewards per epoch
        epochs_total_rewards = []

        
        # mean reversion agent
        mean_rev_lookback = 20

        # Training loop
        for epoch in range(self.config['epochs']):
            for epi_idx in range(self.config['total_no_epi']):
                # We keep `inv_wt` within this agent class as an attribute.
                self.inv_wt = np.zeros((self.config['epi_sz'],))

                # Within a batch, for time stamp t we have:
                rec_wt = np.zeros((self.config['epi_sz'],))
                delta_wt = np.zeros((self.config['epi_sz'],))
                immed_rewards = np.zeros((self.config['epi_sz'],))
                accum_rewards = np.zeros((self.config['epi_sz'],))
                mean_rev_accum_rewards = np.zeros((self.config['epi_sz'],))


                # action_history = np.empty((self.config['epi_sz'],))
                obs_history = np.zeros((self.config['epi_sz'],) + (self.config['obs_sz'],))

                # We start with 0% weight in the target currency.
                
                t, state, price_tuple, is_finished = self.env.reset(epi_idx)

                #  - rec_wt[t]: the recommended investment weight by nn
                #  - delta_wt[t]: a difference in weights between the recommended investment weight by nn
                #                 and current weight in the target currency.
                #  - rewards[t]: rewards from the latest recommended trade (caued by the recommended weight)
                # features = state[:-2]
                rec_wt[t] = state[-2]
                self.inv_wt[t] = rec_wt[t]
                delta_wt[t] = state[-1]
                # immed_rewards[t] = 0.
                # accum_rewards[t] = 0.
                
                cur_bid, cur_ask, _, _ = price_tuple

                # We have observed and not taken any action yet.
                obs = np.append(t, state)
                obs_history[t] = obs

                # action_history is a list of recommneded weights
                # We start with zero weight
                action_history = [0.]

                # buysell_history is a list of 'B'(buy), 'S'(sell), 'N' (No trade)
                buysell_history = ['N']

                ### start of batch ###
                # We loop through time stamps within an episode.
                while not is_finished:
                    # The neural networks take observation history and action history as inputs
                    # and they return an action-like scalar [-1,1].
                    # This scalar is a recommended investment weight (rec_wt) in the target currency.
                    # `state` is a numpy array (features, rec_wt, delta_wt)
                    # The nueral networks are called during self.act() being executed
                    action = self.act(obs_history, action_history)
                    action_history.append(action)

                    t, state, price_tuple, is_finished = self.env.step(action, self.inv_wt[t])
                    # features = state[:-2]
                    rec_wt[t] = state[-2]
                    delta_wt[t] = state[-1]
                    obs = np.append(t, state)
                    obs_history[t] = obs

                    # Since we ran 'env.stop()', step counter increased by 1.
                    # Therefore, price_tuple[0] is interpreted as next_bid
                    next_bid, next_ask, _, _ = price_tuple

                    ### Trading begins
                    # buy/sell decision
                    # buysell = 'B' if delta_wt[t] >=0 else 'S'
                    # if delta_wt[t] > 0:
                    #     buysell = 'B'
                    # elif delta_wt[t] == 0:
                    #     buysell = 'N'
                    # else:
                    #     buysell = 'S'
                    if action > 0:
                        buysell = 'B'
                    elif action == 0:
                        buysell = 'N'
                    else:
                        buysell = 'S'
                    buysell_history.append(buysell)

                    # change an investment weight
                    self.inv_wt[t] = self.inv_wt[t-1] + delta_wt[t]

                    # profit/loss calculation
                    # pass

                    ### Get an immedidate reward
                    immed_rewards[t] = self.get_immed_reward(buysell, cur_bid[t-1], cur_ask[t-1], next_bid[0], next_ask[0])

                    ### Accumulate rewards within this episode.
                    accum_rewards[t] = accum_rewards[t-1] + immed_rewards[t]


                    ####### mean reversion trading #########
                    if t >= mean_rev_lookback:
                        mean = np.average(cur_bid[t-mean_rev_lookback:t])
                        std = np.std(cur_bid[t-mean_rev_lookback:t])
                        z_score = (cur_bid[t-1] - mean)/std
                        if z_score < -2:
                            mean_rev_bs = 'B'
                        elif z_score > 2:
                            mean_rev_bs = 'S'
                    else:
                        mean_rev_bs = 'N'

                    mean_rev_immed_rewards = self.get_immed_reward(mean_rev_bs, cur_bid[t-1], cur_ask[t-1], next_bid[0], next_ask[0])
                    mean_rev_accum_rewards[t] = mean_rev_accum_rewards[t-1] + mean_rev_immed_rewards
                    ########################################
                    if t % 100 == 0:
                        if self.verbose:
                            print('Time {}. Action(rec_wt) is {}. Reward is {}. Bid price goes from {} to {}'.format(t, action, immed_rewards[t], cur_bid[t-1], next_bid[0]))

                ### end of all batches ###
                losses = self.model.train_on_batch(obs_history, immed_rewards)

                print('Epoch:{} Episode:{}. The training loss is {}. The DRL total reward is {} vs A baseline total reward is {}. DRL - Baseline = {}'.format(epoch+1, epi_idx+1, losses, accum_rewards[t], mean_rev_accum_rewards[t], accum_rewards[t] - mean_rev_accum_rewards[t]))
            
            ### enf of one epoch ##
            epochs_total_rewards.append(accum_rewards[-1])
        
        return accum_rewards, mean_rev_accum_rewards


    def val(self):
        # We keep `inv_wt` within this agent class as an attribute.
        self.inv_wt = np.zeros((self.config['epi_sz'],))

        # Within a batch, for time stamp t we have:
        rec_wt = np.zeros((self.config['epi_sz'],))
        delta_wt = np.zeros((self.config['epi_sz'],))
        immed_rewards = np.zeros((self.config['epi_sz'],))
        accum_rewards = np.zeros((self.config['epi_sz'],))

        # action_history = np.empty((self.config['epi_sz'],))
        obs_history = np.zeros((self.config['epi_sz'],) + (self.config['obs_sz'],))
        
        # Accumucated rewards per episode
        # epi_rewards = []

        # Average rewards per epoch
        epochs_total_rewards = []

        
        # mean reversion agent
        mean_rev_lookback = 20

        # Training loop
        for epoch in range(self.config['epochs']):
            for epi_idx in range(self.config['total_no_epi']):
                # We keep `inv_wt` within this agent class as an attribute.
                self.inv_wt = np.zeros((self.config['epi_sz'],))

                # Within a batch, for time stamp t we have:
                rec_wt = np.zeros((self.config['epi_sz'],))
                delta_wt = np.zeros((self.config['epi_sz'],))
                immed_rewards = np.zeros((self.config['epi_sz'],))
                accum_rewards = np.zeros((self.config['epi_sz'],))
                mean_rev_accum_rewards = np.zeros((self.config['epi_sz'],))


                # action_history = np.empty((self.config['epi_sz'],))
                obs_history = np.zeros((self.config['epi_sz'],) + (self.config['obs_sz'],))

                # We start with 0% weight in the target currency.
                
                t, state, price_tuple, is_finished = self.env.reset(epi_idx)

                #  - rec_wt[t]: the recommended investment weight by nn
                #  - delta_wt[t]: a difference in weights between the recommended investment weight by nn
                #                 and current weight in the target currency.
                #  - rewards[t]: rewards from the latest recommended trade (caued by the recommended weight)
                # features = state[:-2]
                rec_wt[t] = state[-2]
                self.inv_wt[t] = rec_wt[t]
                delta_wt[t] = state[-1]
                # immed_rewards[t] = 0.
                # accum_rewards[t] = 0.
                
                cur_bid, cur_ask, _, _ = price_tuple

                # We have observed and not taken any action yet.
                obs = np.append(t, state)
                obs_history[t] = obs

                # action_history is a list of recommneded weights
                # We start with zero weight
                action_history = [0.]

                # buysell_history is a list of 'B'(buy), 'S'(sell), 'N' (No trade)
                buysell_history = ['N']

                ### start of batch ###
                # We loop through time stamps within an episode.
                while not is_finished:
                    # The neural networks take observation history and action history as inputs
                    # and they return an action-like scalar [-1,1].
                    # This scalar is a recommended investment weight (rec_wt) in the target currency.
                    # `state` is a numpy array (features, rec_wt, delta_wt)
                    # The nueral networks are called during self.act() being executed
                    action = self.act(obs_history, action_history)
                    action_history.append(action)

                    t, state, price_tuple, is_finished = self.env.step(action, self.inv_wt[t])
                    # features = state[:-2]
                    rec_wt[t] = state[-2]
                    delta_wt[t] = state[-1]
                    obs = np.append(t, state)
                    obs_history[t] = obs

                    # Since we ran 'env.stop()', step counter increased by 1.
                    # Therefore, price_tuple[0] is interpreted as next_bid
                    next_bid, next_ask, _, _ = price_tuple

                    ### Trading begins
                    # buy/sell decision
                    # buysell = 'B' if delta_wt[t] >=0 else 'S'
                    # if delta_wt[t] > 0:
                    #     buysell = 'B'
                    # elif delta_wt[t] == 0:
                    #     buysell = 'N'
                    # else:
                    #     buysell = 'S'
                    if action > 0:
                        buysell = 'B'
                    elif action == 0:
                        buysell = 'N'
                    else:
                        buysell = 'S'
                    buysell_history.append(buysell)

                    # change an investment weight
                    self.inv_wt[t] = self.inv_wt[t-1] + delta_wt[t]

                    # profit/loss calculation
                    # pass

                    ### Get an immedidate reward
                    immed_rewards[t] = self.get_immed_reward(buysell, cur_bid[t-1], cur_ask[t-1], next_bid[0], next_ask[0])

                    ### Accumulate rewards within this episode.
                    accum_rewards[t] = accum_rewards[t-1] + immed_rewards[t]


                    ####### mean reversion trading #########
                    if t >= mean_rev_lookback:
                        mean = np.average(cur_bid[t-mean_rev_lookback:t])
                        std = np.std(cur_bid[t-mean_rev_lookback:t])
                        z_score = (cur_bid[t-1] - mean)/std
                        if z_score < -2:
                            mean_rev_bs = 'B'
                        elif z_score > 2:
                            mean_rev_bs = 'S'
                    else:
                        mean_rev_bs = 'N'

                    mean_rev_immed_rewards = self.get_immed_reward(mean_rev_bs, cur_bid[t-1], cur_ask[t-1], next_bid[0], next_ask[0])
                    mean_rev_accum_rewards[t] = mean_rev_accum_rewards[t-1] + mean_rev_immed_rewards
                    ########################################
                    if t % 100 == 0:
                        if self.verbose:
                            print('Time {}. Action(rec_wt) is {}. Reward is {}. Bid price goes from {} to {}'.format(t, action, immed_rewards[t], cur_bid[t-1], next_bid[0]))

                ### end of all batches ###
                # losses = self.model.train_on_batch(obs_history, immed_rewards)

                print('Epoch:{} Episode:{}. The DRL total reward is {} vs A baseline total reward is {}. DRL - Baseline = {}'.format(
                    epoch+1, epi_idx+1, accum_rewards[t], mean_rev_accum_rewards[t], accum_rewards[t] - mean_rev_accum_rewards[t]))
            
            ### enf of one epoch ##
            epochs_total_rewards.append(accum_rewards[-1])
        
        return accum_rewards, mean_rev_accum_rewards