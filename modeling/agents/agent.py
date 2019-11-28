import numpy as np

from collections import deque

import tensorflow as tf
import tensorflow.keras.optimizers as kop
import tensorflow.keras.losses as kls

from utils.portfolio import PortfolioManagement

class Memory:
    def __init__(self, maxlen):
        self.buffer = deque(maxlen=maxlen)
        self.keys = ('state', 'action', 'reward', 'next_state', 'is_finished')
    
    def add(self, experience):
        '''
        Appends a tuple of `experience` to the right side of self.buffer (deque)
        '''
        expr_dict = dict(zip(self.keys, experience))
        self.buffer.append(expr_dict)
    
    def sample(self, batch_size):
        '''
        Returns a list of random tuples of sample experiences from this memory.
        '''
        idx = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)

        return [self.buffer[i] for i in idx]
    

class Agent:
    def __init__(self, env, config, mode='train', init_wealth = 10000, verbose=False,
                 explore_start = 1.0, explore_stop = 0.01, decay_rate = 0.00001,
                 gamma = 0.9, memory_size=10**6,
                 test_model_path = None, max_steps=10000):
        self.config = config
        self.env = env
        self.max_steps = max_steps
        self.explore_start = explore_start
        self.explore_stop = explore_stop
        self.decay_rate = decay_rate
        self.gamma = gamma
        self.memory_size = memory_size
        self.memory = Memory(maxlen = self.memory_size)

        if mode == 'train':
            self.mode = 'train'
            self.lr_rate = config['lr_rate']
            self.buffer_sz = config['buffer_sz']
            self.epsilon = 0.97
            self.discount = 0.99
            self.batch_sz = config['batch_sz']
            # self.num_batches = config['num_batches']
            # self.start_learning = config['start_learning']

        self.wealth = init_wealth
        self.state = None
        self.verbose = verbose

class DQNAgent(Agent):
    def __init__(self, model, env, config, init_wealth = 10000, mode='train', verbose=False, test_model_path = None):
        Agent.__init__(self, env, config, mode='train')
        self.model = model
        self.model.compile(
                optimizer=kop.Adam(learning_rate=self.lr_rate),
                # deifne a loss function
                # loss = 'mean_absolute_error'
                loss=[self._compute_loss],
                # metrics=['accuracy']
            )

    def predict_action(self, state, cur_wt, decay_step, explore_start = 1.0, explore_stop = 0.01, decay_rate = 0.00001):
        # We explore if `explore_prob` > `trade_off_prob`
        trade_off_prob = np.random.rand()
        explore_prob = self.explore_stop + (self.explore_start - self.explore_stop)*np.exp(-self.decay_rate*decay_step)

        if explore_prob > trade_off_prob:
            # We take a random action (exploration)
            # i.e. we return random investment weights
            action = np.random.rand(len(cur_wt))
            action = action / np.sum(action)
        else:
            action = self.model(state)

        return action, explore_prob
    
    def train(self):
        # Average rewards per epoch
        epochs_rewards = []

        # Training loop
        for epoch in range(self.config['epochs']):
            # We use `decay_step` and `decay_rate` when we do a variant of the e-greedy strategy.
            decay_step = 0
            epoch_rewards = []
            for epi_idx in range(self.config['total_no_epi']):
                step = 0
                episode_rewards = []
                accum_rewards = []
                accum_losses = []
                action_history = []
                obs_history = []
                

                
                # Reset the environment whenever a new episode starts
                t, state, price_tuple, is_finished = self.env.reset(epi_idx)

                # We have observed and not taken any action yet.
                obs = np.append(t, state[0])
                obs_history.append(obs)

                episode_rewards.append(0.)
                accum_rewards.append(0.)

                # `price` is FX rates for ONE currency pair (config['target_currency'])
                price, next_price = price_tuple
                
                # cash price is always 1. So the initial price vector is [1, price[0]].
                price_vector = np.array([1, price[0]])

                # M = 1 because our FX to trade is just one currency pair(the target currrency) at this moment
                pf = PortfolioManagement(init_price=price_vector, M=1)

                ### start of training ###
                while step < self.max_steps:
                    step += 1

                    # Predict an action to take, take it and add it into a history list.
                    # The nueral networks are called during self.predict_action() being executed
                    action, _ = self.predict_action(state=state, cur_wt=pf.wt[-1], decay_step=decay_step)
                    action_history.append(action)

                    # We observe next prices and we try to trade at those prices.
                    # However, executed prices will be different due to noise and transaction costs.
                    # In this trade() method, we do:
                    #   append self.wt, self.qty, self.price, self.rel_price into the corresponding lists.

                    ## Check if next_price[0] is really the next price or next_price[1] is.
                    pf.trade(cur_wt=pf.wt[-1], target_wt=action, target_price=next_price[0])

                    t, next_state, price_tuple, is_finished = self.env.step(action=action, inv_wt = pf.wt[-1])
                    
                    reward = pf.get_pf_value(-1) - pf.get_pf_value(-2)   # getting a reward function will be implemented within self.env.step()

                    # This immediate reward will be summed up as 'total reweards' later
                    episode_rewards.append(reward)
                    accum_rewards.append(accum_rewards[-1] + reward)

                    obs = np.append(t, next_state[0])
                    obs_history.append(obs)

                    # finishing coditions:
                    #   ??? 
                    if is_finished:
                        # next_state = np.zeros_like(next_state)   # A dummy state since this episode ends.

                        step = self.max_steps

                        total_reward = np.sum(episode_rewards)
                        epoch_rewards.append((epi_idx, total_reward))   # (episode number, total reward in that episode)

                        # Add experience into a memory buffer. This expeirence is about how an episode ends.
                        self.memory.add((state, action, reward, next_state, is_finished))
                    else:
                        # Add experience into a memory buffer.
                        self.memory.add((state, action, reward, next_state, is_finished))

                        state = next_state

                    
                    # We get tuples of (randomly chosen) experiences from memory to run a mini batch for a learning process. 

                    ####### Reply buffer begins... 
                    # ### We should fill(=initialize) this 'memory' at least randomly...
                    # batch_list = self.memory.sample(self.batch_sz)

                    # # Get a dictionary of the first experience
                    # expr_dict = batch_list[0]

                    # # Extract the first experience
                    # state = expr_dict['state']
                    # action = expr_dict['action']
                    # reward = expr_dict['reward']
                    # next_state = expr_dict['next_state']
                    # is_finished = expr_dict['is_finished']

                    # target_Q_batch = []

                    # # Get an action for the next state
                    # action = self.model(next_state)

                    # for expr_dict in batch_list:
                    #     if is_finished:
                    #         target_Q_batch.append(reward)
                    #     state = expr_dict['state']
                    #     action = expr_dict['action']
                    #     reward = expr_dict['reward']
                    #     next_state = expr_dict['next_state']
                    #     is_finished = expr_dict['is_finished']
                    ####### Reply buffer ends...
                        

                    if t % 100 == 0:
                        if self.verbose:
                            print('Time {}. Action(rec_wt) is {}. Reward is {}. Bid price goes from {} to {}'.format(t, action, episode_rewards[t], cur_bid[t-1], next_bid[0]))

                ### end of all batches ###
                losses = self.model.train_on_batch(np.array(obs_history), np.array(episode_rewards))
                accum_losses.append(losses)

                print('Epoch:{} Episode:{}. The training loss is {}. The total losses are {}'.format(epoch+1, epi_idx+1, losses, np.sum(accum_losses)))
            
            ### enf of one epoch ##
            epochs_rewards.append(accum_rewards[-1])
        
        return accum_rewards

    def _compute_loss(self, obs_history, immed_rewards):
        return -tf.reduce_sum(immed_rewards)


class SimpleDRLAgent(Agent):
    def __init__(self, model, env, config, init_wealth = 10000, mode='train', verbose=False, test_model_path = None):
        super(Agent, self).__init__(self, env, config, init_wealth = 10000, mode='train', verbose=False, test_model_path = None)

        self.model = model
        self.model.compile(
                optimizer=kop.Adam(learning_rate=self.lr_rate),
                # deifne a loss function
                # loss = 'mean_absolute_error'
                loss=[self._compute_loss],
                # metrics=['accuracy']
            )

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
        accum_rewards = np.zeros((self.config['total_no_epi'],))

        # action_history = np.empty((self.config['epi_sz'],))
        obs_history = np.zeros((self.config['epi_sz'],) + (self.config['obs_sz'],))
        
        # Accumucated rewards per episode
        # epi_rewards = []

        # Average rewards per epoch
        epochs_total_rewards = []

        
        # mean reversion agent
        mean_rev_lookback = 20
        accum_losses = np.zeros((self.config['total_no_epi'],))

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
                accum_losses[epi_idx] = losses

                print('Epoch:{} Episode:{}. The training loss is {}. The DRL total reward is {} vs A baseline total reward is {}. DRL - Baseline = {}'.format(epoch+1, epi_idx+1, losses, accum_rewards[t], mean_rev_accum_rewards[t], accum_rewards[t] - mean_rev_accum_rewards[t]))
            
            ### enf of one epoch ##
            epochs_total_rewards.append(accum_rewards[-1])
        
        return accum_rewards, mean_rev_accum_rewards, accum_losses


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