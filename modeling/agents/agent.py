import numpy as np

from collections import deque

import tensorflow as tf
import tensorflow.keras.optimizers as kop
import tensorflow.keras.losses as kls

from agents.model import SimpleModel, DiscreteTradingModel

from utils.portfolio import PortfolioManagement

class Memory:
    def __init__(self, init_size, maxlen, env, action_set):
        self.init_size = init_size
        self.buffer = deque(maxlen=maxlen)
        self.keys = ('state', 'action', 'reward', 'next_state', 'is_finished', 'max_Q')
        self.env = env
        self.action_set = action_set
        self._randomly_fill()

    def _randomly_fill(self):
        '''
        Randomly generates experience (state, action, reward, next_state, is_finished)
          - experience size is set to be config['batch_sz']
        '''


        for i in range(self.env.config['batch_sz']):
            if i == 0:
                rand_epi_idx = np.random.randint(self.env.config['total_no_epi'])
                t, obs, price_tuple, is_finished = self.env.reset(rand_epi_idx)
                state = np.append(t, obs)
                price, next_price = price_tuple
                price_vector = np.array([1, price[0]])
                temp_pf = PortfolioManagement(init_price=price_vector, M=1)

            # We take a random action (exploration)
            random_Q = np.random.uniform(size=len(self.env.config['action_set']))
            max_Q = np.max(random_Q)
            choice = np.argmax(random_Q)
            action = self.action_set[choice]

            target_wt = temp_pf.get_target_weight(action)

            _, next_price = price_tuple

            # Do this trade
            temp_pf.trade(cur_wt=temp_pf.wt[-1], target_wt=target_wt, target_price=next_price[0])
            reward = temp_pf.get_pf_value(-1) - temp_pf.get_pf_value(-2)   # getting a reward function will be implemented within self.env.step() later.
            
            t, next_obs, price_tuple, is_finished = self.env.step(action=action)
            next_state = np.append(t, next_obs)
            self.add((state, action, reward, next_state, is_finished, max_Q))

            # move to the next state
            if is_finished:
                rand_epi_idx = np.random.randint(self.env.config['total_no_epi'])
                t, obs, price_tuple, is_finished = self.env.reset(rand_epi_idx)
                state = np.append(t, obs)
            else:
                state = next_state
    
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
                 gamma = 0.9, test_model_path = None, max_steps=10000):
        self.config = config
        self.env = env
        self.max_steps = max_steps
        self.explore_start = explore_start
        self.explore_stop = explore_stop
        self.decay_rate = decay_rate
        self.gamma = gamma

        self.lr_rate = config['lr_rate']
        self.buffer_sz = config['buffer_sz']
        self.epsilon = 0.97
        self.discount = 0.99
        self.batch_sz = config['batch_sz']
        self.num_batches = config['num_batches']
        self.state_sz = config['state_sz']
        self.epochs = config['epochs'] = 10

        self.wealth = init_wealth
        self.state = None
        self.verbose = verbose

class DQNAgent(Agent):
    def __init__(self, env, config, init_wealth = 10000, mode='train', print_step_frq = 10, target_net_update_frq = 1, verbose=False, memory_size=10**5, test_model_path = None):
        Agent.__init__(self, env, config, mode='train')
        self.target_net_update_frq = target_net_update_frq    # The frequency of setting the target net's parameters to be Q net's
        self.print_step_frq = print_step_frq
        self.action_set = config['action_set']
        self.memory_size = memory_size
        self.memory = Memory(init_size = config['batch_sz'], maxlen = memory_size, env=env, action_set=self.action_set)
        self.q_net = DiscreteTradingModel(action_set=self.action_set)
        self.q_net.compile(
                optimizer=kop.Adam(learning_rate=self.lr_rate),
                # deifne a loss function
                loss = 'mean_absolute_error'
                # loss=self._compute_loss(),
                # metrics=['accuracy']
            )
        self.target_net = DiscreteTradingModel(action_set=self.action_set)

    def _compute_loss(self):
        def loss_fn(y_true, y_pred):
             return tf.keras.losses.MSE(y_true, y_pred)
        
        return loss_fn


    def get_random_action(self):
        random_Q = np.random.uniform(size=len(self.action_set))
        max_Q = np.max(random_Q)
        choice = np.argmax(random_Q)
        action = self.action_set[choice]

        return action, max_Q

    def predict_action(self, state, decay_step, explore_start = 1.0, explore_stop = 0.01, decay_rate = 0.00001):
        # We explore if `explore_prob` > `trade_off_prob`
        trade_off_prob = np.random.rand()
        explore_prob = self.explore_stop + (self.explore_start - self.explore_stop)*np.exp(-self.decay_rate*decay_step)

        if explore_prob > trade_off_prob:
            # We take a random action (exploration)
            action, max_Q = self.get_random_action()

        else:
            # Get an estimate of Q values for each possible action
            Qs = self.q_net(state)

            # Get an argument that maximizes the Q values.
            choice = np.argmax(Qs)
            max_Q = np.max(Qs)

            # Take the best action 
            action = self.q_net.action_set[choice]

        return action, max_Q, explore_prob
    
    def learn_from_replay(self):
        # We get tuples of (randomly chosen) experiences from memory to run a mini batch for a learning process. 
        ####### We apply TD steps to update Q network by running batches
        states_batch = np.empty(shape=(0, self.state_sz))
        targets_Qs_batch = np.empty(shape=(0))
        total_losses = []

        for batch_no in range(self.num_batches):
            # Obtain a random mini-batch list from our memory buffer.
            # batch_list = [experience dictionary #1, #2, ..., #batch_sz]
            batch_list = self.memory.sample(self.batch_sz)

            # Get a list of states, a list of actions, ... from all experiences in this mini-batch("mb")
            states_mb = np.array([expr_dict['state'] for expr_dict in batch_list])
            actions_mb = np.array([expr_dict['action'] for expr_dict in batch_list])
            rewards_mb = np.array([expr_dict['reward'] for expr_dict in batch_list])
            next_states_mb = np.array([expr_dict['next_state'] for expr_dict in batch_list])
            is_finished_mb = np.array([expr_dict['is_finished'] for expr_dict in batch_list])
            max_Q_mb = np.array([expr_dict['max_Q'] for expr_dict in batch_list])

            target_Qs_mb = []

            # Get a batch_sz-length list of target Q values in this batch.
            Qs_at_next_state = [self.target_net(next_state) for next_state in next_states_mb]
            for i in range(self.batch_sz):
                idx = batch_no*self.batch_sz+i
                is_finished = is_finished_mb[i]
                if is_finished:
                    target_Qs_mb.append(rewards_mb[i])
                else:
                    target = rewards_mb[i] + self.gamma*np.max(Qs_at_next_state[i])
                    target_Qs_mb.append(target)

            states_batch = np.vstack((states_batch, states_mb))
            target_Qs_mb = np.array(target_Qs_mb)
            targets_Qs_batch = np.append(targets_Qs_batch, target_Qs_mb)

            # Update Q networks using Target Qs from this mini-batch.
            losses = self.q_net.fit(states_mb, target_Qs_mb, verbose=1)
            total_losses.append(losses)
        
        return total_losses

    def learn_from_replay_batch(self):
        # We get tuples of (randomly chosen) experiences from memory to run a mini batch for a learning process. 
        ####### We apply TD steps to update Q network by running batches
        states_batch = np.empty(shape=(0, self.state_sz))
        targets_Qs_batch = np.empty(shape=(0))
        total_losses = 0

        for batch_no in range(self.num_batches):
            # Obtain a random mini-batch list from our memory buffer.
            # batch_list = [experience dictionary #1, #2, ..., #batch_sz]
            batch_list = self.memory.sample(self.batch_sz)

            # Get a list of states, a list of actions, ... from all experiences in this mini-batch("mb")
            states_mb = np.array([expr_dict['state'] for expr_dict in batch_list])
            actions_mb = np.array([expr_dict['action'] for expr_dict in batch_list])
            rewards_mb = np.array([expr_dict['reward'] for expr_dict in batch_list])
            next_states_mb = np.array([expr_dict['next_state'] for expr_dict in batch_list])
            is_finished_mb = np.array([expr_dict['is_finished'] for expr_dict in batch_list])
            max_Q_mb = np.array([expr_dict['max_Q'] for expr_dict in batch_list])

            target_Qs_mb = []

            # Get a batch_sz-length list of target Q values in this batch.
            Qs_at_next_state = [self.target_net(next_state) for next_state in next_states_mb]
            for i in range(self.batch_sz):
                idx = batch_no*self.batch_sz+i
                is_finished = is_finished_mb[i]
                if is_finished:
                    target_Qs_mb.append(rewards_mb[i])
                else:
                    target = rewards_mb[i] + self.gamma*np.max(Qs_at_next_state[i])
                    target_Qs_mb.append(target)

            states_batch = np.vstack((states_batch, states_mb))
            target_Qs_mb = np.array(target_Qs_mb)
            targets_Qs_batch = np.append(targets_Qs_batch, target_Qs_mb)

        # Update Q networks using Target Qs from this mini-batch.
        losses = self.q_net.fit(states_batch, targets_Qs_batch, epochs=self.epochs, verbose=0)
        
        # return a mean of losses in this whole batch.
        return tf.reduce_mean(losses.history['loss'])

    def update_target_net(self, epi_idx):
        if self.q_net.get_weights()[0].shape == self.target_net.get_weights()[0].shape:
            # We only updates target net's weights when 'Q networks' was chosen to use
            # during the 'e-greedy selection' in predict_action().
            self.target_net.set_weights(self.q_net.get_weights())
            print('The target network is updated by cloning the Q nets parameters at Episode {:2d}'.format(epi_idx+1))
        
    def train(self):
        # We use `decay_step` and `decay_rate` when we do a variant of the e-greedy strategy.
        decay_step = 0
        for epi_idx, epi_no in enumerate(np.random.permutation(self.config['total_no_epi'])):
            episode_rewards = []
            accum_rewards = []
            accum_losses = np.empty(shape=(0))
            action_history = []
            state_history = []
            
            # Reset the environment whenever a new episode starts
            t, obs, price_tuple, is_finished = self.env.reset(epi_no)  # epi_no a random index

            # We have returned step counter t, state(=feature) from the environment.
            # No action is taken yet.
            state = np.append(t, obs)
            state_history.append(state)
            action = 0  # initialize an action to be no tprice_vectorrade

            episode_rewards.append(0.)
            accum_rewards.append(0.)

            # `price` is FX rates for ONE currency pair (config['target_currency'])
            price, next_price = price_tuple
            
            # cash price is always 1. So the initial price vector is [1, price[0]].
            price_vector = np.array([1, price[0]])

            # M = 1 because our FX to trade is just one currency pair(the target currrency) at this moment
            pf = PortfolioManagement(init_price=price_vector, M=1)

            ### We start trading in one episode ###
            while not is_finished:
                decay_step += 1

                # Predict an action to take, take it and add it into a history list.
                # The nueral networks are called during self.predict_action() being executed
                # action is a scalar: -1 is short, 0 is neutral (no trade), 1 is long.
                # max_Q is the maximum of Q values, which was chosen as this 'action'
                action, max_Q, _ = self.predict_action(state=state, decay_step=decay_step)
                # action_history.append(action)

                # Get a target weight for trading
                target_wt = pf.get_target_weight(action)

                # Do this trade
                # In this pf.trade() method, we do:
                #   append self.wt, self.qty, self.price, self.rel_price into the corresponding lists.
                pf.trade(cur_wt=pf.wt[-1], target_wt=target_wt, target_price=next_price[0])
                
                reward = pf.get_pf_value(-1) - pf.get_pf_value(-2)   # getting a reward function will be implemented within self.env.step() later.
                # print('Action:{}, Price:{}, Next price:{}, Portfolio value {}, Reward {}'.format(action,  price[0], next_price[0], pf.get_pf_value(), reward))

                t, next_obs, price_tuple, is_finished = self.env.step(action=action)
                price, next_price = price_tuple

                # This immediate reward will be summed up as 'total reweards' later
                episode_rewards.append(reward)
                accum_rewards.append(accum_rewards[-1] + reward)

                next_state = np.append(t, next_obs)
                # state_history.append(next_state)

                # Add experience into a memory buffer. This expeirence is about how an episode ends.
                self.memory.add((state, action, reward, next_state, is_finished, max_Q))

                mean_loss = self.learn_from_replay_batch()
                print('Episode/Time step {:2d}/{:2d}. A mean loss of {:.4f} from replay-learning'.format(epi_idx+1, t, mean_loss))

            ### end of trading ###

            print('Episode {:2d}. Total rewards {:.2f}. Portfolio value {:5.2f}'.format(epi_idx+1, tf.reduce_sum(episode_rewards), pf.get_pf_value()))

            # if step % self.print_step_frq == 0:
            #     print('Epoch:{} Episode:{} Step:{}. The training loss is {}. The average loss is {}. Total rewards: {}'.format(epoch+1, epi_idx+1, step, losses, np.average(accum_losses), accum_rewards[-1]))

            if epi_idx % self.target_net_update_frq == 0:    # epi_idx is a sequencial order (0, 1, 2, .. up to total_no_epi)
                self.update_target_net(epi_idx)
        
        return accum_rewards


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

    def _compute_loss(self, state_history, immed_rewards):
        return -tf.reduce_sum(immed_rewards)


    def _epsilon_greedy_action(self, action, epsilon):
        if tf.random.uniform(shape=(1,))[0] > epsilon:
            # Take a random action
            return np.random.randint(-100,100)/100
        
        return action
    
    def act(self, state_history, action_history):
        # state_history[-1,] == the latest state (timestamp, features, rec_wt, delta_wt)
        state = state_history[-1,:]

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
        state_history = np.zeros((self.config['epi_sz'],) + (self.config['obs_sz'],))
        
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
                state_history = np.zeros((self.config['epi_sz'],) + (self.config['obs_sz'],))

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
                state_history[t] = obs

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
                    action = self.act(state_history, action_history)
                    action_history.append(action)

                    t, state, price_tuple, is_finished = self.env.step(action, self.inv_wt[t])
                    # features = state[:-2]
                    rec_wt[t] = state[-2]
                    delta_wt[t] = state[-1]
                    obs = np.append(t, state)
                    state_history[t] = obs

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
                losses = self.model.train_on_batch(state_history, immed_rewards)
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
        state_history = np.zeros((self.config['epi_sz'],) + (self.config['obs_sz'],))
        
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
                state_history = np.zeros((self.config['epi_sz'],) + (self.config['obs_sz'],))

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
                state_history[t] = obs

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
                    action = self.act(state_history, action_history)
                    action_history.append(action)

                    t, state, price_tuple, is_finished = self.env.step(action, self.inv_wt[t])
                    # features = state[:-2]
                    rec_wt[t] = state[-2]
                    delta_wt[t] = state[-1]
                    obs = np.append(t, state)
                    state_history[t] = obs

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
                # losses = self.model.train_on_batch(state_history, immed_rewards)

                print('Epoch:{} Episode:{}. The DRL total reward is {} vs A baseline total reward is {}. DRL - Baseline = {}'.format(
                    epoch+1, epi_idx+1, accum_rewards[t], mean_rev_accum_rewards[t], accum_rewards[t] - mean_rev_accum_rewards[t]))
            
            ### enf of one epoch ##
            epochs_total_rewards.append(accum_rewards[-1])
        
        return accum_rewards, mean_rev_accum_rewards