import numpy as np 

from features import draw_episode

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
        self.inv_wt = None          # The current investment weight in the target currency.
        # self.rec_wt = None
        # self.delta_wt = None
        self.state = None           # A tuple (observation, cur_wt). observation == a row of `feature_span`
        self.price_tuple = None    # A tuple (bid, ask, next_bid, next_ask)


    def get_features_within_epi(self, step_counter):
        '''
        Parameters:
            step_counter(int): the current step index within an episode
        '''
        return self.bid[step_counter:], self.ask[step_counter:], \
                self.next_bid[step_counter:], self.next_ask[step_counter:], \
                self.feature_span[step_counter]


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

        # len(bid_prices) == epi_sz + 1
        bid_prices, ask_prices, self.feature_span = \
            draw_episode(X=self.X, cur=self.config['target_currency'],
                        n=self.epi_idx, split_sz=self.config['split_sz'],
                        lag=self.config['lag'], ts=self.ts, ccy=self.ccy)
        
        # e.g: b = np.array([1,2,3,4,5])
        #      lag = 2
        #      bid_prices <- [2,3,4,5]
        #      self.bid <- [2,3,4]
        #      self.next_bid <- [3,4,5]
        # Then we can have 'bid price' and 'next bid price' with the same index
        # , which is eaiser.
        self.bid = bid_prices[:-1]
        self.ask = ask_prices[:-1]
        self.next_bid = bid_prices[1:]
        self.next_ask = ask_prices[1:]

        self.step_counter = 0

        # feature_span is a matrix of feature vectors (epi_sz, features size)
        # features is one feature vector (epi_sz, )
        features = self.feature_span[self.step_counter]  
        rec_wt = 0.0
        delta_wt = 0.0

        # state = features + [rec_wt, delta_wt]
        self.state = np.append(features, [rec_wt, delta_wt])

        # All elements of this tuple has a shape of (epi_sz, )
        self.price_tuple = (self.bid, self.ask, self.next_bid, self.next_ask)

        return self.step_counter, self.state, self.price_tuple, False
    
    def step(self, action, inv_wt):
        '''
        Parameters:
            action(float): a recommended investment weight in the target currency.
        
        Returns:
            A tuple: (step_counter, state, price_record, is_finished
        '''
        self.step_counter += 1

        # `is_finished` is true if it arrives at the end of the current episode.
        is_finished = (self.step_counter+1 == self.config['epi_sz'])
        
        rec_wt = action     # We take `action` as a recommended investment weight in the target currency.

        # delta_wt will be used when an agent trades. 
        delta_wt = rec_wt - inv_wt

        bid, ask, next_bid, next_ask, features = self.get_features_within_epi(self.step_counter)

        # state = features + [rec_wt, delta_wt]
        self.state = np.append(features, [rec_wt, delta_wt])
        self.price_tuple = (bid, ask, next_bid, next_ask)

        # We return t(=indix within an episode), state, price tuple, finish flag
        return self.step_counter, self.state, self.price_tuple, is_finished
