import pandas as pd
import numpy as np 

def log_returns(prices):
    '''
    output: 
        returns[t] = log(p_t / p_{t-1})
    '''
    returns = np.zeros(prices.shape[0])
    returns = (np.log(prices) - np.log(np.roll(prices, 1)))[1:]
    return returns

def get_features(X, cur, n, epi_sz, bins, ts, ccy):
    '''
    epi_sz: length of an episode
    snapshot_size: number of past returns to calculate frequencies
    n: get the features for the n-th episode
    
Return:
    target_close: close prices of target currency
    features: a 3d tensor of shape (number of currencies, snapshot_size+2, epi_sz - (snapshot_size+1))
    '''
    snapshot_size = bins.size - 3
    
    start_idx = n * epi_sz
    end_idx = min((n+1) * epi_sz, ts.size) - 1
    data = X[(X.timestamp>=ts[start_idx]) & (X.timestamp<=ts[end_idx])]

    features = np.zeros((len(ccy), snapshot_size+2, epi_sz - (snapshot_size+1)))
    for i in range(len(ccy)):
        data_currency = data[epi_sz*i:epi_sz*(i+1)]['close']
        if cur == ccy[i]:
            target_close = data_currency.values
        for channel in range(epi_sz - (snapshot_size+1)):
            features[i, :, channel] = np.histogram(log_returns(data_currency.iloc[channel:channel + snapshot_size+1]), bins=bins)[0]

    return target_close, features



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

        self.state = None           # A tuple (observation, cur_wt). observation == a row of `feature_span`
        self.price_tuple = None    # A tuple (bid, ask, next_bid, next_ask)

    def get_features_within_epi(self, step_counter):
        '''
        Parameters:
            step_counter(int): the current step index within an episode
        '''
        return self.close[step_counter:], self.next_close[step_counter:], \
                self.feature_span[:, :, self.step_counter:self.step_counter+self.config['channels']]

    def reset(self, epi_idx):
        '''
        Parameters:
            epi_idx(int): an episode index
        
        Returns:
            A tuple: (step_counter, state, price_record, if_finished)
        '''
        self.epi_idx = epi_idx

        close_prices, self.feature_span = get_features(X=self.X, cur=self.config['target_currency'], 
                                              n=self.epi_idx, epi_sz=self.config['epi_sz'], 
                                              bins = self.config['bins'], ts=self.ts, ccy=self.ccy)
        
        # e.g: close_prices <- [2,3,4,5]
        #      self.close <- [2,3,4]
        #      self.next_close <- [3,4,5]
        # Then we can have 'close price' and 'next close price' with the same index
        # , which is eaiser.
        self.close = close_prices[:-1]
        self.next_close = close_prices[1:]

        self.step_counter = 0

        # feature_span is a matrix of feature vectors (ccy, snapshot_size, epi_sz)
        features = self.feature_span[self.step_counter:self.step_counter+self.config['channels']]
        rec_wt = 0.0
        delta_wt = 0.0

        self.state = (features, [rec_wt, delta_wt])

        # All elements of this tuple has a shape of (epi_sz, )
        self.price_tuple = (self.close, self.next_close)

        return self.step_counter, self.state, self.price_tuple, False
    
    def step(self, action, inv_wt):
        '''
        Parameters:
            action(float): a recommended investment weight in the target currency.
        
        Returns:
            A tuple: (step_counter, state, price_record, is_finished)
        '''
        self.step_counter += 1

        # `is_finished` is true if it arrives at the end of the current episode.
        is_finished = (self.step_counter+1 == self.config['epi_sz'])
        
        rec_wt = action     # We take `action` as a recommended investment weight in the target currency.

        # delta_wt will be used when an agent trades. 
        delta_wt = rec_wt - inv_wt

        close, next_close, features = self.get_features_within_epi(self.step_counter)
#         print(features.shape)

        self.state = (features, [rec_wt, delta_wt])
        self.price_tuple = (close, next_close)

        # We return t(=indix within an episode), state, price tuple, finish flag
        return self.step_counter, self.state, self.price_tuple, is_finished
