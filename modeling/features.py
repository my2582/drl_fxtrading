import numpy as np
import pandas as pd

def generate_episode(X, n, cur, ep_size):    
    '''
    Input:
        X: data (bid/ask for each currency pair)
        n: get the n-th episode
        cur: target currency
        ep_size, size of episode
    '''
    
    ts = np.sort(X['timestamp'].unique()) # all timestamps
    ccy = X['ccy'].unique() # all currency pairs
    
    start_idx = n * ep_size
    end_idx = min((n+1) * ep_size, ts.size) - 1
    data = X[(X.timestamp>=ts[start_idx]) & (X.timestamp<=ts[end_idx])]
     
    i = 0
    other_bid = np.zeros((end_idx - start_idx + 1, ccy.shape[0]-1))
    other_ask = np.zeros((end_idx - start_idx + 1, ccy.shape[0]-1))
    for elem in ccy:
        tmp = data[data.ccy == elem]
        if elem == cur:
            target_bid = tmp['bid_price'].values
            target_ask = tmp['ask_price'].values
        else:
            other_bid[:,i] = tmp['bid_price'].values
            other_ask[:,i] = tmp['ask_price'].values
            i += 1
    return target_bid, target_ask, other_bid, other_ask

def log_returns(prices, lag):
    '''
    output: 
        returns[t][i] = log(p_t / p_{t-i}), 1 <= i <= lag
    '''
    returns = np.zeros((prices.shape[0]-lag, lag))
    for i in range(lag):
        returns[:,i] = (np.log(prices) - np.log(np.roll(prices, i+1)))[lag:]
    return returns

def get_features(target_bid, target_ask, other_bid, other_ask, lag):
    '''
    Output:
        features: log returns of ask/bid for all currency pairs; shape (T-lag, 2*lag*cur_pairs); already normalized
    '''
    features = log_returns(target_bid, lag)
    features = np.append(features, log_returns(target_ask, lag), axis = 1)
    for i in range(other_bid.shape[1]):
        features = np.append(features, log_returns(other_bid[:,i], lag), axis = 1)
    for i in range(other_bid.shape[1]):
        features = np.append(features, log_returns(other_ask[:,i], lag), axis = 1)
        
    normalized_fs = (features - features.mean()) / features.std()
    return normalized_fs

def draw_episode(X, cur, n, ep_size, lag):
    '''
    Input:
        X: data (bid/ask for each currency pair)
        cur, currency pair that we target to trade
        n, draw the n-th episode
        ep_size, size of episode
        lag, number of lag log-returns z_1,...z_m
    Output:
        target_bid: target currency's bid prices
        target_ask: target currency's ask prices
        features: features for feeding into the neural network
        
    Note: compared to Dai's code, I am not using min_history (min length of a valid episode) as argument; 
        Also, I am not randomly selecting episodes
    '''
    target_bid, target_ask, other_bid, other_ask = generate_episode(X, n, cur, ep_size)
    features = get_features(target_bid, target_ask, other_bid, other_ask, lag)
    return target_bid, target_ask, features

