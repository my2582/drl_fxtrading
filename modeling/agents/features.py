import numpy as np
import pandas as pd

def generate_episode(X, n, cur, split_sz, ts, ccy):    
    '''
    Input:
        X: data (bid/ask for each currency pair)
        n: get the n-th episode
        cur: target currency
        split_sz: size of one split. len(split) == len(batch) + len(lag).
                  The number of episodes is int()
        ts: a numpy array of unique time stamps
        ccy: a Series of unique currency pairs
    '''

    start_idx = n * split_sz
    end_idx = min((n+1) * split_sz, ts.size) - 1
    data = X[(X.timestamp>=ts[start_idx]) & (X.timestamp<=ts[end_idx])]
     
    i = 0
    other_close = np.zeros((end_idx - start_idx + 1, ccy.shape[0]-1))
    other_ask = np.zeros((end_idx - start_idx + 1, ccy.shape[0]-1))
    for elem in ccy:
        tmp = data[data.ccy == elem]
        if elem == cur:
            target_close = tmp['close'].values
        else:
            other_close[:,i] = tmp['close'].values
            i += 1
    return target_close, other_close

def log_returns(prices, lag):
    '''
    output: 
        returns[t][i] = log(p_t / p_{t-i}), 1 <= i <= lag
    '''
    returns = np.zeros((prices.shape[0]-lag, lag))
    for i in range(lag):
        returns[:,i] = (np.log(prices) - np.log(np.roll(prices, i+1)))[lag:]
    return returns

def get_features(target_close, other_close, lag):
    '''
    Output:
        features: log returns of ask/bid for all currency pairs; shape (T-lag, 2*lag*cur_pairs); already normalized
    '''
    features = log_returns(target_close, lag)
    for i in range(other_close.shape[1]):
        features = np.append(features, log_returns(other_close[:,i], lag), axis = 1)
        
    normalized_fs = (features - features.mean()) / features.std()
    return normalized_fs

def draw_episode(X, cur, n, split_sz, lag, ts, ccy):
    '''
    Input:
        X: data (bid/ask for each currency pair)
        cur: currency pair that we target to trade
        n: draw the n-th episode
        split_sz: size of episode
        lag: number of lag log-returns z_1,...z_m
        ts: a numpy array of unique time stamps
        ccy: a Series of unique currency pairs
    Output:
        target_close: target currency's bid prices
        target_ask: target currency's ask prices
        features: features for feeding into the neural network
        
    Note: compared to Dai's code, I am not using min_history (min length of a valid episode) as argument; 
        Also, I am not randomly selecting episodes
    '''
    target_close, other_close = generate_episode(X, n, cur, split_sz, ts, ccy)
    features = get_features(target_close, other_close, lag)
    return target_close[lag-1:], features

