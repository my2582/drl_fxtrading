import numpy as np

class PortfolioManagement:
    def __init__(self, init_price, M=9, init_wealth=10000):
        '''
        Parameters:
            M(int): the number of assets (excluding cash)
            P0(int): an initial portfolio value (=usually in cash)
        '''

        self.M = M
        self.wt = np.zeros((1, M+1))
        self.wt[0][0] = 1    # cash 100% at time 0
        self.qty = np.zeros((1,M+1))
        self.qty[0][0] = init_wealth
        self.rel_price = np.ones_like(init_price)
        self.price = np.empty(shape=(1, self.M+1))
        self.price[0] = np.array(init_price)
        self.init_price = init_price
        self.transaction_costs = 0
        self.init_wealth = init_wealth

    def get_pf_value_history(self):
        return [self.get_pf_value(i) for i in range(self.qty.shape[0])]

    def get_pf_value(self, idx=-1):
        # an initial condition handling.
        if self.price.shape[0] < abs(idx):
            return self.init_wealth
        
        qty = self.qty[idx]
        price = self.price[idx]
        return np.sum([qty[i]*price[i] for i in range(qty.shape[0])])

    def get_noise_price(self, trade_price):
        # With scale = (3*10)**-2,
        # noise would be ranged 0.996 to 1.004 for 10,000 draws.
        noise = np.random.normal(loc=1, scale=(3*10)**-2, size=self.M)
        return trade_price*noise
    
    def get_trading_quantity(self, cur_wt, target_wt, executed_price):
        return np.floor(self.get_pf_value()*(target_wt - cur_wt) / executed_price)

    def get_target_weight(self, action):
        if action == -1:
            # short
            return [2.0, -1.0]
        elif action == 0:
            # no trade
            return self.wt[-1]
        else:
            # long
            return [0.0, 1.0]
        
    def trade(self, cur_wt, target_wt, target_price, cost_bp = 10):
        '''
        Parameters:
            cur_wt(np array): the current portfolio weight vector (as is)
            target_wt(np array): the target portfolio weight vector (to be)
            target_price(np array): currency prices at which are traded or evaluated
        '''
        # We evaluated using (exact) target prices
        # 1 is cash price (always 1)
        target_price_vector = np.array([1, target_price])
        self.add_price(price = target_price_vector)

        # # But traded prices can be different from the target prices. (can be either of positive or negative)
        # # traded_price = [1, prices...], where 1 is always case price.
        # traded_price = self.get_noise_price(target_price_vector)

        # # Add transaction costs. Buy at executed_price(1+cost_bp) or sell at executed_price(1-cost_bp)
        # executed_price = [traded_price[i]*(1+cost_bp/(10^4)) if (target_wt[i] - cur_wt[i]) >= 0 else traded_price[i]*(1-cost_bp/(10^4)) for i in range(1,len(cur_wt))]
        # executed_price = np.append(1, executed_price)

        # Compute quantities to trade for each currency
        trading_qty = self.get_trading_quantity(cur_wt, target_wt, target_price_vector)

        # Do the trading
        new_qty = self.qty[-1] + trading_qty
        self.add_qty(new_qty=new_qty)

        new_wt = (new_qty*target_price_vector) / self.get_pf_value()
        self.add_wt(new_wt=new_wt)
        self.add_rel_price(price=target_price)
    
    def add_rel_price(self, price):
        self.rel_price = np.vstack((self.rel_price, self.price[-1]/self.init_price))
    
    def add_wt(self, new_wt):
        self.wt = np.vstack((self.wt, new_wt))
    
    def add_qty(self, new_qty):
        self.qty = np.vstack((self.qty, new_qty))
    
    def add_price(self, price):
        self.price = np.vstack((self.price, price))
