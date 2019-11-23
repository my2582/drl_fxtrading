import numpy as np

class PortfolioManagement:
    def __init__(self, init_price, M=9, init_wealth=10000):
        '''
        Parameters:
            M(int): the number of assets (excluding cash)
            P0(int): an initial portfolio value (=usually in cash)
        '''

        self.wt = np.zeros((1, M+1))
        self.wt[0][0] = 1    # cash 100% at time 0
        self.qty = np.zeros((1,M+1))
        self.qty[0][0] = init_wealth
        self.rel_price = np.ones_like(init_price)
        self.price = np.array(init_price)
        self.init_price = init_price
        self.transaction_costs = 0

    def get_pf_value(self, idx=-1):
        return self.qty[idx]*self.price[idx]

    def get_noise_price(self, trade_price):
        # With scale = (3*10)**-2,
        # noise would be ranged 0.996 to 1.004 for 10,000 draws.
        noise = np.random.normal(loc=1, scale=(3*10)**-2, size=trade_price.shape[0])
        return trade_price*noise
    
    def get_quantity(self, cur_wt, target_wt, executed_price):
        return np.floor(self.get_pf_value()*target_wt / executed_price)
        
    def trade(self, cur_wt, target_wt, target_price, cost_bp = 10):
        '''
        Parameters:
            cur_wt(np array): the current portfolio weight vector (as is)
            target_wt(np array): the target portfolio weight vector (to be)
            target_price(np array): currency prices at which are traded or evaluated
        '''
        # We evaluated using (exact) target prices
        self.add_price(price = target_price)

        # But traded prices can be different from the target prices. (can be either of positive or negative)
        traded_price = self.get_noise_price(self.price[-1])

        # Add transaction costs. Buy at executed_price(1+cost_bp) or sell at executed_price(1-cost_bp)
        executed_price = [traded_price[i]*(1+cost_bp/(10^4)) if (target_wt[i] - cur_wt[i]) >= 0 else traded_price[i]*(1-cost_bp/(10^4)) for i in range(len(cur_wt))]

        # Compute quantities to trade for each currency
        trading_qty = self.get_quantity(cur_wt, target_wt, executed_price)

        # Do the trading
        new_qty = self.qty[-1] + trading_qty
        self.add_qty(new_qty=new_qty)

        new_wt = new_qty*target_price / self.get_pf_value()
        self.add_wt(new_wt=new_wt)
        self.add_rel_price(price=target_price)
        self.add_transaction_cost(cost_bp)
    
    def add_rel_price(self, price):
        self.rel_price = np.append(self.rel_price, self.rel_price[-1]/self.init_price, axis=0)
    
    def add_wt(self, new_wt):
        self.wt = np.append(self.wt, new_wt, axis=0)
    
    def add_qty(self, new_qty):
        self.qty = np.append(self.qty, new_qty, axis=0)
    
    def add_price(self, price):
        self.price = np.append(self.price, price, axis=0)
