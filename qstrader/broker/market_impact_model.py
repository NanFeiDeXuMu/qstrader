
class MarketImpactModel(object):
    def __init__(self, gamma_perm=0.01, alpha=1.0, tau=1.0):
        '''
        gamma_perm : `float`
            The permanent market impact coefficient.
        alpha : `float`, optional
            The exponent for the market impact function. Defaults to 1.0 (linear).
        tau : `float`, optional
            The time constant for the decay of temporary market impact. Defaults to 1.0.
        '''
        self.gamma_perm = gamma_perm
        self.alpha = alpha 
        self.tau = tau

    def __call__(self, quantity):
        '''
        Calculate the market impact cost for a given order quantity.

        Parameters
        ----------
        quantity : `int`
            The quantity of the order.

        Returns
        -------
        `float`
            The estimated market impact cost.
        '''
        if quantity == 0:
            return 0.0
        trading_rate = abs(quantity) / self.tau
        g_v = self.gamma_perm * (trading_rate ** self.alpha)
        sign = 1 if quantity > 0 else -1
        return sign * g_v * self.tau