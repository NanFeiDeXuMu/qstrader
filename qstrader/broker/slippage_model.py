class SlippageModel(object):
    def __init__(self, eta=0.1, alpha=0.5, tau=1.0):
        self.eta = eta
        self.alpha = alpha
        self.tau = tau

    def __call__(self, asset, quantity, current_price, dt):
        if quantity == 0:
            return current_price
        trading_rate = abs(quantity) / self.tau
        h_v = self.eta * (trading_rate ** self.alpha)
        sign = 1 if quantity > 0 else -1
        return current_price + sign * h_v