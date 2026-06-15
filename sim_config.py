
class config:
    def __init__(self):
        # capacity parameters
        self.max_capacity = 25
        self.max_order = 15
        self.epoch_length = 30

        # penalty weight parameters
        self.holding_cost = 1.0
        self.stockout_cost = 10.0

        # player behavior
        