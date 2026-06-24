
class config:
    def __init__(self):
        # capacity parameters
        self.max_capacity = 25
        self.max_order = 15
        self.epoch_length = 50

        # penalty weight parameters
        self.holding_cost = 1.0
        self.stockout_cost = 10.0

        # player behavior - choose from L0, L1, L2
        self.retailer_level = 'L0'
        self.manufacturer_level = 'L0'
        self.supplier_level = 'L0'
