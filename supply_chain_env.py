import gymnasium as gym
from gymnasium import spaces
import numpy as np

class LevelKSupplyChainEnv(gym.Env):
    
    metadata = {"render_modes":["human"]}

    def __init__(self, config):
        super(LevelKSupplyChainEnv, self).__init__()

        self.T = config.epoch_length
        self.max_capacity = config.max_capacity
        self.max_order = config.max_order
        self.h_weight = config.holding_cost
        self.s_weight = config.stockout_cost

        # ACTION SPACE
        self.action_space = spaces.Discrete(self.max_order + 1)  # Distributor chooses a discrete shipment quant to send downstream

        # OBSERVATION SPACE
        # state vector consists of [distributor inventory, distributor backlog, incoming retailer order]
        self.observation_space = spaces.MultiDiscrete([
            self.max_capacity + 1,
            self.max_capacity + 1,
            self.max_order + 1
        ])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_step = 0

        # initialize baseline inventory
        self.inv = {"Supplier": 20, "Manufacturer": 20, "Distributor": 20, "Retailer": 20}
        self.backlog = {"Supplier": 0, "Manufacturer": 0, "Distributor": 0, "Retailer": 0}

        # baseline customer demand
        initial_demand = int(self.np_random.poisson(lam=10))
        clamped_initial_demand = int(np.clip(initial_demand, 0, self.max_order))

        # initial state tuple for the distributor
        initial_state = np.array([
            self.inv["Distributor"],
            self.backlog["Distributor"],
            clamped_initial_demand
        ], dtype=np.int32)

        return initial_state, {}
    
    def step(self, action):
        # process one daily tick of the supply chain
        self.current_step += 1
        #distributor_shipment_choice = action

        # ---- 1. Retailer's demand from Customer ----
        customer_demand = int(self.np_random.poisson(lam=10))
        retailer_total_demand = self.backlog["Retailer"] + customer_demand

        # customer "buys" demand, update retailers inventory and backlog
        retailer_shipment = min(self.inv["Retailer"], retailer_total_demand)  # give what customer wants or all they have
        self.inv["Retailer"] -= retailer_shipment
        self.backlog["Retailer"] = retailer_total_demand - retailer_shipment

        # retailer issues order upstream - - LEVEL 0 NAIVE
        retailer_order = customer_demand

        # ---- 2. Distributor moves, our RL agent ----
        distributor_total_demand = self.backlog["Distributor"] + retailer_order

        # get shipment to retailer based on RL agent's choice
        # should be its entire inventory or the entire demand
        distributor_shipment = min(self.inv["Distributor"], distributor_total_demand)

        # send to Retailer, update status
        self.inv["Retailer"] += distributor_shipment
        self.inv["Distributor"] -= distributor_shipment
        self.backlog["Distributor"] = distributor_total_demand - distributor_shipment

        # distributor issues order to manufacturer - PLACEHOLDER, needs to be chosen by RL agent
        distributor_order = action

        # ---- 3. Manufacturer moves ----
        manufacturer_total_demand = self.backlog["Manufacturer"] + distributor_order
        manufacturer_shipment = min(self.inv["Manufacturer"], manufacturer_total_demand)

        # send to distributor, update status
        self.inv["Distributor"] += manufacturer_shipment
        self.inv["Manufacturer"] -= manufacturer_shipment
        self.backlog["Manufacturer"] = manufacturer_total_demand - manufacturer_shipment

        # manufacturer issues order to supplier - - LEVEL 0 NAIVE
        manufacturer_order = manufacturer_total_demand

        # ---- 4. Supplier moves ----
        supplier_total_demand = self.backlog["Supplier"] + manufacturer_order
        supplier_shipment = min(self.inv["Supplier"], supplier_total_demand)

        # send to manufacturer, update status
        self.inv["Manufacturer"] += supplier_shipment
        self.inv["Supplier"] -= supplier_shipment
        self.backlog["Supplier"] = supplier_total_demand - supplier_shipment

        # supplier makes new stock - - NEED TO TUNE PROBABLY
        self.inv["Supplier"] += 15


        # ---- Enforce boundaries to prevent overflow ----
        for key in self.inv:
            self.inv[key] = int(np.clip(self.inv[key], 0, self.max_capacity))
            self.backlog[key] = int(np.clip(self.backlog[key], 0, self.max_capacity))
        clamped_retailer_order = int(np.clip(retailer_order, 0, self.max_order))
        
        # ---- calculate multi-objective reward for agent ----
        # stockout objective
        #r_stockout = -10.0 if self.backlog["Distributor"] > 0 else 0.0
        r_stockout = -self.s_weight * self.backlog["Distributor"]
        # low inventory objective
        r_holding = -self.h_weight * (1.0 * self.inv["Distributor"])
        # combine
        reward = r_stockout + r_holding

        # ---- return packet ----
        next_state = np.array([
            self.inv["Distributor"],
            self.backlog["Distributor"],
            clamped_retailer_order
        ], dtype=np.int32)

        terminated = self.current_step >= self.T
        truncated = False

        return next_state, reward, terminated, truncated, {'customer_demand': customer_demand}


