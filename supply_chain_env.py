import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque

#self.supplier_schedule_queue = deque([10,10], maxlen=2)


class LevelKSupplyChainEnv(gym.Env):
    metadata = {"render_modes":["human"]}

    def __init__(self, config):
        super(LevelKSupplyChainEnv, self).__init__()

        # initialize parameters from the received config object
        self.T = config.epoch_length
        self.max_capacity = config.max_capacity
        self.max_order = config.max_order
        self.h_weight = config.holding_cost
        self.s_weight = config.stockout_cost

        # initialize behavior settings
        self.retailer_level = config.retailer_level
        self.manufacturer_level = config.manufacturer_level
        self.supplier_level = config.supplier_level

        # initialize material queues
        self.ship_DtR_queue = deque([10,10], maxlen=2)
        self.ship_MtD_queue = deque([10,10], maxlen=2)
        self.ship_StM_queue = deque([10,10], maxlen=2)
        self.supplier_production_queue = deque([10,10], maxlen=2)

        # initialize information queues
        self.order_RtD_queue = deque([10,10], maxlen=2)
        self.order_DtM_queue = deque([10,10], maxlen=2)
        self.order_MtS_queue = deque([10,10], maxlen=2)

        # ACTION SPACE
        self.action_space = spaces.Discrete(self.max_order + 1)  # Distributor chooses a discrete order quantity

        # OBSERVATION SPACE
        # state vector will have [distributor inventory, distributor backlog, latest retailer demand]
        self.observation_space = spaces.MultiDiscrete([
            self.max_capacity + 1,
            self.max_capacity + 1,
            self.max_order + 1
        ])

    def reset(self, seed=None):
        super().reset(seed=seed)
        
        self.current_step = 0

        # initialize baseline inventory
        self.inv = {"Supplier": 10, "Manufacturer": 10, "Distributor": 10, "Retailer": 10}
        self.backlog = {"Supplier": 0, "Manufacturer": 0, "Distributor": 0, "Retailer": 0}
        
        # initialize material queues
        self.ship_DtR_queue = deque([10,10], maxlen=2)
        self.ship_MtD_queue = deque([10,10], maxlen=2)
        self.ship_StM_queue = deque([10,10], maxlen=2)
        self.supplier_production_queue = deque([10,10], maxlen=2)

        # initialize information queues
        self.order_RtD_queue = deque([10,10], maxlen=2)
        self.order_DtM_queue = deque([10,10], maxlen=2)
        self.order_MtS_queue = deque([10,10], maxlen=2)

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
    
        
    # helper function for generating orders/demand according to level
    def ordering_logic(self, entity:str, demand:int, level:str):
        # determines order quantity to send upstream accoridng to the day's demand
        if level == 'L0':
            order = demand  # LEVEL 0 NAIVE
        elif level == 'L1':
            order = round(demand * 1.5)  # LEVEL 1 SHORTAGE GAME
        elif level == 'L2':
            order = round(demand * 2)  # LEVEL 2 EXTREME SHORTAGE GAME
        else:
            print('ERROR UNCLEAR LEVEL')
        
        # total order, adds in the backlog
        total_order = order + self.backlog[entity]
        return total_order


    def step(self, action):
        # process one daily tick of the supply chain
        self.current_step += 1

        # RECEIVE SHIPMENTS FROM PREVIOUS DAYS
        DtR_shipment = self.ship_DtR_queue.popleft()
        MtD_shipment = self.ship_MtD_queue.popleft()
        StM_shipment = self.ship_StM_queue.popleft()
        supplier_production = self.supplier_production_queue.popleft()

        self.inv['Retailer'] += DtR_shipment
        self.inv['Distributor'] += MtD_shipment
        self.inv['Manufacturer'] += StM_shipment
        self.inv['Supplier'] += supplier_production

        # RECEIVE ORDERS/DEMAND
        CtR_demand = int(self.np_random.poisson(lam=10)) # demand from customer
        RtD_demand = self.order_RtD_queue.popleft()
        DtM_demand = self.order_DtM_queue.popleft()
        MtS_demand = self.order_MtS_queue.popleft()

        # FILL ORDERS - SHIP OUT
        RtC_fill = min(self.inv['Retailer'], CtR_demand + self.backlog['Retailer'])
        DtR_fill = min(self.inv['Distributor'], RtD_demand + self.backlog['Distributor'])
        MtD_fill = min(self.inv['Manufacturer'], DtM_demand + self.backlog['Manufacturer'])
        StM_fill = min(self.inv['Supplier'], MtS_demand + self.backlog['Supplier'])

        self.ship_DtR_queue.append(DtR_fill)
        self.ship_MtD_queue.append(MtD_fill)
        self.ship_StM_queue.append(StM_fill)

        self.inv['Retailer'] -= RtC_fill
        self.inv['Distributor'] -= DtR_fill
        self.inv['Manufacturer'] -= MtD_fill
        self.inv['Supplier'] -= StM_fill    

        # UPDATE BACKLOG
        self.backlog['Retailer'] = CtR_demand + self.backlog['Retailer'] - RtC_fill
        self.backlog['Distributor'] = RtD_demand + self.backlog['Distributor']  - DtR_fill
        self.backlog['Manufacturer'] = DtM_demand  + self.backlog['Manufacturer'] - MtD_fill
        self.backlog['Supplier'] = MtS_demand  + self.backlog['Supplier'] - StM_fill

        # SELECT A NEW ORDER TO ADD TO QUEUE
        self.order_RtD_queue.append(self.ordering_logic('Retailer', CtR_demand, self.retailer_level))
        action_set = RtD_demand + self.backlog['Distributor']
        self.order_DtM_queue.append(action_set)
        self.order_MtS_queue.append(self.ordering_logic('Manufacturer', DtM_demand, self.manufacturer_level))
        self.supplier_production_queue.append(self.ordering_logic('Supplier', MtS_demand, self.supplier_level))
        
        # AGENT'S REWARD
        # stockout objective
        r_stockout = -self.s_weight * self.backlog["Distributor"]
        # low inventory objective
        r_holding = -self.h_weight * (1.0 * self.inv["Distributor"])
        # combine
        reward = r_stockout + r_holding

        # return packet
        next_state = np.array([
            min(self.inv["Distributor"], self.max_capacity),
            min(self.backlog["Distributor"], self.max_capacity),
            min(RtD_demand, self.max_order)
        ], dtype=np.int32)

        terminated = self.current_step >= self.T
        truncated = False

        return next_state, reward, terminated, truncated, {'customer_demand': CtR_demand, 
                                                           'retailer_order': RtD_demand, 
                                                           'manufacturer_order': MtS_demand,
                                                           'supplier_order': supplier_production
                                                           }




if __name__=='__main__':
    from sim_config import config
    import matplotlib.pyplot as plt

    sim_config = config()
    env = LevelKSupplyChainEnv(sim_config)

    # --- evaluation epoch to visualize ---
    state, _ = env.reset(seed=25)
    terminated = False
    truncated = False
    
    # list initialization for metrics we will track
    # demands and orders
    customer_demands = []
    retailer_orders = []
    distributor_orders = []
    manufacturer_orders = []
    supplier_orders = []
    
    # inventory levels
    retailer_inv = []
    distributor_inv = []
    manufacturer_inv = []
    supplier_inv = []

    # backlog/stockouts of the distributor
    retailer_backlog = []
    distributor_backlog = []
    manufacturer_backlog = []
    supplier_backlog = []
    
    while not (terminated or truncated):
        inv_idx, backlog_idx, order_idx = state[0], state[1], state[2]

        trained_action = order_idx + backlog_idx
        
        # print(f"Day: {env.current_step} | Order In: {order_idx} | Action Chosen: {trained_action} | Current Inv: {inv_idx}")
        
        next_state, reward, terminated, truncated, info_dict = env.step(trained_action)
        
        # track everyone's orders and the demand
        customer_demands.append(info_dict['customer_demand'])
        retailer_orders.append(info_dict['retailer_order'])
        distributor_orders.append(trained_action)  # WHAT RL CHOSE
        manufacturer_orders.append(info_dict['manufacturer_order'])
        supplier_orders.append(info_dict['supplier_order'])

        # track everyone's inventory
        retailer_inv.append(env.inv["Retailer"])
        distributor_inv.append(env.inv["Distributor"])
        manufacturer_inv.append(env.inv["Manufacturer"])
        supplier_inv.append(env.inv["Supplier"])

        # track backlog - missed orders
        retailer_backlog.append(env.backlog["Retailer"])
        distributor_backlog.append(env.backlog["Distributor"])
        manufacturer_backlog.append(env.backlog["Manufacturer"])
        supplier_backlog.append(env.backlog["Supplier"])
        
        state = next_state

    # --- calculate metrics in evaluation epoch ---
    print("\n=== EVALUATION RESULTS: ORDER VARIANCE ===")
    print(f"Customer Demand Variance:    {np.var(customer_demands):.2f}")
    print(f"Retailer Order Variance:     {np.var(retailer_orders):.2f}")
    print(f"Distributor Order Variance:  {np.var(distributor_orders):.2f}")
    print(f"Manufacturer Order Variance: {np.var(manufacturer_orders):.2f}")
    print(f"Supplier Production Variance: {np.var(supplier_orders):.2f}")
    print("=============================================")
    print(f"Total Bullwhip Distortion: {(np.var(manufacturer_orders) / np.var(customer_demands)):.2f}")
    print("=============================================")
    print(f"Total Customer Demand: {sum(customer_demands)}")
    print(f"Total Distributor Inventory: {sum(distributor_inv)}")
    print(f"Total Distributor Stockouts: {sum(distributor_backlog)}")


    # --- create visual for our metrics ---
    fig, axs = plt.subplots(3,1, figsize=(9,8), sharex=True)
    
    # plot the orders and demand
    axs[0].plot(customer_demands, label="End-Customer Demand", color="green", linestyle=":")
    axs[0].plot(retailer_orders, label="Retailer Orders (to Dist)", color="blue", alpha=0.5)
    axs[0].plot(distributor_orders, label="RL Distributor Orders (to Mfg)", color="orange", linewidth=2.5)
    axs[0].plot(manufacturer_orders, label="Manufacturer Orders (to Supplier)", color="red", alpha=0.5)
    axs[0].plot(supplier_orders, label="Supplier Production", color="purple", alpha=0.5)

    
    axs[0].set_title(f"Observed Bullwhip with Tabular RL Distributor ({sim_config.holding_cost}, {sim_config.stockout_cost})")
    axs[0].set_ylabel("Order Quantity (Units)")
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)

    # --- plot inventory levels ---
    axs[1].plot(retailer_inv, label="Retailer", color="blue", alpha=0.6)
    axs[1].plot(distributor_inv, label="RL Distributor", color="orange", linewidth=2.5)
    axs[1].plot(manufacturer_inv, label="Manufacturer", color="red", alpha=0.5)
    axs[1].plot(supplier_inv, label="Supplier", color="purple", alpha=0.5)

    axs[1].set_title("Inventory Levels with Tabular RL Agent")
    axs[1].set_ylabel("Inventory Level (Units)")
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)

    # --- plot distributor backlog ---
    axs[2].plot(retailer_backlog, label="Retailer", color="green", alpha=0.5)
    axs[2].plot(distributor_backlog, label="Distributor", color="orange", linewidth=2.5)
    axs[2].plot(manufacturer_backlog, label="Manufacturer", color="red", alpha=0.5)
    axs[2].plot(supplier_backlog, label="Supplier", color="purple", alpha=0.5)

    axs[2].set_title("Backlog levels")
    axs[2].set_xlabel("Timestep (Days)")
    axs[2].set_ylabel("Backlog (Units)")
    axs[2].legend()
    axs[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()



'''        # add boundaries to prevent overflow
        for key in self.inv:
            self.inv[key] = int(np.clip(self.inv[key], 0, self.max_capacity))
            self.backlog[key] = int(np.clip(self.backlog[key], 0, self.max_capacity))
        
        clamped_retailer_order = int(np.clip(retailer_order, 0, self.max_order))'''
'''

        # MATERIAL LOOP - ship the orders from the previous day
        supplier_shipment =  min(self.inv['Supplier'], self.last_manufacturer_order)
        manufacturer_shipment =  min(self.inv['Manufacturer'], self.last_distributor_order)
        distributor_shipment =  min(self.inv['Distributor'], self.last_retailer_order)
        retailer_shipment =  min(self.inv['Retailer'], self.last_customer_demand + self.backlog['Retailer'])

        # UPDATE INVENTORIES - now that everything has been shipped, update our inventories
        self.inv['Retailer'] -= retailer_shipment  # R to customer
        
        self.inv['Retailer'] += distributor_shipment  # D to R
        self.inv['Distributor'] -= distributor_shipment

        self.inv['Distributor'] += manufacturer_shipment  # M to D
        self.inv['Manufacturer'] -= manufacturer_shipment
        
        self.inv['Manufacturer'] += supplier_shipment  # S to M
        self.inv['Supplier'] -= supplier_shipment

        self.inv['Supplier'] += self.last_supplier_production  # supplier generates its own new inventory

        # UPDATE BACKLOGS
        self.backlog['Retailer'] = (self.backlog['Retailer'] + self.last_customer_demand) - retailer_shipment
        self.backlog['Distributor'] = self.last_retailer_order - distributor_shipment
        self.backlog['Manufacturer'] = self.last_distributor_order - manufacturer_shipment
        self.backlog['Supplier'] = self.last_manufacturer_order - supplier_shipment

        # INFORMATION LOOP - starting with customer, demand moves from downstream to upstream
        customer_demand = int(self.np_random.poisson(lam=10))  # customer demand
        
        retailer_order = self.ordering_logic('Retailer', customer_demand, self.retailer_level)
        distributor_order = action  # RL agent
        manufacturer_order = self.ordering_logic('Manufacturer', distributor_order, self.manufacturer_level)
        supplier_production = self.ordering_logic('Supplier', manufacturer_order, self.supplier_level)

        # SHIFT TODAYS DECISIONS TO YESTERDAYS SLOT
        self.last_customer_demand = customer_demand
        self.last_retailer_order = retailer_order
        self.last_distributor_order = distributor_order
        self.last_manufacturer_order = manufacturer_order
        self.last_supplier_production = supplier_production
'''

'''    def step(self, action):
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

        retailer_order = self.ordering_logic("Retailer", "Distributor", retailer_total_demand, self.retailer_level)

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

        # supplier makes new stock - - currently just replaces what it ships
        self.inv["Supplier"] += supplier_shipment


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
'''