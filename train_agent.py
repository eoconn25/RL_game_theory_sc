import numpy as np
import matplotlib.pyplot as plt
from supply_chain_env import LevelKSupplyChainEnv
from sim_config import config

def train_and_evaluate(sim_config, naive = False):
    # initialize environment with our config settings
    env = LevelKSupplyChainEnv(sim_config)
    
    # initialize tabular Q-table
    # our state will consist of [Distributor Inventory, Distributor Backlog, Retailer Order]
    state_dims = [sim_config.max_capacity + 1, sim_config.max_capacity + 1, sim_config.max_order + 1]
    # the action the agent will choose is how much to order from the Manufacturer
    action_dim = sim_config.max_order + 1
    q_table = np.zeros(state_dims + [action_dim])
    
    # hyperparams
    num_episodes = 8000
    alpha = 0.1  # the LR
    gamma = 0.90  # discount factor
    epsilon = 1.0  # exploration prob
    epsilon_decay = 0.999  # slow decay for good epxloration
    min_epsilon = 0.01

    print(f"Training RL Distributor | h_weight: {sim_config.holding_cost} | s_weight: {sim_config.stockout_cost}")

    # --- training loop ---
    for episode in range(num_episodes):
        state, _ = env.reset()
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            inv_idx, backlog_idx, order_idx = state[0], state[1], state[2]  # get info for state
            
            if naive:
                action = order_idx  # if we want to toggle RL agent as a Naive L0 player
            else:
                # epsilon-greedy selection
                if np.random.rand() < epsilon:
                    action = env.action_space.sample()
                else:
                    action = np.argmax(q_table[inv_idx, backlog_idx, order_idx, :])
                    
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_inv, next_backlog, next_order = next_state[0], next_state[1], next_state[2]
            
            # Q-Table update with Bellman Optimality Equation
            old_value = q_table[inv_idx, backlog_idx, order_idx, action]
            next_max = np.max(q_table[next_inv, next_backlog, next_order, :])
            new_value = old_value + alpha * (reward + gamma * next_max - old_value)
            q_table[inv_idx, backlog_idx, order_idx, action] = new_value
            
            state = next_state
            
        epsilon = max(min_epsilon, epsilon * epsilon_decay)  # decay exploration
        
        if (episode + 1) % 1000 == 0:
            print(f"Episode {episode + 1}/{num_episodes} complete | Epsilon: {epsilon:.3f}")
            
    print("Training complete, running a test sim")

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
    distributor_backlog = []
    
    while not (terminated or truncated):
        inv_idx, backlog_idx, order_idx = state[0], state[1], state[2]

        if naive:
            trained_action = order_idx
        else:        
            # choose the optimal action learned during training, pure exploitation
            trained_action = np.argmax(q_table[inv_idx, backlog_idx, order_idx, :])
        
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

        # track distributor's backlog - missed orders
        distributor_backlog.append(env.backlog["Distributor"])
        
        state = next_state

    # --- calculate metrics in evaluation epoch ---
    print("\n=== EVALUATION RESULTS: ORDER VARIANCE ===")
    print(f"Customer Demand Variance:    {np.var(customer_demands):.2f}")
    print(f"Retailer Order Variance:     {np.var(retailer_orders):.2f}")
    print(f"Distributor Order Variance:  {np.var(distributor_orders):.2f}")
    print(f"Manufacturer Order Variance: {np.var(manufacturer_orders):.2f}")
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
    axs[0].plot(retailer_orders, label="Retailer Orders (to Dist)", color="blue", alpha=0.6)
    axs[0].plot(distributor_orders, label="RL Distributor Orders (to Mfg)", color="orange", linewidth=2.5)
    axs[0].plot(manufacturer_orders, label="Manufacturer Orders (to Supplier)", color="red", alpha=0.5)
    
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
    axs[2].plot(distributor_backlog, label="Distributor", color="orange", linewidth=2.5)
    axs[2].set_title("Distributor Backlog")
    axs[2].set_xlabel("Timestep (Days)")
    axs[2].set_ylabel("Backlog (Units)")
    axs[2].legend()
    axs[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

    #plt.xlabel("Timestep (Days)")


if __name__ == "__main__":
    sim_config = config()
    train_and_evaluate(sim_config, naive=True)

