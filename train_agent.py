import numpy as np
import matplotlib.pyplot as plt
from supply_chain_env import LevelKSupplyChainEnv
from sim_config import config

def train_and_evaluate(sim_config, naive = False):
    # --- 1. INITIALIZE ENVIRONMENT ---
    env = LevelKSupplyChainEnv(sim_config)
    
    # --- 2. INITIALIZE THE TABULAR Q-TABLE ---
    # Dimensions: [Dist_Inv, Dist_Backlog, Retailer_Order, Action_Shipment]
    state_dims = [sim_config.max_capacity + 1, sim_config.max_capacity + 1, sim_config.max_order + 1]
    action_dim = sim_config.max_order + 1
    q_table = np.zeros(state_dims + [action_dim])
    
    # --- 3. HYPERPARAMETERS ---
    num_episodes = 8000     # Increased episodes to ensure strong policy convergence
    alpha = 0.1             # Learning rate
    gamma = 0.90            # Discount factor
    epsilon = 1.0           # Exploration probability
    epsilon_decay = 0.999   # Slower decay to allow thorough state space exploration
    min_epsilon = 0.01

    print(f"Training RL Distributor | h_weight: {sim_config.holding_cost} | s_weight: {sim_config.stockout_cost}")

    # --- 4. THE TRAINING LOOP ---
    for episode in range(num_episodes):
        state, _ = env.reset()
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            inv_idx, backlog_idx, order_idx = state[0], state[1], state[2]
            
            if naive:
                action = order_idx
            else:
                # Epsilon-Greedy selection
                if np.random.rand() < epsilon:
                    action = env.action_space.sample()
                else:
                    action = np.argmax(q_table[inv_idx, backlog_idx, order_idx, :])
                    
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_inv, next_backlog, next_order = next_state[0], next_state[1], next_state[2]
            
            # Q-Table Update using Bellman Optimality Equation
            old_value = q_table[inv_idx, backlog_idx, order_idx, action]
            next_max = np.max(q_table[next_inv, next_backlog, next_order, :])
            new_value = old_value + alpha * (reward + gamma * next_max - old_value)
            q_table[inv_idx, backlog_idx, order_idx, action] = new_value
            
            state = next_state
            
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        
        if (episode + 1) % 1000 == 0:
            print(f"Episode {episode + 1}/{num_episodes} complete. Epsilon dropped to {epsilon:.3f}")
            
    print("Training complete, running a test sim")

    # --- 5. EVALUATION EPOCH (VISUALIZATION CAPTURE) ---
    state, _ = env.reset(seed=25)
    terminated = False
    truncated = False
    
    customer_demands = []
    retailer_orders = []
    distributor_orders = []
    manufacturer_orders = []

    retailer_inv = []
    distributor_inv = []
    manufacturer_inv = []
    supplier_inv = []

    distributor_backlog = []
    
    while not (terminated or truncated):
        inv_idx, backlog_idx, order_idx = state[0], state[1], state[2]

        if naive:
            trained_action = order_idx
        else:        
            # Pure exploitation: choose the optimal action learned during training
            trained_action = np.argmax(q_table[inv_idx, backlog_idx, order_idx, :])
        
        print(f"Day: {env.current_step} | Order In: {order_idx} | Action Chosen: {trained_action} | Current Inv: {inv_idx}")
        
        next_state, reward, terminated, truncated, info_dict = env.step(trained_action)
        
        # track everyone's orders and the demand
        customer_demands.append(info_dict['customer_demand'])
        retailer_orders.append(order_idx)
        distributor_orders.append(trained_action)  # WHAT RL CHOSE
        manufacturer_orders.append(env.backlog["Manufacturer"] + trained_action)

        # track everyone's inventory
        retailer_inv.append(env.inv["Retailer"])
        distributor_inv.append(env.inv["Distributor"])
        manufacturer_inv.append(env.inv["Manufacturer"])
        supplier_inv.append(env.inv["Supplier"])

        # track distributor's backlog - missed orders
        distributor_backlog.append(env.backlog["Distributor"])
        
        state = next_state

    # --- 6. CALCULATE EVALUATION VARIANCE ---
    print("\n=== POST-TRAINING RESULTS: ORDER VARIANCE ===")
    print(f"Customer Demand Variance:    {np.var(customer_demands):.2f}")
    print(f"Retailer Order Variance:     {np.var(retailer_orders):.2f}")
    print(f"Distributor Order Variance:  {np.var(distributor_orders):.2f}")
    print(f"Manufacturer Order Variance: {np.var(manufacturer_orders):.2f}")
    print("=============================================")

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

