import matplotlib.pyplot as plt

# Rewards from previous method
rl_rewards = [500.0,500.0,500.0,500.0,500.0,500.0,500.0,500.0,500.0,500.0,500.0,500.0,500.0,500.0,500.0,500.0,500.0,500.0,500.0,500.0]  # Replace with your actual rewards

# Rewards from RL approach
prev_rewards = [39.0,19.0,17.0,10.0,27.0,21.0,14.0,20.0,14.0,16.0,15.0,16.0,15.0,10.0,17.0,51.0,22.0,33.0,13.0,24.0]  # Replace with your actual rewards

# Plotting rewards
plt.plot(range(len(prev_rewards)), prev_rewards, label='Previous Method')
plt.plot(range(len(rl_rewards)), rl_rewards, label='RL Approach')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Comparison of Rewards: Previous Method vs. RL Approach')
plt.legend()
plt.show()
