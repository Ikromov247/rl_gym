#!/usr/bin/env python
# coding: utf-8

# In[1]:
import numpy as np
import mlflow

# In[2]:
import gymnasium as gym


# The classic cart pole problem is an introduction to Reinforcement Learning. \
# An agent learns to balance a pole on a cart by moving it left or right 

# In[3]:


env = gym.make('CartPole-v1', render_mode=None)
env.reset(seed=42)
# In[4]
experiment = mlflow.set_experiment('cart_pole')
run_name = 'cart_pole rl'

cart_position = 8
cart_velocity = 6 
pole_angle = 10 
pole_velocity = 10
cart_velocity_bounds = (-1.2, 1.2)
pole_velocity_bounds = (-1.2, 1.2)

n_episodes=12000
alpha=0.05
gamma=0.995
epsilon_start=1.0
epsilon_end=0.01
epsilon_decay=4000

# In[7]:

n_buckets = (cart_position, cart_velocity, pole_angle, pole_velocity) # cart position, cart velocity, pole angle, pole velocity

state_bounds = list(zip(env.observation_space.low, env.observation_space.high))
state_bounds[1] =  cart_velocity_bounds # cart velocity bounds
state_bounds[3] = pole_velocity_bounds # pole velocity bounds


# In[8]:

def discretize_state(observation):
    """
    Convert a continuous state to its discrete counterpart
    """
    discretized = []
    for i, (lower, upper) in enumerate(state_bounds):
        if upper == float('inf'):
            upper = env.unwrapped.spec.kwargs.get('thresholds', [2.4, 2.4, 0.418, 0.418])[i]
        if lower == float('-inf'):
            lower = -upper
        
        scaling = (n_buckets[i] -1) / (upper - lower)
        new_obs = int(np.floor(scaling * (observation[i] - lower)))
        new_obs = min(n_buckets[i] - 1, max(0, new_obs))
        discretized.append(new_obs)
    return tuple(discretized)


# ## Q-Table initialization

# In[9]:


def create_q_table():
    """
    Create and initialize Q-table with zeros
    """
    q_table_shape = n_buckets + (env.action_space.n,)
    q_table = np.zeros(q_table_shape)
    return q_table


# ## Epsilon-greedy action selection

# In[10]:


def select_action(state, q_table, epsilon):
    """
    Select an action using epsilon-greedy policy
    """
    if np.random.random() < epsilon:
        # explore: select a random action
        return env.action_space.sample()
    else:
        # exploit: select action with the highest q-value
        return np.argmax(q_table[state])


# ## Q-learning update rule

# In[11]:


def update_q_value(state, action, reward, next_state, q_table, alpha, gamma):
    """
    Update Q-value for a state-action pair
    """
    next_max_q = np.max(q_table[next_state])
    # current q = 100
    # next max q = 70
    # reward 20
    current_q = q_table[state][action]
    new_q = current_q + alpha * (reward + gamma * next_max_q - current_q)
    q_table[state][action] = new_q


# ## Training Loop

# In[12]:


def train_agent(n_episodes, alpha, gamma, epsilon_start, epsilon_end, epsilon_decay):
    """
    Train the Q-learning agent
    """
    q_table = create_q_table()

    total_episode_states = {}
    total_episode_states['total_reward'] = []
    total_episode_states['episode_length'] = []
    total_episode_states['batch_reward_mean'] = []
    total_episode_states['batch_reward_std'] = []

    for episode in range(n_episodes):
        episode_state = {}

        # Reduce epsilon (exploration rate) over time
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * np.exp(-episode / epsilon_decay)
        episode_state['epsilon'] = epsilon

        # reset environment
        observation, info = env.reset()
        state = discretize_state(observation)
        
        done = False
        total_reward = 0
        episode_length = 0
        # One episode of training
        while not done:
            action = select_action(state, q_table, epsilon)

            # Take action and observe the result
            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_state = discretize_state(next_observation)

            update_q_value(state, action, reward, next_state, q_table, alpha, gamma)

            # move to the next state
            state = next_state
            total_reward += reward
            episode_length += 1
        
        total_episode_states['total_reward'].append(total_reward)
        total_episode_states['episode_length'].append(episode_length)

        if episode % 50 == 0:
            batch_rewards = total_episode_states['total_reward'][episode-50:]
            mean_reward = np.mean(batch_rewards).round(0)
            std_reward = np.std(batch_rewards).round(2)
            print(f"Episode {episode}, Average reward: {mean_reward} with std {std_reward}, Epsilon: {epsilon:.4f}")
            
            total_episode_states['batch_reward_mean'].append(mean_reward)
            total_episode_states['batch_reward_std'].append(std_reward)

    return q_table, total_episode_states


# ## Testing the trained agent

# In[13]:


def test_agent(q_table, n_episodes=10, render=True):
    """
    Test the trained Q-learning agent over several episodes
    
    Args:
        q_table: The learned Q-table
        n_episodes: Number of test episodes to run
        render: Whether to render the environment (set to True to visualize)
    
    Returns:
        Average episode length across all test episodes
    """
    env_test = gym.make('CartPole-v1', render_mode='human' if render else None)
    episode_lengths = []
    
    for episode in range(n_episodes):
        # Reset the environment
        observation, info = env_test.reset()
        state = discretize_state(observation)
        
        done = False
        episode_length = 0
        
        # Run one episode
        while not done:
            # Always select the best action (no exploration)
            action = np.argmax(q_table[state])
            
            # Take action
            next_observation, reward, terminated, truncated, info = env_test.step(action)
            done = terminated or truncated
            
            # Update state and counter
            state = discretize_state(next_observation)
            episode_length += 1
            
        episode_lengths.append(episode_length)
        print(f"Test Episode {episode+1}/{n_episodes}, Length: {episode_length}")
    
    avg_length = sum(episode_lengths) / len(episode_lengths)
    print(f"Average episode length: {avg_length:.2f}")
    
    env_test.close()
    return avg_length


# In[14]:


q_table, total_episode_states = train_agent(
    n_episodes=n_episodes,
    alpha=alpha,
    gamma=gamma,
    epsilon_start=epsilon_start,
    epsilon_end=epsilon_end,
    epsilon_decay=epsilon_decay
)


# In[15]:


avg_length = test_agent(q_table, render=False, n_episodes=20)


# In[16]:
params = {
    "cart_position_bins": cart_position,
    "cart_velocity_bins": cart_velocity,
    "pole_angle_bins": pole_angle,
    "pole_velocity_bins": pole_velocity,
    "cart_velocity_bounds_min": cart_velocity_bounds[0],
    "cart_velocity_bounds_max": cart_velocity_bounds[1],
    "pole_velocity_bounds_min": pole_velocity_bounds[0],
    "pole_velocity_bounds_max": pole_velocity_bounds[1],
    "n_episodes": n_episodes,
    "alpha": alpha,
    "gamma": gamma,
    "epsilon_start": epsilon_start,
    "epsilon_end": epsilon_end,
    "epsilon_decay": epsilon_decay
}

with mlflow.start_run(run_name=run_name) as run:
    mlflow.log_metric('average_test_length', avg_length)
    
    average_train_reward = np.mean(total_episode_states['batch_reward_mean']).round(0)
    mlflow.log_metric('average_train_reward', average_train_reward)

    average_train_std = np.mean(total_episode_states['batch_reward_std']).round(0)
    mlflow.log_metric('average_train_std', average_train_std)

    mlflow.log_metrics(params)