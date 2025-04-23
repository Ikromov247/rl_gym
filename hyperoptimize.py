#!/usr/bin/env python
# coding: utf-8

import numpy as np
import gymnasium as gym
import mlflow
import optuna
from optuna.integration.mlflow import MLflowCallback


class CartPoleAgent:
    """Q-learning agent for CartPole environment with discretized state space"""
    
    def __init__(self, config):
        """
        Initialize the agent with the given configuration
        
        Args:
            config: Dictionary containing hyperparameters
        """
        # Environment setup
        self.env = gym.make('CartPole-v1', render_mode=None)
        self.env.reset(seed=42)
        
        # Discretization parameters
        self.cart_position_bins = config['cart_position_bins']
        self.cart_velocity_bins = config['cart_velocity_bins']
        self.pole_angle_bins = config['pole_angle_bins']
        self.pole_velocity_bins = config['pole_velocity_bins']
        self.cart_velocity_bounds = (config['cart_velocity_bounds_min'], 
                                    config['cart_velocity_bounds_max'])
        self.pole_velocity_bounds = (config['pole_velocity_bounds_min'], 
                                    config['pole_velocity_bounds_max'])
        
        # Training parameters
        self.n_episodes = config['n_episodes']
        self.alpha = config['alpha']
        self.gamma = config['gamma']
        self.epsilon_start = config['epsilon_start']
        self.epsilon_end = config['epsilon_end']
        self.epsilon_decay = config['epsilon_decay']
        
        # Set up state buckets
        self.n_buckets = (self.cart_position_bins, 
                          self.cart_velocity_bins, 
                          self.pole_angle_bins, 
                          self.pole_velocity_bins)
        
        # Define state bounds
        self.state_bounds = list(zip(self.env.observation_space.low, 
                                     self.env.observation_space.high))
        self.state_bounds[1] = self.cart_velocity_bounds
        self.state_bounds[3] = self.pole_velocity_bounds
        
        # Initialize Q-table
        self.q_table = self.create_q_table()
        
        # Initialize metrics storage
        self.metrics = {}
    
    def discretize_state(self, observation):
        """
        Convert a continuous state to its discrete counterpart
        
        Args:
            observation: Raw environment observation
            
        Returns:
            Tuple representing the discretized state
        """
        discretized = []
        for i, (lower, upper) in enumerate(self.state_bounds):
            if upper == float('inf'):
                upper = self.env.unwrapped.spec.kwargs.get(
                    'thresholds', [2.4, 2.4, 0.418, 0.418])[i]
            if lower == float('-inf'):
                lower = -upper
            
            scaling = (self.n_buckets[i] - 1) / (upper - lower)
            new_obs = int(np.floor(scaling * (observation[i] - lower)))
            new_obs = min(self.n_buckets[i] - 1, max(0, new_obs))
            discretized.append(new_obs)
        return tuple(discretized)
    
    def create_q_table(self):
        """
        Create and initialize Q-table with zeros
        
        Returns:
            Initialized Q-table
        """
        q_table_shape = self.n_buckets + (self.env.action_space.n,)
        return np.zeros(q_table_shape)
    
    def select_action(self, state, epsilon):
        """
        Select an action using epsilon-greedy policy
        
        Args:
            state: Current discretized state
            epsilon: Exploration rate
            
        Returns:
            Selected action
        """
        if np.random.random() < epsilon:
            # Explore: select a random action
            return self.env.action_space.sample()
        else:
            # Exploit: select action with the highest q-value
            return np.argmax(self.q_table[state])
    
    def update_q_value(self, state, action, reward, next_state):
        """
        Update Q-value for a state-action pair
        
        Args:
            state: Current discretized state
            action: Action taken
            reward: Reward received
            next_state: Next discretized state
        """
        next_max_q = np.max(self.q_table[next_state])
        current_q = self.q_table[state][action]
        new_q = current_q + self.alpha * (reward + self.gamma * next_max_q - current_q)
        self.q_table[state][action] = new_q
    
    def train(self):
        """
        Train the Q-learning agent
        
        Returns:
            Dictionary containing training metrics
        """
        self.q_table = self.create_q_table()  # Reset Q-table
        
        metrics = {
            'total_reward': [],
            'episode_length': [],
            'batch_reward_mean': [],
            'batch_reward_std': []
        }
        
        for episode in range(self.n_episodes):
            # Reduce epsilon (exploration rate) over time
            epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                      np.exp(-episode / self.epsilon_decay)
            
            # Reset environment
            observation, _ = self.env.reset()
            state = self.discretize_state(observation)
            
            done = False
            total_reward = 0
            episode_length = 0
            
            # One episode of training
            while not done:
                action = self.select_action(state, epsilon)
                
                # Take action and observe the result
                next_observation, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                next_state = self.discretize_state(next_observation)
                
                self.update_q_value(state, action, reward, next_state)
                
                # Move to the next state
                state = next_state
                total_reward += reward
                episode_length += 1
            
            metrics['total_reward'].append(total_reward)
            metrics['episode_length'].append(episode_length)
            
            # Calculate batch statistics every 50 episodes
            if episode % 50 == 0 and episode > 0:
                batch_rewards = metrics['total_reward'][max(0, episode-50):episode]
                mean_reward = np.mean(batch_rewards).round(0)
                std_reward = np.std(batch_rewards).round(2)
                
                metrics['batch_reward_mean'].append(mean_reward)
                metrics['batch_reward_std'].append(std_reward)
                
                print(f"Episode {episode}, Average reward: {mean_reward} "
                      f"with std {std_reward}, Epsilon: {epsilon:.4f}")
        
        # Store metrics for analysis
        self.metrics = metrics
        return metrics
    
    def test(self, n_episodes=20, render=False):
        """
        Test the trained Q-learning agent over several episodes
        
        Args:
            n_episodes: Number of test episodes to run
            render: Whether to render the environment
            
        Returns:
            Average episode length across all test episodes
        """
        env_test = gym.make('CartPole-v1', render_mode='human' if render else None)
        episode_lengths = []
        
        for episode in range(n_episodes):
            # Reset the environment
            observation, _ = env_test.reset()
            state = self.discretize_state(observation)
            
            done = False
            episode_length = 0
            
            # Run one episode
            while not done:
                # Always select the best action (no exploration)
                action = np.argmax(self.q_table[state])
                
                # Take action
                next_observation, _, terminated, truncated, _ = env_test.step(action)
                done = terminated or truncated
                
                # Update state and counter
                state = self.discretize_state(next_observation)
                episode_length += 1
            
            episode_lengths.append(episode_length)
            print(f"Test Episode {episode+1}/{n_episodes}, Length: {episode_length}")
        
        avg_length = sum(episode_lengths) / len(episode_lengths)
        print(f"Average episode length: {avg_length:.2f}")
        
        env_test.close()
        return avg_length
    
    def get_optimization_metrics(self):
        """
        Calculate optimization metrics based on training results
        
        Returns:
            Tuple of (average reward, average std)
        """
        avg_reward = np.mean(self.metrics['batch_reward_mean']).round(0)
        avg_std = np.mean(self.metrics['batch_reward_std']).round(2)
        return avg_reward, avg_std
    
    def close(self):
        """Clean up resources"""
        self.env.close()


def objective(trial):
    """
    Optuna objective function for hyperparameter optimization
    
    Args:
        trial: Optuna trial object
        
    Returns:
        Optimization score (higher is better)
    """
    # Define hyperparameter search space
    config = {
        # State discretization parameters
        'cart_position_bins': trial.suggest_int('cart_position_bins', 5, 20),
        'cart_velocity_bins': trial.suggest_int('cart_velocity_bins', 5, 20),
        'pole_angle_bins': trial.suggest_int('pole_angle_bins', 5, 20),
        'pole_velocity_bins': trial.suggest_int('pole_velocity_bins', 5, 20),
        
        # State bounds
        'cart_velocity_bounds_min': trial.suggest_float('cart_velocity_bounds_min', -2.0, -0.5),
        'cart_velocity_bounds_max': trial.suggest_float('cart_velocity_bounds_max', 0.5, 2.0),
        'pole_velocity_bounds_min': trial.suggest_float('pole_velocity_bounds_min', -3.0, -1.0),
        'pole_velocity_bounds_max': trial.suggest_float('pole_velocity_bounds_max', 1.0, 3.0),
        
        # Training parameters
        'n_episodes': trial.suggest_int('n_episodes', 5000, 20000, step=5000),
        'alpha': trial.suggest_float('alpha', 0.01, 0.2, log=True),
        'gamma': trial.suggest_float('gamma', 0.9, 0.999),
        'epsilon_start': 1.0,  # Fixed parameters
        'epsilon_end': 0.001,
        'epsilon_decay': trial.suggest_int('epsilon_decay', 1000, 10000),
    }
    
    # Create and train agent
    agent = CartPoleAgent(config)
    agent.train()
    
    # Get optimization metrics
    avg_reward, avg_std = agent.get_optimization_metrics()
    
    # Test the agent
    avg_test_length = agent.test(n_episodes=10, render=False)
    
    # Clean up
    agent.close()
    
    # Log metrics to MLflow
    mlflow.log_metric('average_train_reward', float(avg_reward))
    mlflow.log_metric('average_train_std', float(avg_std))
    mlflow.log_metric('average_test_length', float(avg_test_length))
    
    # Log all config parameters
    for param_name, param_value in config.items():
        mlflow.log_param(param_name, param_value)
    mlflow.end_run()
    # Optimization goal: maximize reward, minimize std deviation
    # We want to penalize high standard deviation, so we'll subtract it from reward
    # You can adjust the weight of std_penalty to balance these objectives
    std_penalty_weight = 0.5
    optimization_score = avg_reward - (std_penalty_weight * avg_std)
    
    return optimization_score


def run_optimization(n_trials=50):
    """
    Run Optuna optimization with MLflow tracking
    
    Args:
        n_trials: Number of optimization trials to run
    """
    # Set up MLflow experiment
    mlflow.set_experiment('cart_pole_optuna')

    # Create MLflow callback for Optuna
    mlflow_callback = MLflowCallback(
        tracking_uri=mlflow.get_tracking_uri(),
        metric_name="optimization_score"
    )
    
    # Create and run Optuna study
    study = optuna.create_study(
        direction="maximize",
        study_name="cart_pole_optimization",
        pruner=optuna.pruners.MedianPruner(),
    )
    
    study.optimize(objective, n_trials=n_trials, callbacks=[mlflow_callback])
    
    # Print optimization results
    print("Number of finished trials:", len(study.trials))
    print("Best trial:")
    best_trial = study.best_trial
    print(f"  Value: {best_trial.value}")
    print("  Params:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")
    
    return study.best_params


def train_with_best_params(best_params):
    """
    Train a final model with the best parameters found by Optuna
    
    Args:
        best_params: Dictionary of best parameters
        
    Returns:
        Trained CartPoleAgent
    """
    # Set up MLflow run for final model
    with mlflow.start_run(run_name="cart_pole_best_model"):
        # Create agent with best parameters
        agent = CartPoleAgent(best_params)
        
        # Train agent
        agent.train()
        
        # Test agent
        avg_test_length = agent.test(n_episodes=20, render=False)
        
        # Get final metrics
        avg_reward, avg_std = agent.get_optimization_metrics()
        
        # Log metrics and parameters
        mlflow.log_metric('average_train_reward', float(avg_reward))
        mlflow.log_metric('average_train_std', float(avg_std))
        mlflow.log_metric('average_test_length', float(avg_test_length))
        
        # Log all parameters
        for param_name, param_value in best_params.items():
            mlflow.log_param(param_name, param_value)
    
    return agent


if __name__ == "__main__":
    # Run Optuna optimization
    best_params = run_optimization(n_trials=200)
    
    # Train final model with best parameters
    final_agent = train_with_best_params(best_params)
    
    # Test with rendering to visualize performance
    final_agent.test(n_episodes=5, render=False)
    
    # Close environment
    final_agent.close()