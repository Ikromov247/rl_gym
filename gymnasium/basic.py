import gymnasium as gym

# create the environment
env = gym.make(
    id="LunarLander-v3", # Lunar lander  
    render_mode="human" # renders a gui representation of the environment and agent actions
)
# reset the env to get the first observation
# we will use this observation to take our first action
observation, info = env.reset()
# info contains additional info in a dictionary, complementing observation. 

episode_over = False

while not episode_over:
    action = env.action_space.sample() # take a random action 
    # we can also select an action based on policy, if we have one
    # e.g.: action = policy(observation)

    # we provide our action to the environment, which 'executes' that action, and 
    # returns the new observation, reward, termination/truncation flag and additional info
    observation, reward, terminated, truncated, info = env.step(action) 

    # in-depth explanation for terminated/truncated flags: https://farama.org/Gymnasium-Terminated-Truncated-Step-API
    # tldr; 
    episode_over = terminated or truncated

env.close()