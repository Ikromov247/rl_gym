
**1. A _k_-armed Bandit Problem**  
​ a.  Formally define the _k_-armed bandit problem.  
        k-armed bandit problem is a decision-making problem with k different 'arms' (actions to take), each with
        rewards of stationary distribution, which is unknown. The agent taking the actions has no context but
        the immediate reward for the arm it just selected.

​ b.  What is the performance objective, and how is “regret” defined in this setting?
        The objective is to maximize reward over time by selecting arms with the highest reward. However, since we do not 
        know the true value of rewards, we need to develop an accurate estimate of rewards for each arm. 
        If the estimate of rewards is innacurate, the agent will select unoptimal actions which reduces potential reward. 
        In order to update our estimate of rewards, we need to explore by trying new arms to update our knowledge on their reward, 
        whereas to maximize potential reward we need to exploit by select an arm with the highest reward according to our estimate. 
        Since an agent can only either explore or exploit, balancing these  decisions is the core problem of reinforcement learning.

        Regret is the difference between earned reward over time with maximum reward possible if the true 
        reward distribution was known:
                regret = actual_reward - true_reward
        Minimizing regret is a similar objective to maximizing reward.

**2. Action-value Methods**  
​ a.  Describe the ε-greedy action-selection method and explain how it balances exploration vs. exploitation.  
        When an action-selection method always selects the action with highest reward, it is called 'greedy'. 
        This results in little exploration, and thus biased estimates of action rewards. 
        To counter this problem, we can explore some times, while selecting the best action most of the time. 
        This method is called epsilon-greedy action-selection, where epsilon is the probability of selecting
        a random action (exploration). 

​ b.  What bias does the sample-average method introduce when paired with ε-greedy, and why?
        pass

**3. The 10-armed Testbed**  
​ a.  In the 10-armed testbed experiment, what metrics (plots) are used to compare methods? Why are those metrics chosen?  
​ b.  How does the randomness in the testbed’s true action values affect your interpretation of the results?

**4. Incremental Implementation**  
​ a.  Derive the incremental update formula for the sample-average estimate \( Q_{n+1} = Q_n + \tfrac{1}{n}[R_n - Q_n]\).  
​ b.  Explain why this implementation is more memory-efficient than storing all past rewards.

**5. Tracking a Nonstationary Problem**  
​ a.  Why does a constant-step-size rule \( Q_{n+1} = Q_n + \alpha [R_n - Q_n] \) track nonstationarities better than the sample-average rule?  
​ b.  Show that the constant-α estimate is a weighted average of past rewards with exponentially decaying weights.

**6. Optimistic Initial Values**  
​ a.  How do optimistic initial action-value estimates encourage exploration?  
​ b.  What pathological behavior can arise if the problem is nonstationary or very noisy?

**7. Upper-Confidence-Bound (UCB) Action Selection**  
​ a.  Derive the UCB action-selection rule  
\[
A_t = \arg\max_a \Bigl[ Q_t(a) + c\,\sqrt{\frac{\ln t}{N_t(a)}}\Bigr].
\]  
​ b.  Explain the effect of the parameter \(c\) and the \(\ln t\) term on exploration.

**8. Gradient Bandit Algorithms**  
​ a.  Define the preference parameter \(H_t(a)\) and show how it is updated in the gradient bandit algorithm.  
​ b.  Explain why subtracting a baseline (e.g.\, the average reward) reduces variance in the updates.

**9. Softmax (Gibbs) Action Selection**  
​ a.  Write down the softmax (Gibbs) distribution over actions in terms of preferences \(H_t(a)\).  
​ b.  Compare softmax action selection to ε-greedy—what are the pros and cons of each?

**10. Associative Search (Contextual Bandits)**  
​ a.  What distinguishes an associative-search (contextual) bandit from a plain _k_-armed bandit?  
​ b.  Give a concrete example of how you would include a one-dimensional context \(x\) in your value estimates.

**11. Putting It All Together**  
​ a.  Compare optimistic initial values, UCB, and gradient bandit methods in terms of exploration efficiency, parameter sensitivity, and computational cost.  
​ b.  In which situations would you prefer one method over the others?

**12. Chapter Summary**  
​ In your own words, list the three most important design principles you’ll carry forward from this chapter into more general reinforcement-learning algorithms.
