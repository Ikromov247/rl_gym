
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
        At first I thought it was initial bias, but then I remembered sample-average does not have that. 
        Idk the answer

**3. The 10-armed Testbed**  
​ a.  In the 10-armed testbed experiment, what metrics (plots) are used to compare methods? Why are those metrics chosen?  
        The methods are compared using their performance on two metrics: average reward and percentage of optimal action chosen 
        at each timestamp. This essentially represents the methods' learning rate over time, showing how quickly they learn
        and what is their plateu on these metrics.  

​ b.  How does the randomness in the testbed’s true action values affect your interpretation of the results?
        A method's performance on the testbed reflects its ability to arrive at the 'mean' value of each action, 
        which is the best estimate it can ever have given randomness to achieve highest reward over time. 
        Still, sometimes methods that do not explore properly or stop exploring too early might 
        struggle if the sample of rewards they observed had a different distribution compared to 
        the correct distribution. Therefore, the textbook instructs us to conduct 2000 runs and average the 
        method's performance for each timestamp to reflect its average accuracy. 

**4. Incremental Implementation**  
​ a.  Derive the incremental update formula for the sample-average estimate \( Q_{n+1} = Q_n + \tfrac{1}{n}[R_n - Q_n]\). 
        The formula for calculating the sample mean of all rewards is:
                Q_n = sum(rewards_1 + rewards_n) / n
        This requires storing all past rewards to calculate the new estimate. However, a new reward estimate can be simplified as:
                Q_n = (previous_estimate * (n-1) + current_reward) / n -> previous_estimate * n / n - previous_estimate / n + current_reward / n 
                        -> previous_estimate + (current_reward - previous_estimate) / n
                which only requires storing the previous estimate. 
        

​ b.  Explain why this implementation is more memory-efficient than storing all past rewards.
        the formula 
                 \( Q_{n+1} = Q_n + \tfrac{1}{n}[R_n - Q_n]\). 
        only requires us to save the previous sample average estimate instead of all past rewards. 

**5. Tracking a Nonstationary Problem**  
​ a.  Why does a constant-step-size rule \( Q_{n+1} = Q_n + \alpha [R_n - Q_n] \) track nonstationarities better than the sample-average rule?  
        We can rewrite the constant-step-size update formula to demonstrate how it gives more weight to recent rewards:
        Q_{n+1} = Q_n + alpha * [R_n - Q_n]
        Q_{n+1} = Q_n + alpha * R_n - alpha * Q_n # multiply both elements in brackets by alpha
        Q_{n+1} = (1-alpha) * Q_n + alpha * R_n # move Q_n, the common multiplier, outside brackets of 1 and -alpha
        Q_{n+1} = (1-alpha) * Q_n + alpha * R_n # express Q_n in terms of Q_{n-1}, R_{n-1}, essentially turning the formula into recursion
        Q_{n+1} = (1-alpha) * ((1-alpha) * Q_{n-1} + alpha * R_{n-1}) + alpha * R_n # multiply 1-alpha by elements inside the brackets
        Q_{n+1} = (1-alpha) ^ 2 * Q_{n-1} + alpha * (1-alpha) * R_{n-1} + alpha * R_n # we can apply the same recursion to express Q_{n-1}
        Q_{n+1} = (1-alpha) ^ 2 * ((1-alpha) * Q_{n-2} + alpha * R_{n-2}) + alpha * (1-alpha) * R_{n-1} + alpha * R_n
        Q_{n+1} = (1-alpha) ^ 3 * Q_{n-2} + alpha * (1-alpha)^2 * R_{n-2} + alpha * (1-alpha) * R_{n-1} + alpha * R_n # we can see a pattern. 
        Q_{n+1} = (1-alpha) ^ k * Q_{n-k} + alpha * SUM {i=0, i=k, i++} [ R_{n-i} * (1-alpha) ^ i ]
        where k is how far behind the element is. the higher k means the element is much earlier. 
        since we the depth of recursion only goes up to Q_0, we know n-k is eventually 0. which means n = k
        Q_{n+1} = (1-alpha) ^ n * Q_0 + alpha * SUM {i=0, i=n, i++} [ R_{n-i} * (1-alpha) ^ i ]
        here, pay attention to the first expression. since 0 < alpha < 1, 1-alpha is  higher than 0 and lower than 1. 
        raising it to the power of n diminishes its value exponentially, thus reducing the weight of Q_0 to 0 as n -> inf.
        similar can be said of other elements < n. the further away an element is from n (most recent value),
        the smaller its weights become at an exponential rate. 
        If reward distributions change, our value estimate adapts quickly because it 'forgets' earlier estimates. 
        It can be shown that the number of most recent estimates that have impact on the current estimate is ~1/alpha. 
        e.g alpha=0.1, 1/alpha=10. The 10 recent estimates have most impact on Q estimation
        

​ b.  Show that the constant-α estimate is a weighted average of past rewards with exponentially decaying weights.
        I showed it in the answer to the previous question. 

**6. Optimistic Initial Values**  
​ a.  How do optimistic initial action-value estimates encourage exploration?  
        When the initial action-value estimates are much higher than true rewards, 
        the model will try each action at least once until the action-value estimate is 
        close to true reward values. This is because, when the agent tries an action at first, 
        it receives a low reward, which 'disapppoints' it, reduces its estimate of that action's value,
        and forces to try the next action. This repeats until all estimates are equal or lower to the 
        true reward mean.

​ b.  What pathological behavior can arise if the problem is nonstationary or very noisy?
        Optimistic intial values only encourage exploration early on, since the estimates very quickly
        converge to the observed mean rewards. However, if the problem is nonstationary, the estimates 
        do not update to the changing reward distribution, leading to unopyimal action selection. 
        If the reward signal is very noisy, the estimates do not converge to an accurate mean, instead oscillating and
        varying the agent's behavior randomly (not sure).

**7. Upper-Confidence-Bound (UCB) Action Selection**  
​ a.  Derive the UCB action-selection rule  
        \[
        A_t = \arg\max_a \Bigl[ Q_t(a) + c\,\sqrt{\frac{\ln t}{N_t(a)}}\Bigr].
        \]  
        
​ b.  Explain the effect of the parameter \(c\) and the \(\ln t\) term on exploration.
        c regulates how much weight we give to the uncertainty term. The higher c magnifies our uncertainty,
        leading us to explore an action even if its reward estimate is lower than sum of other actions's estimate + their uncertainty.
        ln(t) is a natural logarithm of the current step. it is unbounded, which means it never vanishes, but its growth rate reduces exponentially,
        which means at later timestamps its value changes very slowly. Since it is in the numerator, its square root is directly correlated  
        with the uncertainty term. Over time, the uncertainty term's growth rate diminishes exponentially. 
        
        How I think of it is that if an action is selected once, its uncertainty will decrease (since action count is in denominator). 
        since ln(t) is in the numerator, for uncertainty to go up, many timestamps need to pass, especially later in the run.
        Thus, uncertainty is high at the start which leads to more exploration, but grows very slowly wrt timestamp (ln and sqrt) 
        and reduces at a higher rate each time an action is selected.


**8. Gradient Bandit Algorithms**  
​ a.  Define the preference parameter \(H_t(a)\) and show how it is updated in the gradient bandit algorithm.  
        The preference parameter in Gradient bandits 
        It is used to calculate probability of each action being selected using softmax function. 
        Pr{A_t=a}=(e^H_t(a)) / (sum{i=1, i=k, i++} [e^H_t(i)]) = pi(a)
        where k is the number of arms. the preference value is updated as follows:
        `H_t(a) = H_t(a) + alpha * (R_n - R_baseline) * (1_predicate{A_t=a} - pi(a))`.
        In this formula, 1_predicate is 1 if a is the selected action, 0 if a is not the selected action. 
        alpha is a hyperparameter that regulates how much weight is assigned to the error term.
        pi(a) is the probability that this action is selected. 
        when A_t=a, updating the preference for selected action:
                if pi(a) is high (i.e. this action is already heavily favored), 1-pi(a) will be low, leading to a small update.
                if pi(a) is low (i.e. this action is not favored but still got chosen), 1-pi(a) will be high, leading to a larger update. 
        If the reward is higher than baseline, the update is positive, otherwise negative. 
        `H_t(a) = H_t(a) - alpha * (R_n - R_baseline) * pi(a)`.
        when updating preference for other actions. 
        If the reward is higher than baseline, the update is *negative*, otherwise positive.
        Thus, the update direction is regulated by the term (R_n - R_baseline). 
        The selected action is always updated in the opposite direction to non-selected actions. 
        

​ b.  Explain why subtracting a baseline (e.g.\, the average reward) reduces variance in the updates.
        setting R_baseline as the sample-average of observed rewards simply means we only
        update the preference values if the observed reward is much higher or lower than the mean of
        observed rewards. This may reduce variance.
        I dont know the answer.  

**9. Softmax (Gibbs) Action Selection**  
​ a.  Write down the softmax (Gibbs) distribution over actions in terms of preferences \(H_t(a)\).  
        Pr{A_t=a}=(e^H_t(a)) / (sum{i=1, i=k, i++} [e^H_t(i)]) = pi(a)

​ b.  Compare softmax action selection to ε-greedy—what are the pros and cons of each?
        ?idk

**10. Associative Search (Contextual Bandits)**  
​ a.  What distinguishes an associative-search (contextual) bandit from a plain _k_-armed bandit?  
        Associative search presents several different states, which may have a different reward 
        distribution for the same actions. in a plain k-armed bandit, we only learn which action is 
        best, whereas in contextual bandit, we learn which action is best in each state. 

​ b.  Give a concrete example of how you would include a one-dimensional context \(x\) in your value estimates.
        A simple way to do this is to create a 2-dimensional matrix, called q-table, where each row represents a state, 
        and each column represents an action value for each state. Thus, its shape is (x, a) where x is the number of possible states,
        a is the number of arms. 

**11. Putting It All Together**  
​ a.  Compare optimistic initial values, UCB, and gradient bandit methods in terms of exploration efficiency, parameter sensitivity, and computational cost.  
        - Optimistic initial values with greedy selection is the simplest in terms of implementation, and fairly good at exploration given
        stationarity. It is not very sensitive to its parameter (initial value) as long as its large enough. Even if the initial value is too large, it will 
        quickly converge close to the true reward mean (i think) at log_2(initial_value) time. However, this method fails to converge if the reward distribution is non-stationary.
        I am not sure about the effect of noise though. What is noise in this context (referring to a previous question)?
        - UCB is an improvement over epsilon greedy in that it does not explore randomly but only selects unoptimal actions if their uncertainty value is very high.
         Its computational cost is slightly higher than greedy since we calculate logarithms and square roots of values. Also, we need to store action count array of size O(a) in memory.
         UCB is very sensitive to parameters, since when the exploration rate (c) is too high, it might select unoptimal actions too often. 
         when 0 < c < 1, ucb is fairly stable but slower to converge. 
        - Gradient bandit is most expensive in terms of computational cost since we need to update preference values and calculate probability for each action at each step. 
         Besides, we need to update the reward baseline. 
         I am not sure if it performs well in non-stationary problems either, since R_baseline's sample-average update rule is not efficient in such cases. An incorrect
         R_baseline value might hurt learning. I would not prefer it over a simple epsilon-greedy with constant step-size update rule. 


​ b.  In which situations would you prefer one method over the others?
        For most problems, I prefer epsilon-greedy with constant step-size update rule for its simplicity, ease of implementation and interpretation.
        UCB is a close second, but I don't think it performs just as well in non-stationary problems. It might excel in stationary problems.
        Gradient bandit is computationally expensive, harder to optimize and interpret. I would not choose it unless it leads to significantly higher reward over time.  

**12. Chapter Summary**  
​ In your own words, list the three most important design principles you’ll carry forward from this chapter into more general reinforcement-learning algorithms.
        1. I will not choose action-value estimation methods that behave differently in the beginning of the run or treat the start of the run as special.
           (optimistic estimate with sample average update, and greedy selection, ucb policies) unless the problem is stationary. 
           But since most reinforcement problems are non-stationary, I probably will not use these methods often. 
        2. The action value update rule should be robust to noise, since reward often has stochastic noise. 
        3. Direct exploration using uncertainty term or gradients.