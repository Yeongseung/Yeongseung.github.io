---
date: '2026-01-20T16:14:29+09:00'
draft: false
title: 'Mastering CartPole'

tags: ["CartPole", "Reinforcement Learning"]
---
(Writing)
# intro

If you are interested in Reinforcement Learning, you've probably heard about [CartPole](https://gymnasium.farama.org/environments/classic_control/cart_pole/), which is apparently "Hello world" in the RL field. In my opinion, this is because, first, the CartPole game is relatively easy to understand in terms of how it works and, second, the game ends quickly, so you can see the results fast.

<figure class="figure-center">
  <img src="/posts/Mastering_CartPole/cartpole.png" width="300">
  <figcaption>Figure 1. CartPole image</figcaption>
</figure>

Actually, come to think of it, I've conducted expriences in the CartPole environment, but not in a organized way, leading me forget most of it. So, here my motive is to put together many trials I've done and, hopefully, to get insights from it.

# Experiments

## 1. Starts from Q-Learning
Intuitively, Q-learning comes from Temporal-Difference(TD) or,as I believe I could say, Q-learning is a part of TD. TD focuses on updating the value function, not the policy itself. we can consider $V(S_t)$ as the prediction $\hat{y}$ and $R_{t+1}+\gamma V(S_{t+1})$ as the target $y$.

<div>
$$
\begin{aligned}
V(S_t) \leftarrow V(S_t) + \alpha (R_{t+1} + \gamma V(S_{t+1}) - V(S_t)) \tag{1}
\end{aligned}
$$
</div>

This analogy to supervised learning makes the formula(1) simple to codify when we use Neural Network to approximate value function.

```python
#torch
env = gym.make("CartPole-v1", render_mode="none")
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

current_value = model(state) # V(S_t)

# (select a action to take somehow)
next_state, reward, ... = env.step(action)

next_value = model(next_state)
target_value = reward + (gamma* next_value)

criterion = nn.MSELoss()
loss = criterion(current_value, target_value)
```

However, the `(select a action to take somehow)` part is a big problem. $V(s)$ itself doesn't contain information about what action I should take. If the environment was so simple and deterministic such as Chess or Go, then we could simulate all actions from $S_t$ and calculate all $V(S_{t+1})$ without really taking those actions. All possible $V(S_{t+1})$s in hand, finally we can select a action by choosing the action leading to a state having the best value. 

### SARSA
That's where the action-value function Q is effective. $V(S_t) = max_{a}Q(S_t,a)$. Below is the SARSA algorithm, which leverages the action-value function Q. As you will see, now we can choose next action without really taking it or simulating it.

<details>
<summary><b>Click to see the SARSA torch code</b></summary>

```python
model = qnetwork(4,2) # state consists of four scalars. 
env = gym.make("CartPole-v1", render_mode="none")
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

for episode in range(3000): # 3000 episodes
    obs, info = env.reset()
    # Example output: [ 0.01234567 -0.00987654  0.02345678  0.01456789]
    # [cart_position, cart_velocity, pole_angle, pole_angular_velocity]

    game_over = False
    total_reward = 0

    action_space = [0,1] # 0: left, 1: right
    
    # First action of a episode should be selected here.
    state_tensor = torch.tensor(obs, dtype=torch.float32)
    q_values = model(state_tensor)
    if np.random.random() < 0.1: # epsilon = 0.1
        act = np.random.choice(action_space)
    else:
        act = torch.argmax(q_values).item()

    while not game_over:
        state_tensor = torch.tensor(obs, dtype=torch.float32)
        q_values = model(state_tensor)
        # SARSA: act is already selected. (This is the key point of SARSA.)
        
        # current_q is where the backpropagation starts.
        current_q = q_values[act]

        # step the environment
        next_obs, reward, terminated, truncated, info = env.step(act)
        total_reward += reward

        # SARSA: calculate the target using next_act
        with torch.no_grad(): # to block the gradient
            next_state_tensor = torch.tensor(next_obs, dtype=torch.float32)
            if np.random.random() < 0.1:
                next_act = np.random.choice(action_space)
            else:
                next_act = torch.argmax(model(next_state_tensor)).item()
            
            next_q = model(next_state_tensor)[next_act]
            target_q = reward + (0.99 * next_q * (1 - terminated)) 
            # no next reward when terminated
            # But, truncated is okay. because we can still get the next reward.

        # calculate the loss and update the model
        loss = criterion(current_q, target_q)
        #target_q here is like the label for the current_q.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # SARSA: update act not only obs.
        obs = next_obs
        act = next_act  # This is the key of SARSA.
```
</details>

In my opinion, the main point of SARSA is to select the next action and target q-value using epsilon-greedy. (see the `with torch.no_grad():` part) Let's check the SARSA's performance.

<figure class="figure-center">
  <img src="/posts/Mastering_CartPole/SARSA_Q.png" width="700">
  <figcaption>Figure 3. Q-value based on fixed states</figcaption>
</figure>

<figure class="figure-center">
  <img src="/posts/Mastering_CartPole/SARSA_reward_graph.png" width="700">
  <figcaption>Figure 2. Episode-Total reward(Left), Episode-TD Error(right)</figcaption>
</figure>

SARSA reached the maximum reward of 500, and the moving average showed an increasing trend till some point. However, it fails to stay at the peak. What could be the reason for these periodic drops in total reward?
> 1. FIXED EPSILON?

I fixed epsilon as 0.1 meaning that approximately every 10 steps, a action is selected randomly and CartPole is very sensitive game. But even so, I believe that if model were trained well, then It should have been robust enough to handle the 10% of randomness. 

> 2. NN MODEL?

```python
class qnetwork(torch.nn.Module):
    def __init__(self, state_size:4, action_size:2):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, action_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```
Was my model not large enough or not structured well to memorize all the past information? ...could be. Even though the training data is correlated, if the model was large and has many state-of-the-art features, it could memorize every past experience. 

> 3. FORGETTING?

If the problem mainly caused from forgetting previous expriences, there could be several ways to mitigate this. [(Mnih et al., 2013)](https://arxiv.org/abs/1312.5602) says that leveraging **Experience replay** is effective to reduce data correlation and to smooth the training distribution. Intuitively, I believe that you can easily see why Experience replay can reduce data correlation because experience replay allows us to randomly choose samples from several saved episodes, not consecutively. Then what does it mean that Experience Replay "smooths the training distribution?"

<figure class="figure-center">
  <img src="/posts/Mastering_CartPole/distributions.png" width="700">
  <figcaption>Figure 3. Example distributions</figcaption>
</figure>

In CartPole there are four scalar in state, but Let's think about only one scalar, **Pole Angle**. If training data is dominated by specific scenarios, such as left leaning angles (Curve 1), the agent overfits and fail to handle opposite situations like right tilts. As the agent masters the task, the data distribution squezees into a narrow curve around zero (Curve 2), causing the agent to forget recovery strategies for large deviations. (This is similar to a problem of imitation learning, which learns from expert's behaviors.)

To build a robust model, the agent must learn from a comprehensive distribution that covers the entire state space. Experience Replay kind of addresses this by randomly sampling from a diverse history of episodes. If the Experience buffer(storage) successfully saves diverse scenarios, then it could reconstruct a broad distribution that encompasses all scenarios.

### Q-learning

Okay, I decided to use Experience Replay, but SARSA is incompatible with Experience Replay because it's an on-policy algorithm. So, it's time to move on to Q-learning since Q-learning is an off-policy algorithm, which means we can leverage Experience Replay, and it's similar to SARSA. 

However, Firstly, Let's look at Q-learning **without** Experience Replay. The main difference between Q-learning and SARSA is that when calculating the target Q, we don't consider random actions(like in epsilon-greedy), even if the actual next action might be random. we just do exploitation. Intuitively, it makes sense to define the value of the current state based on the best possible result achievable by ideal actions from that state.

```python
        # calculate the target
        with torch.no_grad():
            next_state_tensor = torch.tensor(next_obs, dtype=torch.float32)
            next_q_max = torch.max(model(next_state_tensor))
            target_q = reward + (0.99 * next_q_max * (1 - terminated)) 

        loss = criterion(current_q, target_q)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        episode_losses.append(loss.item()) 

        obs = next_obs
```
Let's check the results.

<figure class="figure-center">
  <img src="/posts/Mastering_CartPole/Qlearning_fixed_states.png" width="700">
  <figcaption>Figure 3. Example distributions</figcaption>
</figure>

<figure class="figure-center">
  <img src="/posts/Mastering_CartPole/Qlearning_episode_total_reward.png" width="700">
  <figcaption>Figure 3. Example distributions</figcaption>
</figure>

The results seems like not that different from results from SARSA.

### Q-learning with Experience Replay
Okay, I set the capacity of the replay buffer to 10,000, the batch size to 32, and the update (training) frequency to every four steps. Let's see the results.

<figure class="figure-center">
  <img src="/posts/Mastering_CartPole/Qlearning_replay_fixed_states.png" width="700">
  <figcaption>Figure 3. Example distributions</figcaption>
</figure>

<figure class="figure-center">
  <img src="/posts/Mastering_CartPole/Qlearning_replay_total_reward.png" width="700">
  <figcaption>Figure 3. Example distributions</figcaption>
</figure>

Surprisingly, many things changed. **First**, the episode-total reward plot became more stable (compared to before), even though periodic drops still occurred. This makes sense because gathering experiences from various episodes and using a batch of 32 leads to longer periods of stability(staying at the peak). However, this prolonged stability causes the replay buffer to be filled with only stable data, which narrows the buffer's distribution. This is likely why we still see periodic drops. 

**Second**, Q-values also stabilized relatively well, but there was sudden spike. Let's focus on episodes between 1200 and 1400. Total rewards dropped during that period, while the Q-values of the five fixed states skyrocketed. **Why?** Most likely, Staying at the peak caused the buffer's distribution to become too narrow, which in turn narrowed the Q-network's coverage. At that point, when the Q-network encountered an unfamilar state, such as leaning right, edgeing left or leaning left, it incorrectly outputted a very high Q-value. Consequently, the Q-network learned that leaning right or left is desirable state. Ultimately, this caused the total reward drop.(It's my theory..)

So, the problem is stil forgetting, which, I believe, cannot be solved by target network, Double DQN, or Dueling DQN. Intuitively, even the episodes become full of successful experiences, buffer still has to keep unsuccessful experiences and how to recover from that.

### Dual Buffer
Dual buffer is the idea that seperating successful experiences and unsuccessful experience, and sample from them by fixed ratio like 7:3. I think this could be direct remedy of periodic drops. I set the capacities of successful buffer and unsuccessful buffer as 5000, and same batch size 32. Let's see the results.

<figure class="figure-center">
  <img src="/posts/Mastering_CartPole/dual_buffer_fixed_states.png" width="700">
  <figcaption>Figure 3. Example distributions</figcaption>
</figure>

<figure class="figure-center">
  <img src="/posts/Mastering_CartPole/dual_buffer_rewards.png" width="700">
  <figcaption>Figure 3. Example distributions</figcaption>
</figure>

Logically, I believe adding a dual buffer was the right approach, but it hasn't solved the periodic drop problem so far. In the meantime, the Q-values became too high and the TD error skyrocketed. I think we should solve this problem first.

### Double DQN
The idea of separating the main Q-network and the target Q-network is a well-known solution for Q-value overestimation. The reason I didn't use this idea earlier was that I thought the dual buffer could solve periodic drop problem even without seperating Q-network. Anyway, let's see the results.

<figure class="figure-center">
  <img src="/posts/Mastering_CartPole/double_soft_Q_fixed_states.png" width="700">
  <figcaption>Figure 3. Example distributions</figcaption>
</figure>

<figure class="figure-center">
  <img src="/posts/Mastering_CartPole/double_soft_Q_reward.png" width="700">
  <figcaption>Figure 3. Example distributions</figcaption>
</figure>

The results improved incredibly. Q-values stabilized, which caused the TD error to stabilize as well. Periodic drops are still there, but they have been somewhat mitigated. 

**So, why did it improve?** By separating the main Q-network and the target Q-network, and updating the target Q-network slowly toward the main Q-network, we can mitigate the sudden overestimation of the main Q-network.

For example, even when the buffer is filled with 500-reward experiences, if the main Q-network encounters a forgotten situation, like leaning right, it might overestimate the Q-value. However, because the target Q-network is updated slowly, this overestimation can be mitigated. Specifically, with a soft update factor $\tau=0.01$, the target network only incorporates 1% of the main network's weights at each step, ensuring the target values remain stable.

<details>
<summary><b>Click to see Double DQN without Dual buffer case</b></summary>

<figure class="figure-center">
  <img src="/posts/Mastering_CartPole/nodual_double_Q_fixed_states.png" width="700">
  <figcaption>Figure 3. Example distributions</figcaption>
</figure>

<figure class="figure-center">
  <img src="/posts/Mastering_CartPole/nodual_double_Q_reward.png" width="700">
  <figcaption>Figure 3. Example distributions</figcaption>
</figure>

</details>

### Back to the buffer
Okay, Let's back to the buffer modification. I noticed that the Q-values tended to drift upward as the steps progressed, which was especially obvious before implementing DOuble DQN. **My theory here is that** because I defined the "unsuccessful buffer" as simply as the storage storing every state of episode where the total reward was less than 500, the buffer eventually became saturated with "long" unsuccessful experiences.

**Why could it be a problem?** "Long" unsuccessful experiences (e.g., an episode lasting 400 steps) look almost identical to successful ones for the first 399 steps. Since the majority of transitions in these episodes still yield a reward of 1, the unsuccessful buffer begins to mirror the successful buffer, diluting the "failure signal" the agent needs to learn from.

**If we look closely at the CartPole reward system**, the environment outputs a reward of 1 for every step the cart remains upright and a reward of 0 only when it falls or moves out of bounds. This means 1 is the dominant reward, while 0 is extremely rare. So, the moment the environment outputs 0 is the only time a truly meaningful signal is transmitted to the network. It marks the exact transition that should be avoided. Therefore, again, when "long" unsuccessful experiences dominate the buffer, these crucial reawrd 0 transitions become even scarcer. 

### Failed Buffer
**To fix this**, I renamed the unsuccessful buffer to the Failed Buffer. Instead of saving entire "bad" episodes, I modified it to store only the specific terminal states where the reward is 0. By specifically sampling these "failure points," we ensure the model constantly remembers exactly what "losing" looks like, preventing the distribution from narrowing too much even when the agent becomes highly skilled. Let's check the results.

<figure class="figure-center">
  <img src="/posts/Mastering_CartPole/batch32_fixed_states.png" width="700">
  <figcaption>Figure 3. Example distributions</figcaption>
</figure>

<figure class="figure-center">
  <img src="/posts/Mastering_CartPole/batch32_total_reward.png" width="700">
  <figcaption>Figure 3. Example distributions</figcaption>
</figure>

The total rewards plot looks perfect. TD error also decreasing smoothly like loss plot in the supervised learning. Q-values of action right and left also are visibly seperated. And at the Stable state, it merges to 100, which is intuitively, and ideally correct, because I set gamma(discount value) as 0.99. $1/(1-0.99) = 100$

However, it turned out that it was a lucky case. I did two more runs, and below are the results.

<figure class="figure-center">
  <img src="/posts/Mastering_CartPole/65_32batch_fixed_states.png" width="500">
  <figcaption>Figure 3. Example distributions</figcaption>
</figure>

<figure class="figure-center">
  <img src="/posts/Mastering_CartPole/65_32batch_total_rewards.png" width="500">
  <figcaption>Figure 3. Example distributions</figcaption>
</figure>

<figure class="figure-center">
  <img src="/posts/Mastering_CartPole/batch32_case3_2.png" width="500">
  <figcaption>Figure 3. Example distributions</figcaption>
</figure>

<figure class="figure-center">
  <img src="/posts/Mastering_CartPole/batch32_case3_1.png" width="500">
  <figcaption>Figure 3. Example distributions</figcaption>
</figure>

As shown in the upper figure, reaching a stable score 500 could take a considerable amount of time. Also, the lower figure indicates that the performance remains somewhat unstable. **Why?** The first one likely occurs because the Q-network learns primarily from two extremes:the successful and failed buffers. Consequently, the network struggles to bridge the gap between failure and success.

To avoid slow learning, and to be more robust, I believe the Q-network should learn incrementally rather than attempting to jump directly to 'lucky' successful cases. To address this, I am considering removing the successful buffer and instead maintaining "normal" buffer, which stores all states except for failures, alongside the failed buffer. This normal buffer could effectively facilitate the incremental connection between failure and success.

### Just Normal buffer and failed buffer
Let's check the results.

<figure class="figure-center">
  <img src="/posts/Mastering_CartPole/normal_fixed_states.png" width="700">
  <figcaption>Figure 3. Example distributions</figcaption>
</figure>

<figure class="figure-center">
  <img src="/posts/Mastering_CartPole/normal_total_rewards.png" width="700">
  <figcaption>Figure 3. Example distributions</figcaption>
</figure>

The plot get prettier before. I think this change is beneficial for two reasons. First, it prevents too slow learning by incrementally connecting failed to successful as I said. Second, It also could help recovery from significant weight change, which I didn't meet yet.

Now, I think changing sampling ratio from two buffers, which was 7:3, might improve the stability.

### Sampling ratio
Let's try 6:4.

<figure class="figure-center">
  <img src="/posts/Mastering_CartPole/0.6_fixed_states.png" width="700">
  <figcaption>Figure 3. Example distributions</figcaption>
</figure>

<figure class="figure-center">
  <img src="/posts/Mastering_CartPole/0.6_total_rewards.png" width="700">
  <figcaption>Figure 3. Example distributions</figcaption>
</figure>

The results could be ambivalent. However, these results presents critical insights. if we look at the episodes between like 2700 to 2800, there are sudden many drops. **Why?** Below is the log of the range.

<figure class="figure-center">
  <img src="/posts/Mastering_CartPole/logs.png" width="500">
  <figcaption>Figure 3. Example distributions</figcaption>
</figure>

| Category | Count | Percentage |
| :--- | ---: | ---: |
| Edge Left | 33 | 2.4% |
| Edge Right | 679 | 49.9% |
| Leaning Left | 566 | 41.6% |
| Leaning Right | 83 | 6.1% |


This drop coincided with the agent encountering Edge Left and Leaning Right scenarios,classes that are significantly underrepresented in our current failure buffer (2.4% and 6.1%, respectively). This suggests that the model’s previous success was partly due to favorable sampling. Once exploration triggered these sparse failure modes, the agent lacked sufficient data in the buffer to recover quickly. Consequently, balancing the class distribution within the failure buffer is critical for robust learning.