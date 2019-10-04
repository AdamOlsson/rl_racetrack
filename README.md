## Reinforcement Learning Part 4
Part 4 of my Reinforcement Learning (RL) series. During this series, I dwell into the field of RL by applying various methods to video games to learn and understand how an algorthm can learn to play by itself. The motivation for doing this series is simply by pure interest and to gain knowledge and experience in the field of Machine Learning.

The litterature follow throughout this series is Reinforcement Learning "An Introduction" by Ricard S. Button and Andrew G. Barto. ISBN: 9780262039246

### Off-Policy Methods
Previous methods used in the series has used a single policy and improved it. What Off-Policy methods does in its simplest form is to learn an optimal policy by looking at another policy interact in the envirnoment. We call the policy learning _target policy_ and the policy interacting in the environment _behaviour policy_. These two policies can be compltely separate and an advantage of using off-policy methods is that the target policy can deterministic while the behaviour policy can continue to sample other state-action pairs.

### Off-Policy Monte Carlo Control
Off-Policy MC Control follow the GPI as previous algorithms and weighted importance sampling. Convergance to an optimal policy is assured if the each state-action pair is visited an infinite number of times. We solve this by simply letting the behaviour policy be non-deterministic. A potential problem with these methods are that they only learn from tails of episodes, when all of the remaining actions in the episode are greedy. If non-greedy actions are common, learning will be slow. 

This code has a bug that I currently don't know how to fix. I will get back to this in the future though to either fix the problem or apply a new Reinforcement Learning method.

### The Bug
Currently there exists a bug that makes Q(s,a) never be anything else than -1 once set. This occurs when traversing the episode and updating Q(s,a). During the first iteration in the for-loop when t = T, the reward is 0 because it is the terminal state. However, we skip this iteration because the pseudo code of Off-Policy Monte Carlo Control starts at t = T-1. Moving on to the second iteration when t = T-1, G is set to -1 because of G = gamma\*G + reward = gamma\*0 + (-1). If we then look at the second term in the update step of Q(s,a), more specifically (G - Q(s,a)) we discover a problem. Have we visited Q(s,a) once before Q(s,a) will be set to -1. This results in (G - Q(s,a)) = 0 and the whole update product to be 0 as well. This means that Q(s,a) will never be anythings else than -1 since we always add 0 to it.
