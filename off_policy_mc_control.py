from racetrack import Racetrack
from collections import defaultdict
import numpy as np


def play_episode(env, policy):
    episode =[]
    state = env.reset()
    while True:

        available_actions = env.get_actions()
        # For each available action, get respective probability
        action_probs = [policy[state, x, y] for (x,y) in available_actions]
        
        action = np.random.choice(available_actions, p=action_probs)

        next_state, reward, game_over, info = env.step(action)
        episode.append(state, action, reward)
        state = next_state
        if game_over:
            break
    return episode

def mc_control(env, gamma=0.01, iterations=1000):
    
    Q = defaultdict(float)
    C = defaultdict(float)
    target_policy = defaultdict(float)

    for e in range(iterations):
        behaviour_policy = np.ones([env.nS, env.action_space[0], env.action_space[1]])/env.nA # no need to recreate every episode
        
        episode = play_episode(env, behaviour_policy)

        G = 0.0
        W = 1.0
        
        for (state, action, reward) in episode:
            G = gamma*G + reward
            C[state, action] += W
            Q[state, action] += (W/C[state, action])*(G - Q[state, action])
            #target_policy[state, action] = np.argmax(Q[state])




if __name__ == "__main__":
    env = Racetrack()
    env.reset()
