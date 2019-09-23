from racetrack import Racetrack
from collections import defaultdict
import numpy as np


def play_episode(env, policy):
    episode =[]
    state = env.reset()
    while True:
        available_actions = env.get_actions()

        # TODO: Fix so a none uniform policy will work as well
        action = np.random.choice(available_actions) # works if policy has uniform probability over actions

        next_state, reward, game_over, info = env.step(action)
        episode.append(state, action, reward)
        state = next_state
        if game_over:
            break
    return episode

def mc_control(env, iterations=1000):
    
    Q = defaultdict(float)
    C = defaultdict(float)
    target_policy = defaultdict(float)

    for e in range(iterations):
        behaviour_policy = np.ones([env.nS, env.nA])/env.nA # no need to recreate every episode
        
        episode = play_episode(env, behaviour_policy)

        G = 0
        W = 1
        
        for (s,a,r) in episode:
            pass




if __name__ == "__main__":
    env = Racetrack()
    env.reset()
