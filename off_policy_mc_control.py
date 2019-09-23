from racetrack import Racetrack
from collections import defaultdict
import numpy as np


def play_episode(env, policy):
    episode =[]
    state = env.reset()
    while True:

        available_actions = env.get_actions()
        # For each available action, get respective probability
        action_probs = [policy[unravel_state(state), x, y] for (x,y) in available_actions]
        
        action = np.random.choice(available_actions, p=action_probs)

        next_state, reward, game_over, info = env.step(action)
        episode.append(state, action, reward)
        state = next_state
        if game_over:
            break
    return episode

def unravel_state(s):
    return s[0][0], s[0][1], s[1][0], s[1][1]

def unravel_action(a):
    return a[0], a[1]

def mc_control(env, gamma=0.01, iterations=1000):
    
    Q = np.zeros([unravel_state(env.state_space), unravel_action(env.action_space)])
    C = defaultdict(float)

    # TODO: Make this a np array
    target_policy = defaultdict(float)

    for e in range(iterations):
        behaviour_policy = np.ones(Q.shape)/env.nA # no need to recreate every episode
        
        episode = play_episode(env, behaviour_policy)

        G = 0.0
        W = 1.0
        
        for (state, action, reward) in episode:
            G = gamma*G + reward
            C[state, action] += W
            Q[unravel_state(state), unravel_action(action)] += (W / C[state, action])*(G - Q[unravel_state(state), unravel_action(action)])
            target_policy[state] = np.unravel_index(np.argmax(Q[unravel_state(state)], axis=None), env.action_space)

            if action != target_policy[state]:
                break
            else:
                W = W*(1/behaviour_policy[unravel_state(state), unravel_action(action)])
        
        return target_policy




if __name__ == "__main__":
    env = Racetrack()
    env.reset()
