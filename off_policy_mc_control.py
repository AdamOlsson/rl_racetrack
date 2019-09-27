from racetrack import Racetrack
from collections import defaultdict
import numpy as np

ACTION_TO_INDEX = { (-1,-1):0, (-1,0):1, (-1,1):2, (0,-1):3, (0,0):4, (0,1):5, (1,-1):6, (1,0):7, (1,1):8}

def play_episode(env):
    episode =[]
    state = env.reset()
    while True:
        action, _ = behaviour_policy(env)
        next_state, reward, game_over, info = env.step(action)
        episode.append((state, action, reward))
        state = next_state
        if game_over:
            break
    return episode

def behaviour_policy(env): 
    actions = env.get_available_actions()
    # Behaviour policy will try to keep the car on the track
    for i in range(len(actions)):
        random_action = actions[np.random.randint(len(actions))]
        npx = env.px + env.vx + random_action[0]
        npy = env.py - (env.vy + random_action[1])
        if env.is_on_track(npx, npy):
            break
    return random_action, 1/len(actions)

def unravel_state(s):
    return s[0][0], s[0][1], s[1][0], s[1][1]

def unravel_action(a):
    return a[0], a[1]


def mc_control(env, gamma=0.01, iterations=1000):
    
    Q = np.zeros([env.state_space[0], env.state_space[1], env.state_space[2], env.state_space[3], env.nA])
    C = defaultdict(float)

    target_policy = np.zeros(Q.shape)

    for e in range(iterations):
        if e % 10 == 0:
            print("Playing episode {} out of {}.".format(e, iterations))

        episode = play_episode(env)

        G = 0.0
        W = 1.0
        
        for (state, action, reward) in reversed(episode):
            G = gamma*G + reward
            C[state, action] += W
            
            s = unravel_state(state)
            Q[s[0], s[1], s[2], s[3], ACTION_TO_INDEX[action]] += (W / C[state, action])*(G - Q[s[0], s[1], s[2], s[3], ACTION_TO_INDEX[action]])
            
            best_action = np.argmax(Q[unravel_state(state)])
            target_policy[unravel_state(state)] = np.eye(env.nA)[best_action]

            if ACTION_TO_INDEX[action] != best_action:
                break
            else:
                _, prob = behaviour_policy(env)
                W = W*(1/prob)
        
    return target_policy, Q


if __name__ == "__main__":
    env = Racetrack()
    state = env.reset()

    policy, Q = mc_control(env)

    print(policy)