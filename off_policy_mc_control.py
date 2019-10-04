from racetrack import Racetrack
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
# https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/blackJack-off-policy.py

ACTION_TO_INDEX = { (-1,-1):0, (-1,0):1, (-1,1):2, (0,-1):3, (0,0):4, (0,1):5, (1,-1):6, (1,0):7, (1,1):8}
INDEX_TO_ACTION = { 0:(-1,-1), 1:(-1,0), 2:(-1,1), 3:(0,-1), 4:(0,0), 5:(0,1), 6:(1,-1), 7:(1,0), 8:(1,1)}

def play_episode(env, Q):
    episode =[]
    state = env.reset()
    while True:
        action, _ = behaviour_policy(env, Q, state)
        next_state, reward, game_over, info = env.step(action)
        episode.append((state, action, reward))
        state = next_state
        if game_over:
            break
    return episode

def greedy_behaviour_policy(actions, env):
    '''
    Selects and action greedily with respect to distance to finnish and stays on track
    '''
    closest = 1000000 # not good but works for now
    best_action = actions[0]

    for (ax, ay) in actions:
        npx = env.px + env.vx + ax
        npy = env.py - (env.vy + ay)

        # return first action that gets us to goal
        if 29 <= npx and 0 <= npy <= 6:
            best_action = (ax, ay)
            break
        else:
            a = 29 - npx
            b = 0 if 0 <= npy <= 6 else npy - 6
            # pythagoras
            c = np.sqrt(a*a + b*b)

            if c < closest and env.is_on_track(npx, npy):
                closest = c
                best_action = (ax, ay)

    return best_action


def behaviour_policy(env, Q, state):
    # Among the available actions, we select the best one. If there is no best action, we select
    # and action randomly

    q_state = Q[unravel_state(state)]
    available_actions = env.get_available_actions() 
    available_action_indices = [ACTION_TO_INDEX[action] for action in available_actions]
    available_action_values = q_state[available_action_indices]

    if np.sum(available_action_values) != 0:
        # We have been in this state before and we choose the best action
        best_actions = [available_actions[action] for action in np.where(available_action_values == available_action_values.max())[0]]
        return greedy_behaviour_policy(best_actions, env), 1/len(best_actions)
    else:
        # We have not been in this state before and chose a random action
        return random_behaviour_policy(env)



def random_behaviour_policy(env):
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


def mc_control(env, gamma=0.01, iterations=10):
    
    Q = np.zeros([env.state_space[0], env.state_space[1], env.state_space[2], env.state_space[3], env.nA])
    C = defaultdict(float)

    target_policy = np.zeros(Q.shape)

    for e in range(iterations):
        if e % 10 == 0:
            print("Playing episode {} out of {}.".format(e, iterations), end='\r')

        episode = play_episode(env, Q)

        G = 0.0
        W = 1.0
        
        for i, (state, action, reward) in enumerate(reversed(episode)):

            #if e > 1000:
            #    dum = 0

            # We do not consider the winning timestep since it contributes 0
            # reward and we wont be able to improve from it
            if i == 0:
                continue

            G = gamma*G + reward

            C[state, action] += W

            s = unravel_state(state)
            #a1 = W / C[state, action]
            #a2 = G - Q[s[0], s[1], s[2], s[3], ACTION_TO_INDEX[action]]
            Q[s[0], s[1], s[2], s[3], ACTION_TO_INDEX[action]] += (W / C[state, action])*(G - Q[s[0], s[1], s[2], s[3], ACTION_TO_INDEX[action]])

            # if there are multiple best actions, select one at random
            best_action = np.random.choice(np.where(Q[s] == Q[s].max())[0])
            target_policy[unravel_state(state)] = np.eye(env.nA)[best_action]

            if ACTION_TO_INDEX[action] != best_action: 
                break
            else:
                _, prob = behaviour_policy(env, Q, state)
                W = W*(1/prob)
        
    return target_policy, Q

def draw_Q(Q, env):
    '''
    Display the headings of each state that has none 0 value. The purpose is only to visualize the paths explored during training.
    '''
    fig = plt.figure(figsize=(15,15))
    plt.imshow(env.racetrack, cmap='gray')
    xs = []; ys = []; vxs = []; vys = []

    for i, q in np.ndenumerate(Q):
        if q != 0:
            xs.append( i[0])
            ys.append( i[1])
            vxs.append(i[2])
            vys.append(i[3])

    
    plt.quiver(xs, ys, vxs, vys)

    plt.show()


if __name__ == "__main__":
    env = Racetrack()
    state = env.reset()

    policy, Q = mc_control(env, iterations=2000)

    draw_Q(Q, env)
    #print(policy)