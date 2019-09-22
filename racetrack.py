# https://gist.github.com/vfdev-5/f88007b0d0f7ef68a84c269f74f18ca9

import numpy as np
import matplotlib.pyplot as plt

class Racetrack():
    def __init__(self):

        # probably not my proudest function...
        def create_turn():
            racetrack = np.zeros([30,30])
            racetrack[0:6,29]   = 0.75 # finnish line
            racetrack[6:,23:]   = 1
            racetrack[7:,22]    = 1
            racetrack[0,:15]    = 1
            racetrack[1:3,:14]  = 1
            racetrack[3,:13]    = 1
            racetrack[4:,:12]   = 1
            racetrack[13:,12]   = 1
            racetrack[21:,13]   = 1
            racetrack[28:,14]   = 1
            racetrack[29,15:22] = 0.25 # start line
            return racetrack, ((29,15),(29,22)), ((0,29),(6,29))

        # TODO
        self.nA = 3 # +1, -1, 0 to the velocity components

        self.vx = self.vy = None
        self.px = self.py = None

        self.racetrack, self.start_line, self.end_line = create_turn()

    def reset(self):
        self.vx = self.vy = 0
        self.py = self.racetrack.shape[0] -1
        self.px = np.random.randint(self.start_line[0][1], self.start_line[1][1]+1) # randomly set starting pos on starting line

        return ((self.px, self.py), (self.vx, self.vy))

    def get_actions(self):
        
        actions = []

        for dvx in range(-3,4):
            for dvy in range(-3,4):
                nvx = self.vx + dvx
                nvy = self.vy + dvy

                if 0 < nvx <= 5 and 0 < nvy <= 5:
                    actions.append((dvx, dvy))
                # for the early cases when vx or vy hasn't been changed since start pos
                elif 0 < nvx <= 5 and nvy == 0 or 0 < nvy <= 5 and nvx == 0:
                    actions.append((dvx, dvy))
        return actions

    def step(self, action):
        
        # each timestep actions are with prob 0.1 set to 0 
        #self.vx, self.vy = np.random.choice([(self.vx, self.vy), (self.vx + action[0], self.vy + action[1])], p=[0.1, 0.9])
        if np.random.choice(10) > 0:
           self.vx += action[0]
           self.vy += action[1]
        

        tpx = self.px + self.vx # temp pos x
        tpy = self.py - self.vy # temp pos y
        
        # crossed finnish line?
        if self.end_line[0][0] < tpy and tpy < self.end_line[1][0] and tpx >= self.end_line[0][1]:
            # YES!
            reward = 0
            game_over = True
            info = ('Crossed the finnish line!')
            self.px = tpx
            self.py = tpy
            state = ((tpx, tpy, (self.vx, self.vy)))

        # check for pos out of bounds of map
        elif tpy < 0 or self.racetrack.shape[1] < tpy:
            # pos out of bounds and reset pos to starting line
            reward = -1
            game_over = False
            info = ('Pos out of bounds so reset to start line.')
            state = self.reset()

        # Not crossed finnish line && pos on map
        else:
            reward = -1
            game_over = False
            info = ()
            state = self.reset() if self.racetrack[tpy,tpx] == 1 else ((tpx, tpy), (self.vx, self.vy))

        # update pos
        self.px = tpx
        self.py = tpy

        return state, reward, game_over, info



if __name__ == "__main__":

    env = Racetrack()
    env.reset()

    a = env.get_actions()

    env.step(a[0])

    fig = plt.figure(figsize=(15,15))
    plt.imshow(env.racetrack)
    plt.show()

