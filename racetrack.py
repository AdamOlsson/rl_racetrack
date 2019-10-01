# https://gist.github.com/vfdev-5/f88007b0d0f7ef68a84c269f74f18ca9

import numpy as np

class Racetrack():
    def __init__(self):
        self.OFFROAD_VALUE = 0
        self.FINISH_LINE_VALUE = 0.75
        self.START_LINE_VALUE = 0.25
        self.ONROAD_VALUE = 1
        # probably not my proudest function...
        def create_turn(fl, ofr, sl):
            racetrack = np.ones([30,30])
            racetrack[0:6,29]   = fl # finnish line
            racetrack[6:,23:]   = ofr
            racetrack[7:,22]    = ofr
            racetrack[0,:15]    = ofr
            racetrack[1:3,:14]  = ofr
            racetrack[3,:13]    = ofr
            racetrack[4:,:12]   = ofr
            racetrack[13:,12]   = ofr
            racetrack[21:,13]   = ofr
            racetrack[28:,14]   = ofr
            racetrack[29,15:22] = sl # start line
            return racetrack, ((29,15),(29,22)), ((0,29),(6,29))

        self.racetrack, self.start_line, self.end_line = create_turn(self.FINISH_LINE_VALUE, self.OFFROAD_VALUE, self.START_LINE_VALUE)

        self.max_v = 4

        self.nA = 9

        # state = ((px, py), (vx,vy))
        self.nS = self.racetrack.shape[1]*self.racetrack[0]*self.nA # this includes off road which can't be reached
        # all positions on track as well as all possible velocities (v = 0,1,2,3,4)
        self.state_space = (self.racetrack.shape[0], self.racetrack.shape[1], 5, 5)

        self.vx = self.vy = None
        self.px = self.py = None


    def reset(self):
        #print("Reset!")
        self.vx = self.vy = 0
        self.py = self.racetrack.shape[0] -1
        self.px = np.random.randint(self.start_line[0][1], self.start_line[1][1]+1) # randomly set starting pos on starting line
        return ((self.px, self.py), (self.vx, self.vy))


    def get_available_actions(self):
        actions = []
        px = self.px; py = self.py; vx = self.vx; vy = self.vy 
        for dvx in range(-1,2):
            for dvy in range(-1,2):
                nvx = vx + dvx
                nvy = vy + dvy
                if 0 < nvx < 5 and 0 < nvy < 5:
                    actions.append((dvx, dvy))
                elif 0 < nvx < 5 and nvy == 0 or 0 < nvy < 5 and nvx == 0:
                    actions.append((dvx, dvy))
                elif nvx == 0 and nvy == 0 and py == self.racetrack.shape[0]-1:
                    actions.append((dvx, dvy))
        return actions

    def is_on_track(self, x,y):
        return 0 <= x < self.racetrack.shape[1] and 0 <= y < self.racetrack.shape[0] and self.racetrack[y,x] != self.OFFROAD_VALUE


    def step(self, action):

        a = [(0,0), (action[0], action[1])]
        # each timestep actions are with prob 0.1 set to 0 
        (dvx, dvy) = a[np.random.choice(len(a), p=[0.1, 0.9])]

        #print("act ({}, {})".format(dvx,dvy))

        self.vx += dvx
        self.vy += dvy

        #print("vel ({}, {})".format(self.vx, self.vy))

        # temporary position
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
        elif not (0 <= tpy and tpy < self.racetrack.shape[1]) or not(0 <= tpx and tpx < self.racetrack.shape[0]):
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
            state = self.reset() if self.racetrack[tpy,tpx] == self.OFFROAD_VALUE else ((tpx, tpy), (self.vx, self.vy))

        # update pos
        self.px += self.vx
        self.py -= self.vy

        #print("pos ({}, {})".format(self.px, self.py))

        return state, reward, game_over, info