from racetrack import Racetrack
from collections import defaultdict


def play_episode(policy):
    pass

def mc_control(iterations=1000):
    
    Q = defaultdict(float)
    C = defaultdict(float)
    policy = defaultdict(float)

    for e in range(iterations):
        pass
        # b = any soft policy



if __name__ == "__main__":
    env = Racetrack()
    env.reset()
