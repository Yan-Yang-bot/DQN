import numpy as np
from collections import deque


class ReplayBuffer:

    replay_mem_size = 10000

    """ Class for Experience Replay """
    def __init__(self):
        """
        :param replay_mem_size: The maximum buffer size
        """
        self.mem = deque()

    def add(self, experience):
        """
        Add a transition to the buffer. In case of buffer size overflow,
        remove the oldest transition in record.
        :param experience: the transition (s_t, a_t, r_{t+1}, s_{t+1}, done_flag)
        """
        self.mem.append(experience)
        if len(self.mem)>self.replay_mem_size:
            self.mem.popleft()

    def sample(self, rewFunc, batch_size=32):
        """
        Sample a batch of transitions (s_t, a_t, r_{t+1}, s_{t+1}, done_flag)
        where done_flag indicates whether s_t is a terminal state of an episode
        :param batch_size: Number of transitions in this batch
        :return: A batch of transitions uniformly sampled from the buffer
        """
        row_i = np.random.choice(len(self.mem), batch_size)
        yy, ss, aa = [], [], []
        for i in row_i:
            experience = self.mem[i]
            ss.append(experience.s)
            aa.append([experience.a])
            if experience.done:
                yy.append([experience.r])
            else:
                yy.append([experience.r + rewFunc(experience.sp)])
        return yy, ss, aa
