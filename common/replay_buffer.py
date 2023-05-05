import threading
import numpy as np
import torch


class Buffer:
    def __init__(self, args):
        self.size = args.buffer_size
        self.args = args
        # memory management
        self.current_size = 0
        self.idx = 0
        # create the buffer to store info
        self.buffer = dict()
        for i in range(self.args.n_agents):
            self.buffer['o_%d' % i] = np.empty([self.size, self.args.state_dim + self.args.goal_dim])
            self.buffer['u_%d' % i] = np.empty([self.size, self.args.action_dim])
            self.buffer['r_%d' % i] = np.empty([self.size])
            self.buffer['o_next_%d' % i] = np.empty([self.size, self.args.state_dim + self.args.goal_dim])
        # thread lock
        self.lock = threading.Lock()

    # store the episode
    def store_episode(self, o, u, r, o_next):
        idxs = self._get_storage_idx()  # 以transition的形式存，每次只存一条经验
        idx = self.idx
        for i in range(self.args.n_agents):
            with self.lock:
                self.buffer['o_%d' % i][idxs] = o[i]
                self.buffer['u_%d' % i][idxs] = u[i]
                self.buffer['r_%d' % i][idxs] = r[i]
                self.buffer['o_next_%d' % i][idxs] = o_next[i]

    # sample the data from the replay buffer
    def sample(self, batch_size):
        temp_buffer = {}
        idx = np.random.randint(0, self.current_size, batch_size)
        for key in self.buffer.keys():
            temp_buffer[key] = self.buffer[key][idx]
        return temp_buffer

    def _get_storage_idx(self):
        if self.idx + 1 <= self.size:
            idx = self.idx
            self.idx += 1
        else:
            self.idx = 0
            idx = 0
        self.current_size = min(self.size, self.current_size + 1)
        return idx

    def clear(self):
        # memory management
        self.idx = 0
        self.current_size = 0
        # create the buffer to store info
        # self.buffer = dict()
        # for i in range(self.args.n_agents):
        #     self.buffer['o_%d' % i] = np.empty([self.size, self.args.state_dim])
        #     self.buffer['u_%d' % i] = np.empty([self.size, self.args.action_dim])
        #     self.buffer['r_%d' % i] = np.empty([self.size])
        #     self.buffer['o_next_%d' % i] = np.empty([self.size, self.args.state_dim])
        # self.lock = threading.Lock()


class PrioritizedReplayBuffer(Buffer):
    def __init__(self, args):
        self.weights = np.zeros(args.buffer_size, dtype=np.float32)  # stores weights for importance sampling
        self.eps = args.eps  # minimal priority for stability
        self.alpha = args.alpha  # determines how much prioritization is used, α = 0 corresponding to the uniform case
        self.beta = args.beta  # determines the amount of importance-sampling correction, b = 1 fully compensate for the non-uniform probabilities
        self.max_priority = args.eps  # priority for new samples, init as eps
        super().__init__(args)

    def store_episode(self, o, u, r, o_next):
        """
        Add a new experience to memory, and update it's priority to the max_priority.
        """
        idxs = self._get_storage_idx()  # 以transition的形式存，每次只存一条经验
        for i in range(self.args.n_agents):
            with self.lock:
                self.buffer['o_%d' % i][idxs] = o[i]
                self.buffer['u_%d' % i][idxs] = u[i]
                self.buffer['r_%d' % i][idxs] = r[i]
                self.buffer['o_next_%d' % i][idxs] = o_next[i]
        self.weights[idxs] = self.max_priority

    def sample(self, batch_size):
        """
        Sample a batch of experiences from the buffer with priority, and calculates the weights used for the correction of bias used in the Q-learning update
        Returns:
            batch: a batch of experiences as in the normal replay buffer
            weights: torch.Tensor (batch_size, ), importance sampling weights for each sample
            sample_idxs: numpy.ndarray (batch_size, ), the indexes of the sample in the buffer
        """
        ############################
        # YOUR IMPLEMENTATION HERE #
        P = self.weights[:self.current_size] / sum(self.weights[:self.current_size])
        # print(batch_size,  self.current_size, P)
        sample_idxs = np.random.choice(self.current_size, batch_size, replace=False, p=P)
        temp_buffer = {}
        for key in self.buffer.keys():
            temp_buffer[key] = self.buffer[key][sample_idxs]
        weights = torch.tensor((1 / P / self.current_size) ** self.beta)
        weights = weights / self.max_priority
        ############################
        return temp_buffer, weights, sample_idxs

    def update_priorities(self, data_idxs, priorities: np.ndarray):
        priorities = (priorities + self.eps) ** self.alpha
        self.weights[data_idxs] = priorities
        self.max_priority = max(self.weights)
