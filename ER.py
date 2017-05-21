import numpy as np
import random


class ER(object):

    def __init__(self, memory_size, state_dim, action_dim, reward_dim, qpos_dim, qvel_dim, batch_size, history_length=1):
        self.memory_size = memory_size
        self.actions = np.random.normal(scale=0.35, size=(self.memory_size, action_dim))
        self.rewards = np.random.normal(scale=0.35, size=(self.memory_size, ))
        self.states = np.random.normal(scale=0.35, size=(self.memory_size, state_dim))
        self.qpos = np.random.normal(scale=0.35, size=(self.memory_size, qpos_dim))
        self.qvel = np.random.normal(scale=0.35, size=(self.memory_size, qvel_dim))
        self.terminals = np.zeros(self.memory_size, dtype=np.float32)
        self.batch_size = batch_size
        self.history_length = history_length
        self.count = 0
        self.current = 0
        self.state_dim = state_dim
        self.action_dim = action_dim

        # pre-allocate prestates and poststates for minibatch
        self.prestates = np.empty((self.batch_size, self.history_length, state_dim), dtype=np.float32)
        self.poststates = np.empty((self.batch_size, self.history_length, state_dim), dtype=np.float32)
        self.traj_length = 2
        self.traj_states = np.empty((self.batch_size, self.traj_length, state_dim), dtype=np.float32)
        self.traj_actions = np.empty((self.batch_size, self.traj_length-1, action_dim), dtype=np.float32)

    def add(self, actions, rewards, next_states, terminals, qposs=[], qvels = []):
        # state is post-state, after action and reward
        for idx in range(len(actions)):
            self.actions[self.current, ...] = actions[idx]
            self.rewards[self.current] = rewards[idx]
            self.states[self.current, ...] = next_states[idx]
            self.terminals[self.current] = terminals[idx]
            if len(qposs) == len(actions):
                self.qpos[self.current, ...] = qposs[idx]
                self.qvel[self.current, ...] = qvels[idx]
            self.count = max(self.count, self.current + 1)
            self.current = (self.current + 1) % self.memory_size

    def get_state(self, index):
        assert self.count > 0, "replay memory is empy"
        # normalize index to expected range, allows negative indexes
        index = index % self.count
        # if is not in the beginning of matrix
        if index >= self.history_length - 1:
            # use faster slicing
            return self.states[(index - (self.history_length - 1)):(index + 1), ...]
        else:
            # otherwise normalize indexes and use slower list based access
            indexes = [(index - i) % self.count for i in reversed(range(self.history_length))]
            return self.states[indexes, ...]

    def sample(self):
        # memory must include poststate, prestate and history
        assert self.count > self.history_length
        # sample random indexes
        indexes = []
        while len(indexes) < self.batch_size:
            # find random index
            while True:
                # sample one index (ignore states wraping over
                index = random.randint(self.history_length, self.count - 1)
                # if wraps over current pointer, then get new one
                if index >= self.current > index - self.history_length:
                    continue
                # if wraps over episode end, then get new one
                # poststate (last screen) can be terminal state!
                if self.terminals[(index - self.history_length):index].any():
                    continue
                # otherwise use this index
                break

            # having index first is fastest in C-order matrices
            self.prestates[len(indexes), ...] = self.get_state(index - 1)
            self.poststates[len(indexes), ...] = self.get_state(index)
            indexes.append(index)

        actions = self.actions[indexes, ...]
        rewards = self.rewards[indexes, ...]
        if hasattr(self, 'qpos'):
            qpos = self.qpos[indexes, ...]
            qvels = self.qvel[indexes, ...]
        else:
            qpos = []
            qvels = []
        terminals = self.terminals[indexes]

        return np.squeeze(self.prestates, axis=1), actions, rewards, \
               np.squeeze(self.poststates, axis=1), terminals, qpos, qvels