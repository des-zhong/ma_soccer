import maddpg
import utility
import numpy as np
from agent import Agent
import torch
from common.arguments import get_env_arg, get_args
from common.replay_buffer import Buffer
import os


class Runner:
    def __init__(self, args, env_arg, env):
        self.args = args
        self.env_arg = env_arg
        self.noise = args.noise_rate
        self.epsilon = args.epsilon
        self.episode_limit = args.max_episode_len
        self.env = env
        self.agents = self._init_agents()
        self.buffer = Buffer(args)
        self.save_path = self.args.save_dir + '/' + self.args.scenario_name
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def _init_agents(self):
        agents = []
        for i in range(self.args.n_agents):
            agent = Agent(i, self.args)
            agents.append(agent)
        return agents

    def process_train(self):
        returns = []
        for trial in range(self.args.max_iter):
            u = []
            s = [0] * self.args.n_agents
            s_ = [0] * self.args.n_agents
            print(f'{trial}/ {self.args.max_iter}')
            while True:
                state = self.env.reset()
                if self.env.collide():
                    for i in range(self.args.n_agents):
                        s[i] = state + np.random.uniform(-0.1, 0.1, state.shape)
                    break
            command = np.array([])
            for steps in range(self.args.max_episode_len):

                with torch.no_grad():
                    for i in range(self.args.n_agents):
                        action = self.agents[i].select_action(s[i], self.args.noise_rate, self.args.epsilon)
                        u.append(action)
                        command = np.concatenate((command, action))
                for i in range(self.env_arg.num_teamB):
                    command = np.concatenate((command, np.random.rand(2, 1).squeeze() * 20 - 10))
                state_next, done, r = self.env.run(command)
                for i in range(self.args.n_agents):
                    s_[i] = state_next + np.random.uniform(-0.1, 0.1, state_next.shape)

                self.buffer.store_episode(s, u, r, s_)
                s = s_.copy()

                if self.buffer.current_size >= self.args.batch_size:
                    transitions = self.buffer.sample(self.args.batch_size)
                    for j in range(self.args.n_agents):
                        other_agents = self.agents.copy()
                        other_agents.remove(self.agents[j])
                        self.agents[j].learn(transitions, other_agents)
                if trial > 0 and trial % self.args.evaluate_rate == 0:

                    returns.append(self.evaluate())
                    self.noise = max(0.05, self.noise - 0.0000005)
                    self.epsilon = max(0.05, self.epsilon - 0.0000005)
                    np.save(self.save_path + '/returns.pkl', returns)

    def evaluate(self):
        returns = []
        for episode in range(self.args.evaluate_episodes):
            # reset the environment
            while True:
                self.env.reset()
                if self.env.collide():
                    break
            rewards = 0
            for time_step in range(self.args.evaluate_episode_len):
                command = np.array([])
                state, index_a, index_b = self.env.derive_arc()
                with torch.no_grad():
                    for i in range(self.env_arg.num_teamA):
                        action = self.agents[i].select_action(state, self.args.noise_rate, self.args.epsilon)
                        command = np.concatenate((command, action))
                for i in range(self.env_arg.num_teamB):
                    command = np.concatenate((command, np.array([0, 0])))
                print(command)
                done, r = self.env.run(command)
                rewards += r[0]
            returns.append(rewards)
            print('Returns is', rewards)
        return sum(returns) / self.args.evaluate_episodes


if __name__ == '__main__':
    env_arg = get_env_arg()
    args = get_args()
    field = utility.field(env_arg)
    runner = Runner(args, env_arg, field)
    runner.process_train()
