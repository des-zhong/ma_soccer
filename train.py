import maddpg
import utility
import numpy as np
from agent import Agent
import torch
from common.arguments import get_env_arg, get_args
from common.replay_buffer import Buffer
import os
import visualize
import time


class Runner:
    def __init__(self, args, env):
        self.args = args
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
        for trial in range(self.args.max_iter):
            v_s = []
            trial_s = []
            trial_u = []
            trial_r = []
            s = [0] * self.args.n_agents
            s_ = [0] * self.args.n_agents
            u = [0] * self.args.n_agents
            while True:
                state = self.env.reset()
                if self.env.collide():
                    for i in range(self.args.n_agents):
                        s[i] = state
                        # + np.random.uniform(-0.1, 0.1, state.shape)
                    break
            trial_s.append(s.copy())
            trial_step = 0
            flag = 0
            for steps in range(self.args.max_episode_len):
                command = np.array([])
                with torch.no_grad():
                    for i in range(self.args.n_agents):
                        action = self.agents[i].select_action(s[i], self.args.noise_rate, self.args.epsilon)
                        u[i] = action.copy()
                        # print(action)
                        command = np.concatenate((command, action))
                for i in range(self.args.n_adversaries):
                    command = np.concatenate((command, np.random.rand(2, 1).squeeze() * 20 - 10))
                state_next, flag, r = self.env.run(command)
                for i in range(self.args.n_agents):
                    s_[i] = state_next.copy()
                s = s_.copy()
                trial_s.append(s.copy())
                trial_u.append(u.copy())
                trial_r.append(r.copy())
                trial_step += 1
                if not flag == 0:
                    break

            for i in range(trial_step):
                self.buffer.store_episode(trial_s[i], trial_u[i], trial_r[i], trial_s[i + 1])
                if self.buffer.current_size >= self.args.batch_size:
                    transitions = self.buffer.sample(self.args.batch_size)
                    for j in range(self.args.n_agents):
                        other_agents = self.agents.copy()
                        other_agents.remove(self.agents[j])
                        self.agents[j].learn(transitions, other_agents)
            print(trial, flag)
            if trial > 0 and trial % self.args.evaluate_rate == 0:
                self.evaluate()

    def evaluate(self):
        rewards = 0
        for trial in range(self.args.evaluate_episodes):
            while True:
                state = self.env.reset()
                if self.env.collide():
                    break

            for time_step in range(self.args.evaluate_episode_len):
                command = np.array([])
                with torch.no_grad():
                    for i in range(self.args.n_agents):
                        action = self.agents[i].select_action(state, 0, 0)
                        command = np.concatenate((command, action))
                for i in range(self.args.n_adversaries):
                    command = np.concatenate((command, np.array([0, 0])))
                state, flag, r = self.env.run(command)
                if not flag == 0:
                    rewards += (flag + 1) / 2
                    break
        print('goal out of {} is {}'.format(self.args.evaluate_episodes, rewards))

    def match(self, match_num):
        for trial in range(match_num):
            trial_s = []
            while True:
                state = self.env.reset()
                if self.env.collide():
                    trial_s.append(self.env.derive_state())
                    break
            print(trial)
            for steps in range(self.args.max_episode_len):
                command = np.array([])

                with torch.no_grad():
                    for i in range(self.args.n_agents):
                        action = self.agents[i].select_action(state, 0, 0)
                        command = np.concatenate((command, action))
                for i in range(self.args.n_adversaries):
                    command = np.concatenate((command, np.random.rand(2, 1).squeeze() * 20 - 10))
                state_, flag, r = self.env.run(command)

                if not flag == 0:
                    break
                state = state_.copy()
                trial_s.append(self.env.derive_state())
            visualize.draw(trial_s)


if __name__ == '__main__':
    env_arg = get_env_arg()
    args = get_args()
    field = utility.field(env_arg)
    runner = Runner(args, field)
    runner.process_train()
