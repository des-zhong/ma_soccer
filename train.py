import maddpg
import utility
import numpy as np
from agent import Agent
import torch
from common.arguments import get_env_arg, get_args
from common.replay_buffer import Buffer
from common.replay_buffer import PrioritizedReplayBuffer
import os
import visualize
import matplotlib.pyplot as plt
import time
from logger import Logger
import sys


class Runner:
    def __init__(self, args, env):
        np.random.seed(1)
        self.args = args
        self.noise = args.noise_rate
        self.epsilon_step = (args.max_epsilon - args.min_epsilon) / args.max_iter
        self.epsilon = args.max_epsilon
        self.episode_limit = args.max_episode_len
        self.env = env
        self.agents = self._init_agents()
        self.buffer = PrioritizedReplayBuffer(args) if args.use_per else Buffer(args)
        self.save_path = self.args.save_dir + self.args.scenario_name
        self.eval_reward = []
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def _init_agents(self):
        agents = []
        for i in range(self.args.n_agents):
            agent = Agent(i, self.args)
            agents.append(agent)
        return agents

    def process_train(self):

        print(f'model save path: {self.save_path}, use per: {args.use_per}, scenario name: {self.args.scenario_name}')
        for trial in range(self.args.max_iter):
            if trial % self.args.evaluate_rate == 0:
                self.evaluate()
            t1 = time.time()
            trial_s = []
            trial_u = []
            trial_r = []
            s = [0] * self.args.n_agents
            s_ = [0] * self.args.n_agents
            u = [0] * self.args.n_agents
            state = self.env.reset()
            for i in range(self.args.n_agents):
                s[i] = state
            trial_s.append(s.copy())
            trial_step = 0
            flag = 0
            kick_ord = [[0, 0, 0]]
            for steps in range(self.args.max_episode_len):
                command = np.array([])
                with torch.no_grad():
                    for i in range(self.args.n_agents):
                        action = self.agents[i].select_action(s[i], 0, self.epsilon)
                        u[i] = action.copy()
                        command = np.concatenate((command, action))
                for i in range(self.args.n_adversaries):
                    command = np.concatenate((command, np.random.rand(2, 1).squeeze() * 20 - 10))
                state_next, flag, r, kick, v_ball = self.env.run(command)
                if kick > 0:
                    kick_ord.append([steps, kick, v_ball])
                for i in range(self.args.n_agents):
                    s_[i] = state_next.copy()
                s = s_.copy()
                trial_s.append(s.copy())
                trial_u.append(u.copy())
                trial_r.append(r.copy())
                trial_step += 1
                if not flag == 0:
                    break
            kick_ord.append([trial_step, 0, 0])
            k = 0
            order = True
            t2 = time.time()
            for i in range(trial_step):
                # if kick_ord[k][0] <= i < kick_ord[k + 1][0]:
                #     trial_r[i][kick_ord[k + 1][1] - 1] += kick_ord[k + 1][2]
                # elif kick_ord[k + 1][0] < trial_step:
                #     k = k + 1
                if not flag == 0:
                    shoot_player = kick_ord[-1][1] - 1
                    trial_r[i] = 0.4 * flag + trial_r[i]
                    trial_r[i][shoot_player] += 0.3 * flag
                self.buffer.store_episode(trial_s[i], trial_u[i], trial_r[i], trial_s[i + 1])
                if self.buffer.current_size >= self.args.batch_size:
                    train_order = list(range(self.args.n_agents)) if order else list(
                        range(self.args.n_agents - 1, -1, -1))
                    order = not order
                    if isinstance(self.buffer, PrioritizedReplayBuffer):
                        # sample with priorities and update the priorities with td_error
                        transitions, weights, tree_idxs = self.buffer.sample(self.args.batch_size)
                        td_error = np.zeros(self.args.batch_size)
                        for j in train_order:
                            other_agents = self.agents.copy()
                            other_agents.remove(self.agents[j])
                            td_error += self.agents[j].learn(transitions, other_agents, weights)
                        self.buffer.update_priorities(tree_idxs, td_error)
                    else:
                        transitions = self.buffer.sample(self.args.batch_size)
                        for j in train_order:
                            other_agents = self.agents.copy()
                            other_agents.remove(self.agents[j])
                            self.agents[j].learn(transitions, other_agents)
            t3 = time.time()
            print(f"trial: {trial}, flag: {flag}, record cost: {t2 - t1}, train cost: {t3 - t2}")
            self.epsilon = max(self.args.min_epsilon, self.epsilon - self.epsilon_step)
        sys.stdout = Logger(args.save_dir + args.scenario_name + '/train.log')

    def evaluate(self):
        score = 0
        goal = 0
        k = 0
        t1 = time.time()
        for trial in range(self.args.evaluate_episodes):
            state = self.env.reset()
            for steps in range(self.args.max_episode_len):
                command = np.array([])
                with torch.no_grad():
                    for i in range(self.args.n_agents):
                        action = self.agents[i].select_action(state, 0, 0)
                        command = np.concatenate((command, action))
                for i in range(self.args.n_adversaries):
                    command = np.concatenate((command, np.random.rand(2, 1).squeeze() * 20 - 10))
                state_, flag, r, kick, v_ball = self.env.run(command)
                score += sum(r) + v_ball
                k = k + 1
                if not flag == 0:
                    score += 0.5 * flag * steps
                    if flag == 1:
                        goal += 1
                    break
                state = state_.copy()
        self.eval_reward.append(score / k)
        t = np.array(range(len(self.eval_reward))) * self.args.evaluate_rate
        plt.plot(t, self.eval_reward)
        plt.grid()
        plt.savefig(self.save_path + '/reward.png')
        t2 = time.time()
        print('goal out of {} is {}, evaluation time cost = {}'.format(self.args.evaluate_episodes, goal, t2 - t1))

    def match(self, match_num):
        print(
            f'model save path: {self.save_path}, use per: {self.args.use_per}, scenario name: {self.args.scenario_name}')
        goal = 0
        flag = 0
        for trial in range(match_num):
            trial_s = []
            state = self.env.reset()
            trial_s.append(self.env.derive_abs_state())
            for steps in range(self.args.max_episode_len):
                command = np.array([])
                with torch.no_grad():
                    for i in range(self.args.n_agents):
                        action = self.agents[i].select_action(state, 0, 0)
                        command = np.concatenate((command, action))

                for i in range(self.args.n_adversaries):
                    command = np.concatenate((command, np.random.rand(2, 1).squeeze() * 20 - 10))
                state_, flag, r, kick,_ = self.env.run(command)
                if not flag == 0:
                    if flag == 1:
                        goal += 1
                    break
                state = state_.copy()
                trial_s.append(self.env.derive_abs_state())
            print(f"trial: {trial}, flag: {flag}")
            visualize.draw(trial_s)
        print('goal out of {} is {}'.format(match_num, goal))


if __name__ == '__main__':
    env_arg = get_env_arg()
    state_form = 'polar' if env_arg.state_term == 1 else 'euclid'
    print(f'input state is in {state_form} form')
    args = get_args()
    field = utility.field(env_arg)
    runner = Runner(args, field)
    runner.process_train()
