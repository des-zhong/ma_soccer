import argparse

"""
Here are the param for the training

"""
a=2
b=2

def get_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario-name", type=str, default="football match", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=3000, help="maximum episode length")
    parser.add_argument("--max-iter", type=int, default=5000, help="number of time steps")
    parser.add_argument("--high-action", type=int, default=100, help="maximum action value")
    # 一个地图最多env.n个agents，用户可以定义min(env.n,num-adversaries)个敌人，剩下的是好的agent
    parser.add_argument("--num-adversaries", type=int, default=b, help="number of adversaries")
    parser.add_argument("--n-players", type=int, default=a + b, help="number of all agents")
    parser.add_argument("--n-agents", type=int, default=a, help="number of controlled agents")
    parser.add_argument("--state-dim", type=int, default=2*a+2*b+2, help="dim of state")
    parser.add_argument("--action-dim", type=int, default=2, help="dim of action")
    parser.add_argument("--command-dim", type=int, default=2*a, help="dim of action")
    parser.add_argument("--all-state-dim", type=int, default=a*(2 * a + 2 * b+2), help="dim of action")
    # Core training parameters
    parser.add_argument("--lr-actor", type=float, default=1e-4, help="learning rate of actor")
    parser.add_argument("--lr-critic", type=float, default=1e-3, help="learning rate of critic")
    parser.add_argument("--epsilon", type=float, default=0.1, help="epsilon greedy")
    parser.add_argument("--noise_rate", type=float, default=0.1, help="noise rate for sampling from a standard normal distribution ")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--tau", type=float, default=0.01, help="parameter for updating the target network")
    parser.add_argument("--buffer-size", type=int, default=int(5e5), help="number of transitions can be stored in buffer")
    parser.add_argument("--batch-size", type=int, default=256, help="number of episodes to optimize at the same time")
    # Checkpointing
    parser.add_argument("--save-dir", type=str, default="./model/2v2test", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=1000, help="save model once every time this many episodes are completed")
    parser.add_argument("--model-dir", type=str, default="", help="directory in which training state and model are loaded")

    # Evaluate
    parser.add_argument("--evaluate-episodes", type=int, default=10, help="number of episodes for evaluating")
    parser.add_argument("--evaluate-episode-len", type=int, default=100, help="length of episodes for evaluating")
    parser.add_argument("--evaluate", type=bool, default=False, help="whether to evaluate the model")
    parser.add_argument("--evaluate-rate", type=int, default=200, help="how often to evaluate model")
    args = parser.parse_args()

    return args

def get_env_arg():
    parser = argparse.ArgumentParser("args for simulation env")
    parser.add_argument("--field-length", type=int, default=2500, help="length of the field")
    parser.add_argument("--field-width", type=int, default=2500, help="width of the field")
    parser.add_argument("--radius-soccer", type=int, default=25, help="radius of the soccer")
    parser.add_argument("--radius-player", type=int, default=65, help="radius of the player")
    parser.add_argument("--gate-length", type=int, default=2500, help="length of the gate")
    parser.add_argument("--num-teamA", type=int, default=a, help="number of players in team A")
    parser.add_argument("--num-teamB", type=int, default=b, help="number of players in team B")
    args = parser.parse_args()

    return args
