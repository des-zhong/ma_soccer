import utility
from common.arguments import get_env_arg
from common.arguments import get_args
from train import Runner

if __name__ == '__main__':
    env_arg = get_env_arg()
    args = get_args()
    field = utility.field(env_arg)
    runner = Runner(args, field)
    runner.match(20)
