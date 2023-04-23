import utility
from common.arguments import get_env_arg
from common.arguments import get_args
from train import Runner
import sys
from logger import Logger

if __name__ == '__main__':
    env_arg = get_env_arg()
    args = get_args()
    field = utility.field(env_arg)
    runner = Runner(args, field)
    runner.match(10)
    # field.match()
    # field.match(10000, 30)
    # field.test_collide(1)