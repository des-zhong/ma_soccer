import fieldEnv
from common.arguments import get_args
from runner import Runner
import sys
from logger import Logger

if __name__ == '__main__':
    args = get_args()
    field = fieldEnv.field(args)
    runner = Runner(args, field)
    runner.match(5, True)
    # field.match()
    # field.match(10000, 30)
    # field.test_collide(1)