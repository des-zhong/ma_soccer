import utility
from common.arguments import get_env_arg
from common.arguments import get_args

if __name__ == '__main__':
    env_arg = get_env_arg()
    args = get_args()
    field = utility.field(env_arg)
    field.match(10, args)
    # field.test_collide(1)