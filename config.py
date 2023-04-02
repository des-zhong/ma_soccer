'''
长度单位为mm
速度为mm/s
'''
import numpy as np
actor_lr = 0.00001
critic_lr = 0.0001
gamma = 0.1
time_step = 0.01
field_length = 2500
field_width = 2500
scale = np.sqrt(field_width ** 2 + field_length ** 2) / 4
radius_soccer = 50/2
radius_player = 131/2
gate_length = 800
teamA_num = 2
teamB_num = 2
max_velocity = 100
max_iter = 3000
state_dim = 2 * (teamA_num + teamB_num) + 2
action_dim = 2*teamA_num
model_id = 373

fc1_dim = 128
fc2_dim = 64
fc3_dim = 64
