train.py: 定义runner类 

- process_train用于训练
- evaluate用于在训练过程中评估效果, 
- match加载已保存的网络进行比赛

main.py: 加载环境和训练完成的网络并进行仿真

utility: 定义足球仿真环境，使用pygame进行可视化

- derive _pos/arc/state: 提取state，分别是相对位置直角坐标、相对位置极坐标、绝对位置与速度。
- reset: 重置球场
- set_vel: 给球场中所有球员赋上速度
- set_coord: 根据已有速度更新所有物体位置
- run: 根据输入得command运行一个时间步长，返回为：next_state, flag, reward，flag为1时进球，-1时终止，0时继续

comman.arguments: 定义所有参数



运行：

训练：train.py

测试与可视化：main.py