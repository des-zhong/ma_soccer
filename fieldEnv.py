import numpy as np
import CalcRule as CR
from object import object


class field():
    def __init__(self, arg):
        self.numA = arg.num_teamA
        self.numB = arg.num_teamB
        self.length = arg.field_length
        self.width = arg.field_width
        self.radius_player = arg.radius_player
        self.radius_soccer = arg.radius_soccer
        self.time_step = arg.time_step
        self.gamma = arg.gamma_velocity

        self.gate_length = arg.gate_length
        # self.xlimA = [-width / 2 + radius_player, width / 2 - radius_player]
        self.xlimA = [-self.width / 8, self.width / 8]
        self.xlimB = [self.width / 4, self.width / 2 - arg.radius_player]
        self.ylimA = [-self.width / 16, self.width / 16]
        self.ylimB = [-self.length / 2 + arg.radius_player, self.length / 2 - arg.radius_player]
        self.score = [0, 0]
        self.teamA = []
        self.teamB = []
        self.soccer = None
        self.derive_state = self.derive_arc if arg.state_term == 1 else self.derive_pos

    def derive_abs_state(self):
        state = []
        for i in range(self.numA):
            state.append(self.teamA[i].coord.x)
            state.append(self.teamA[i].coord.y)
            state.append(self.teamA[i].vel.x)
            state.append(self.teamA[i].vel.y)
        for i in range(self.numB):
            state.append(self.teamB[i].coord.x)
            state.append(self.teamB[i].coord.y)
            state.append(self.teamB[i].vel.x)
            state.append(self.teamB[i].vel.y)
        state.append(self.soccer.coord.x)
        state.append(self.soccer.coord.y)
        state.append(self.soccer.vel.x)
        state.append(self.soccer.vel.y)
        return np.array(state)

    def derive_abs_pos(self):
        state = []
        for i in range(self.numA):
            state.append(self.teamA[i].coord.x)
            state.append(self.teamA[i].coord.y)
        for i in range(self.numB):
            state.append(self.teamB[i].coord.x)
            state.append(self.teamB[i].coord.y)
        state.append(self.soccer.coord.x)
        state.append(self.soccer.coord.y)
        return np.array(state) / self.width * 2

    def derive_pos(self):
        state = []
        for i in range(self.numA):
            state.append(-self.teamA[i].coord.x + self.soccer.coord.x)
            state.append(-self.teamA[i].coord.y + self.soccer.coord.y)
        for i in range(self.numB):
            state.append(-self.teamB[i].coord.x + self.soccer.coord.x)
            state.append(-self.teamB[i].coord.y + self.soccer.coord.y)
        state.append(self.width / 2 - self.soccer.coord.x)
        state.append(-self.soccer.coord.y)
        return np.array(state) / self.width * 2

    def derive_arc(self):
        state = []

        for i in range(self.numA):
            dx = -self.teamA[i].coord.x + self.soccer.coord.x
            dy = self.teamA[i].coord.y - self.soccer.coord.y
            r, theta = CR.arc(dx, dy)
            state = state + [4 * r / self.width, theta]
        for i in range(self.numB):
            dx = self.teamB[i].coord.x - self.soccer.coord.x
            dy = self.teamB[i].coord.y - self.soccer.coord.y
            r, theta = CR.arc(dx, dy)
            state = state + [4 * r / self.width, theta]
        dx = self.width / 2 - self.soccer.coord.x
        dy = -self.soccer.coord.y
        r, theta = CR.arc(dx, dy)
        state = state + [4 * r / self.width, theta]
        return np.array(state)

    def reset(self):
        while True:
            self.teamA = []
            self.teamB = []
            for i in range(self.numA):
                random_coord = CR.random(self.xlimA, self.ylimA)
                # random_coord = CR.vec2D(0, 0)
                self.teamA.append(object(random_coord, CR.vec2D(0, 0), self.radius_player, i))
            for i in range(self.numB):
                random_coord = CR.random(self.xlimB, self.ylimB)
                # random_coord = vec2D(800, 200)
                self.teamB.append(object(random_coord, CR.vec2D(0, 0), self.radius_player, i))
            # random_coord = CR.vec2D(self.width/7, 0)
            random_coord = CR.random([self.width / 8, self.width / 7], [-self.length / 16, self.length / 16])
            self.soccer = object(random_coord, CR.vec2D(0, 0), self.radius_soccer)
            if not self.collide():
                state = self.derive_state()
                break
        return state

    def detect_soccer(self):
        if self.soccer.coord.y > self.length / 2 - self.soccer.radius or self.soccer.coord.y < -self.length / 2 + self.soccer.radius:
            self.soccer.soccer_bounce(False)
        if self.soccer.coord.x < -self.width / 2 + self.soccer.radius:
            self.soccer.soccer_bounce(True)
        if self.soccer.coord.x > self.width / 2 - self.soccer.radius:
            if self.gate_length / 2 > self.soccer.coord.y > - self.gate_length / 2:
                return 1
            else:
                self.soccer.soccer_bounce(True)
        return 0

    def collide(self):
        for i in range(self.numA):
            for j in range(i + 1, self.numA):
                dist_to_other = self.teamA[i].coord.dist(self.teamA[j].coord)
                if dist_to_other <= 2 * self.radius_player:
                    return True
            for j in range(0, self.numB):
                dist_to_other = self.teamA[i].coord.dist(self.teamB[j].coord)
                if dist_to_other <= 2 * self.radius_player:
                    return True
            dist_to_ball = self.teamA[i].coord.dist(self.soccer.coord)
            if dist_to_ball <= self.radius_player + self.radius_soccer:
                return True
        for i in range(self.numB):
            for j in range(i + 1, self.numB):
                dist_to_other = self.teamB[i].coord.dist(self.teamB[j].coord)
                if dist_to_other <= 2 * self.radius_player:
                    return True
            dist_to_ball = self.teamB[i].coord.dist(self.soccer.coord)
            if dist_to_ball <= self.radius_player + self.radius_soccer:
                return True
        return False

    def detect_player(self):
        kick = 0
        for i in range(self.numA):
            if abs(self.teamA[i].coord.x) > self.width / 2 - self.radius_player:
                self.teamA[i].coord.x = (self.width / 2 - self.radius_player) * self.teamA[i].coord.x / abs(
                    self.teamA[i].coord.x)
            if abs(self.teamA[i].coord.y) > self.length / 2 - self.radius_player:
                self.teamA[i].coord.y = (self.length / 2 - self.radius_player) * self.teamA[i].coord.y / abs(
                    self.teamA[i].coord.y)
            for j in range(i + 1, self.numA):
                dist_to_other = self.teamA[i].coord.dist(self.teamA[j].coord)
                if dist_to_other <= 2 * self.radius_player:
                    self.teamA[i].crush(self.teamA[j])
            for j in range(0, self.numB):
                dist_to_other = self.teamA[i].coord.dist(self.teamB[j].coord)
                if dist_to_other <= 2 * self.radius_player:
                    self.teamA[i].crush(self.teamB[j])
            dist_to_ball = self.teamA[i].coord.dist(self.soccer.coord)
            if dist_to_ball < self.radius_player:
                return True, kick
            if dist_to_ball <= self.radius_player + self.radius_soccer:
                self.soccer.strike(self.teamA[i])
                kick = i + 1
        for i in range(self.numB):
            if abs(self.teamB[i].coord.x) > self.width / 2 - self.radius_player:
                self.teamB[i].coord.x = (self.width / 2 - self.radius_player) * self.teamB[i].coord.x / abs(
                    self.teamB[i].coord.x)
            if abs(self.teamB[i].coord.y) > self.length / 2 - self.radius_player:
                self.teamB[i].coord.y = (self.length / 2 - self.radius_player) * self.teamB[i].coord.y / abs(
                    self.teamB[i].coord.y)
            for j in range(i + 1, self.numB):
                dist_to_other = self.teamB[i].coord.dist(self.teamB[j].coord)
                if dist_to_other <= 2 * self.radius_player:
                    self.teamB[i].crush(self.teamB[j])
            dist_to_ball = self.teamB[i].coord.dist(self.soccer.coord)
            if dist_to_ball < self.radius_player:
                return True, kick
            if dist_to_ball <= self.radius_player + self.radius_soccer:
                self.soccer.strike(self.teamB[i])
                kick = -1
        return False, kick

    def set_vel(self, command):
        for i in range(self.numA):
            self.teamA[i].vel = CR.vec2D(command[2 * i], command[2 * i + 1])
        for i in range(self.numB):
            self.teamB[i].vel = CR.vec2D(command[2 * self.numA + 2 * i], command[2 * self.numA + 2 * i + 1])

    def set_coord(self):
        kick = 0
        for i in range(self.numA):
            self.teamA[i].process(False, self.time_step, self.gamma)
        for i in range(self.numB):
            self.teamB[i].process(False, self.time_step, self.gamma)
        # print(self.soccer.vel.y)
        flag = self.detect_soccer()
        self.soccer.process(True, self.time_step, self.gamma)
        return flag

    def vel_polar2eu(self, command):
        state = self.derive_arc()
        eu_command = command.copy()
        for i in range(self.numA):
            c = np.cos(state[2 * i + 1])
            s = np.sin(state[2 * i + 1])
            eu_command[2 * i] = command[2 * i + 1] * s - command[2 * i] * c
            eu_command[2 * i + 1] = command[2 * i + 1] * c + command[2 * i] * s
        for i in range(self.numA, self.numA + self.numB):
            c = np.cos(state[2 * i + 1])
            s = np.sin(state[2 * i + 1])
            eu_command[2 * i] = -command[2 * i + 1] * s + command[2 * i] * c
            eu_command[2 * i + 1] = command[2 * i + 1] * c + command[2 * i] * s
        return eu_command

    def run(self, command):
        state = self.derive_state()
        # command_eu = self.vel_polar2eu(command)
        self.set_vel(command)
        done, kick = self.detect_player()
        r = np.zeros(self.numA)
        v_ball = 0
        if done:
            state_ = state.copy()
            flag = -1
        else:
            flag = self.set_coord()
            state_ = self.derive_state()
            for i in range(self.numA):
                # if np.linalg.norm(state[-2:] - np.array([1 / 2, 0])) < 0.072:
                if np.linalg.norm(state[-2:] - np.array([0.1, 0])) < 0.1:
                    r[i] = 0
                    print(state)
                    flag = 1
                else:
                    r[i] = -1
        if kick > 0:
            print('kick')
        return state_, flag, r, kick, v_ball
