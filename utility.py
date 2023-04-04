import numpy as np
import visualize
import maddpg


class vec2D():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def add(self, vec2):
        return vec2D(self.x + vec2.x, self.y + vec2.y)

    def min(self, vec2):
        return vec2D(self.x - vec2.x, self.y - vec2.y)

    def mul(self, c):
        return vec2D(self.x * c, self.y * c)

    def dist(self, vec2):
        return np.sqrt((self.x - vec2.x) ** 2 + (self.y - vec2.y) ** 2)


def random(xlim, ylim):
    x = np.random.uniform(xlim[0], xlim[1])
    y = np.random.uniform(ylim[0], ylim[1])
    return vec2D(x, y)


def arc(dx, dy):
    r = np.sqrt(dx ** 2 + dy ** 2)
    theta = np.arctan(dy / dx)
    if dx < 0:
        if dy > 0:
            theta += np.pi
        else:
            theta -= np.pi
    return r, theta


def reArrangeOrder(state, index, index_a):
    id = int(2 * index_a[index])
    if id == 0:
        return state
    else:
        temp = state[id:id + 2]
        S = np.delete(state, [id, id + 1])
        S = np.insert(S, 0, temp)
        return S


class object():
    def __init__(self, coord, vel, radius, index=-1):
        self.coord = coord
        self.vel = vel
        self.radius = radius
        self.index = index

    def strike(self, player):
        d = player.coord.dist(self.coord)
        dx = self.coord.x - player.coord.x
        dy = self.coord.y - player.coord.y
        vy1 = player.vel.y
        vx1 = player.vel.x
        vy2 = self.vel.y
        vx2 = self.vel.x
        s = dy / d
        c = dx / d
        vrx = vx1 - vx2
        vry = vy1 - vy2
        if vrx * c + vry * s < 0:
            return

        self.vel = vec2D(2 * vry * s * c + vrx * (c ** 2 - s ** 2) + vx1 + 0.1 * dx,
                         2 * vrx * s * c - vry * (c ** 2 - s ** 2) + vy1 + 0.1 * dy)
        # print('vy=', 2 * vrx * s * c - vry * (c ** 2 - s ** 2) + vy1 + 0.1 * dy)

    def soccer_bounce(self, xbounds):
        coef = 0.9
        if xbounds:
            self.vel.x = -coef * self.vel.x
        else:
            self.vel.y = -coef * self.vel.y

    def crush(self, player):
        coef = -1
        d = player.coord.dist(self.coord)
        dx = self.coord.x - player.coord.x
        dy = self.coord.y - player.coord.y
        vy1 = player.vel.y
        vx1 = player.vel.x
        vy2 = self.vel.y
        vx2 = self.vel.x
        s = dy / d
        c = dx / d
        self.vel = vec2D(vx1 * s ** 2 - vy1 * c * s + vx2 * c ** 2 + vy2 * s * c - coef * d * c,
                         vx2 * c * s + vy2 * s ** 2 - vx1 * s * c + vy1 * c ** 2 - coef * d * s)
        player.vel = vec2D(vx2 * s ** 2 - vy2 * c * s + vx1 * c ** 2 + vy1 * s * c + coef * d * c,
                           vx1 * c * s + vy1 * s ** 2 - vx2 * s * c + vy2 * c ** 2 + coef * d * s)

    def process(self, fade, time_step, gamma):
        self.coord = self.coord.add(self.vel.mul(time_step))
        if not fade:
            return

        prev_vel = self.vel
        self.vel = self.vel.min(self.vel.mul(gamma * time_step))
        if self.vel.x * prev_vel.x < 0:
            self.vel.x = 0
        if self.vel.y * prev_vel.y < 0:
            self.vel.y = 0


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
        self.xlimA = [-self.width / 2 + arg.radius_player, 0]
        self.xlimB = [self.width / 4, self.width / 2 - arg.radius_player]
        self.ylimA = [-self.length / 2 + arg.radius_player, self.length / 2 - arg.radius_player]
        self.ylimB = [-self.length / 2 + arg.radius_player, self.length / 2 - arg.radius_player]
        self.score = [0, 0]
        self.teamA = []
        self.teamB = []
        for i in range(arg.num_teamA):
            random_coord = random(self.xlimA, self.ylimA)
            self.teamA.append(object(random_coord, vec2D(0, 0), arg.radius_player, i))
        for i in range(arg.num_teamB):
            random_coord = random(self.xlimB, self.ylimB)
            self.teamB.append(object(random_coord, vec2D(0, 0), arg.radius_player, i))
        self.soccer = object(vec2D(0, 0), vec2D(0, 0), arg.radius_soccer)

    def derive_state(self):
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
        return np.array(state)

    def derive_arc(self):
        state = []

        for i in range(self.numA):
            dx = -self.teamA[i].coord.x + self.soccer.coord.x
            dy = self.teamA[i].coord.y - self.soccer.coord.y
            r, theta = arc(dx, dy)
            state = state + [4 * r / self.width, theta]
        for i in range(self.numB):
            dx = self.teamB[i].coord.x - self.soccer.coord.x
            dy = self.teamB[i].coord.y - self.soccer.coord.y
            r, theta = arc(dx, dy)
            state = state + [4 * r / self.width, theta]
        dx = self.width / 2 - self.soccer.coord.x
        dy = -self.soccer.coord.y
        r, theta = arc(dx, dy)
        state = state + [4 * r / self.width, theta]
        return np.array(state)

    def reset(self):
        self.teamA = []
        self.teamB = []
        for i in range(self.numA):
            random_coord = random(self.xlimA, self.ylimA)
            random_coord = vec2D(-800, 200)
            self.teamA.append(object(random_coord, vec2D(0, 0), self.radius_player, i))
        for i in range(self.numB):
            # random_coord = random(self.xlimB, self.ylimB)
            random_coord = vec2D(800, 200)
            self.teamB.append(object(random_coord, vec2D(0, 0), self.radius_player, i))
        random_coord = vec2D(-100, 100)
        # random_coord = random([-self.width / 8, 0], [-self.length / 8, self.length / 8])
        self.soccer = object(random_coord, vec2D(0, 0), self.radius_soccer)
        state = self.derive_arc()
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
                    return False
            for j in range(0, self.numB):
                dist_to_other = self.teamA[i].coord.dist(self.teamB[j].coord)
                if dist_to_other <= 2 * self.radius_player:
                    return False
            dist_to_ball = self.teamA[i].coord.dist(self.soccer.coord)
            if dist_to_ball <= self.radius_player + self.radius_soccer:
                return False
        for i in range(self.numB):
            for j in range(i + 1, self.numB):
                dist_to_other = self.teamB[i].coord.dist(self.teamB[j].coord)
                if dist_to_other <= 2 * self.radius_player:
                    return False
            dist_to_ball = self.teamB[i].coord.dist(self.soccer.coord)
            if dist_to_ball <= self.radius_player + self.radius_soccer:
                return False
        return True

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
            self.teamA[i].vel = vec2D(command[2 * i], command[2 * i + 1])
        for i in range(self.numB):
            self.teamB[i].vel = vec2D(command[2 * self.numA + 2 * i], command[2 * self.numA + 2 * i + 1])

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

    def vel_polar2eu(self, state, command):
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
        state = self.derive_arc()
        command_eu = self.vel_polar2eu(state, command)
        self.set_vel(command_eu)
        done, kick = self.detect_player()
        r = np.zeros(self.numA)

        if done:
            state_ = state
            flag = -1
        else:
            flag = self.set_coord()
            state_ = self.derive_arc()
            for i in range(self.numA):
                r[i] = -command[2*i]

                # r[i] = -8*(np.cos(state[2*i+1])-np.cos(state_[2*i+1]))
                # _, t1 = arc(self.width / 2 - self.teamA[i].coord.x, self.teamA[i].coord.y - self.gate_length / 2)
                # _, t2 = arc(self.width / 2 - self.teamA[i].coord.x, self.teamA[i].coord.y + self.gate_length / 2)
            if kick > 0:
                print('kick')
                r[kick - 1] += 10

        return state_, flag, r
