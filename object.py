import CalcRule as CR


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

        self.vel = CR.vec2D(2 * vry * s * c + vrx * (c ** 2 - s ** 2) + vx1 + 0.1 * dx,
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
        self.vel = CR.vec2D(vx1 * s ** 2 - vy1 * c * s + vx2 * c ** 2 + vy2 * s * c - coef * d * c,
                         vx2 * c * s + vy2 * s ** 2 - vx1 * s * c + vy1 * c ** 2 - coef * d * s)
        player.vel = CR.vec2D(vx2 * s ** 2 - vy2 * c * s + vx1 * c ** 2 + vy1 * s * c + coef * d * c,
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
