import numpy as np

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