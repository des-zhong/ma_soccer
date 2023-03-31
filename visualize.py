import sys
import pygame
from config import *
import time

white = (255, 255, 255)


def text_objects(text, font):
    textSurface = font.render(text, True, white)
    return textSurface, textSurface.get_rect()


def draw(all_state):
    pygame.init()
    ratio = 3
    my_font = pygame.font.SysFont("arial", 16)
    # 设置主屏窗口 ；设置全屏格式：flags=pygame.FULLSCREEN
    screen = pygame.display.set_mode((field_width / ratio, field_length / ratio))
    # 设置窗口标题
    pygame.display.set_caption('soccer game')
    wid = 20
    fzone_wid = 150
    gap = 50
    for state in all_state:
        teamA = state[0:4 * teamA_num]
        teamB = state[4 * teamA_num:4 * teamA_num + 4 * teamB_num]
        soccer_coord = [state[-4], state[-3]]
        screen.fill('green')
        gate = pygame.Rect(field_width / ratio - wid, (field_length - gate_length) / 2 / ratio, wid,
                           gate_length / ratio)
        fzone = pygame.Rect(field_width / ratio - fzone_wid, (field_length - gate_length) / 2 / ratio - gap, fzone_wid,
                            gate_length / ratio + gap * 2)
        pygame.draw.circle(screen, (255, 255, 255), (0, field_length / 2 / ratio), 200, width=3)
        pygame.draw.lines(screen, (255, 255, 255), False, [(0, 0), (0, field_length / ratio)], width=3)
        pygame.draw.lines(screen, (255, 255, 255), False,
                          [(field_width / 2 / ratio, 0), (field_width / 2 / ratio, field_length / ratio)], width=3)
        pygame.draw.rect(screen, (190, 190, 190), gate)
        pygame.draw.rect(screen, (255, 255, 255), fzone, width=3)
        for i in range(teamA_num):
            pygame.draw.circle(screen, (255, 0, 0), (
                (teamA[4 * i] + field_width / 2) / ratio, (-teamA[4 * i + 1] + field_length / 2) / ratio),
                               radius_player / ratio)
        for i in range(teamB_num):
            pygame.draw.circle(screen, (0, 0, 255), (
                (teamB[4 * i] + field_width / 2) / ratio, (-teamB[4 * i + 1] + field_length / 2) / ratio),
                               radius_player / ratio)
        pygame.draw.circle(screen, (255, 255, 255),
                           ((soccer_coord[0] + field_width / 2) / ratio, (-soccer_coord[1] + field_length / 2) / ratio),
                           radius_soccer / ratio)
        pygame.display.update()  # 更新屏幕内容
        time.sleep(time_step/20)
