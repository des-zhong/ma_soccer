import sys
import pygame
# from config import *
import time
from common.arguments import get_env_arg

white = (255, 255, 255)


def text_objects(text, font):
    textSurface = font.render(text, True, white)
    return textSurface, textSurface.get_rect()


def draw(all_state):
    args = get_env_arg()
    pygame.init()
    ratio = 3
    my_font = pygame.font.SysFont("arial", 16)
    # 设置主屏窗口 ；设置全屏格式：flags=pygame.FULLSCREEN
    screen = pygame.display.set_mode((args.field_width / ratio, args.field_length / ratio))
    # 设置窗口标题
    pygame.display.set_caption('soccer game')
    wid = 20
    fzone_wid = 150
    gap = 50
    for state in all_state:
        teamA = state[0:4 * args.num_teamA]
        teamB = state[4 * args.num_teamA:4 * args.num_teamA + 4 * args.num_teamB]
        soccer_coord = [state[-4], state[-3]]
        screen.fill('green')
        gate = pygame.Rect(args.field_width / ratio - wid, (args.field_length - args.gate_length) / 2 / ratio, wid,
                           args.gate_length / ratio)
        fzone = pygame.Rect(args.field_width / ratio - fzone_wid,
                            (args.field_length - args.gate_length) / 2 / ratio - gap, fzone_wid,
                            args.gate_length / ratio + 2 * gap)
        pygame.draw.circle(screen, (255, 255, 255), (0, args.field_length / 2 / ratio), 200, width=3)
        pygame.draw.lines(screen, (255, 255, 255), False, [(0, 0), (0, args.field_length / ratio)], width=3)
        pygame.draw.lines(screen, (255, 255, 255), False,
                          [(args.field_width / 2 / ratio, 0),
                           (args.field_width / 2 / ratio, args.field_length / ratio)], width=3)
        pygame.draw.rect(screen, (190, 190, 190), gate)
        pygame.draw.rect(screen, (255, 255, 255), fzone, width=3)
        for i in range(args.num_teamA):
            color = (155 * i / args.num_teamA, 155 * (args.num_teamA - i - 1) / args.num_teamA, 0)
            coord = ((teamA[4 * i] + args.field_width / 2) / ratio, (-teamA[4 * i + 1] + args.field_length / 2) / ratio)
            pygame.draw.circle(screen, color, coord, args.radius_player / ratio)
            pygame.draw.circle(screen, color, (10 + i * 20, 10), 10)
        for i in range(args.num_teamB):
            color = (0, 0, 255)
            coord = ((teamB[4 * i] + args.field_width / 2) / ratio, (-teamB[4 * i + 1] + args.field_length / 2) / ratio)
            pygame.draw.circle(screen, color, coord, args.radius_player / ratio)
        soccer_color = (255, 255, 255)
        coord = ((soccer_coord[0] + args.field_width / 2) / ratio,
                 (-soccer_coord[1] + args.field_length / 2) / ratio)
        pygame.draw.circle(screen, soccer_color, coord, args.radius_soccer / ratio)
        pygame.display.update()  # 更新屏幕内容
        time.sleep(args.time_step / 20)
