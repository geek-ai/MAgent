from __future__ import absolute_import
from __future__ import division

import math

import pygame

from magent.server.base_server import BaseServer
from magent.renderer.base_renderer import BaseRenderer


class PyGameRenderer(BaseRenderer):
    def __init__(self):
        super(PyGameRenderer, self).__init__()

    def start(
            self,
            server,
            animation_total=2,
            animation_stop=0,
            resolution=None,
            fps_soft_bound=60,
            background_rgb=pygame.Color(255, 255, 255),
            attack_line_rgb=pygame.Color(0, 0, 0),
            attack_dot_rgb=pygame.Color(0, 0, 0),
            attack_dot_size=0.3,
            text_rgb=pygame.Color(0, 0, 0),
            text_height=16,
            text_spacing=3,
            grid_rgba=(pygame.Color(0, 0, 0), 30),
            grid_size=10,
            grid_min_size=2,
            grid_max_size=100,
            zoom_rate=1 / 60,
            move_rate=4,
            stop_step=50,
            add_counter=10,
            full_screen=True
    ):
        def draw_line(surface, color, a, b):
            pygame.draw.line(
                surface, color,
                map(int, (round(a[0]), round(a[1]))),
                map(int, (round(b[0]), round(b[1])))
            )

        def draw_rect(surface, color, a, w, h):
            pygame.draw.rect(surface, color, pygame.Rect(
                map(int, (round(a[0]), round(a[1]), round(w + a[0] - round(a[0])), round(h + a[1] - round(a[1]))))
            ))

        if not isinstance(server, BaseServer):
            raise BaseException('property server must be an instance of BaseServer')

        pygame.init()
        pygame.display.init()

        if resolution is None:
            info = pygame.display.Info()
            resolution = info.current_w, info.current_h

        clock = pygame.time.Clock()
        
        if full_screen:
            canvas = pygame.display.set_mode(resolution, pygame.DOUBLEBUF, 0)
        else:
            canvas = pygame.display.set_mode(resolution, pygame.DOUBLEBUF, 0)
        pygame.display.set_caption('MAgent Renderer Window')
        text_formatter = pygame.font.SysFont(None, text_height, True)

        banner_formatter = pygame.font.SysFont(None, 32)

        map_size = server.get_map_size()
        view_position = [map_size[0] / 2 * grid_size - resolution[0] / 2, 
                         map_size[1] / 2 * grid_size - resolution[1] / 2]
        frame_id = 0

        groups = server.get_group_info()
        walls  = server.get_static_info()['wall']

        old_data = None
        new_data = None

        grids = None
        show_grid = True
        animation_progress = 0

        pause = False
        counter = 0

        while True:
            done = False

            # calculate the relative moues coordinates in the gridworld
            mouse_x, mouse_y = pygame.mouse.get_pos()
            mouse_x = int((mouse_x + view_position[0]) / grid_size)
            mouse_y = int((mouse_y + view_position[1]) / grid_size)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    done = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_g:
                        show_grid = not show_grid
                    elif event.key == pygame.K_a:
                        if pause:
                            server.add_agents(mouse_x, mouse_y, 0)
                            pause = False

            pressed = pygame.key.get_pressed()
            if pressed[pygame.K_ESCAPE]:
                pygame.quit()
                done = True

            if pressed[pygame.K_COMMA] or pressed[pygame.K_PERIOD]:
                # center before means the center before zoom operation
                # center after means the center after zoom operation
                # we need to keep that the above two are consistent during zoom operation
                # and hence we need to adjust view_position simultaneously
                center_before = (
                    (view_position[0] + resolution[0] / 2) / grid_size,
                    (view_position[1] + resolution[1] / 2) / grid_size
                )
                if pressed[pygame.K_COMMA]:
                    grid_size = max(grid_size - grid_size * zoom_rate, grid_min_size)
                    grids = None
                else:
                    grid_size = min(grid_size + grid_size * zoom_rate, grid_max_size)
                    grids = None
                center_after = (
                    (view_position[0] + resolution[0] / 2) / grid_size,
                    (view_position[1] + resolution[1] / 2) / grid_size
                )
                view_position[0] += (center_before[0] - center_after[0]) * grid_size
                view_position[1] += (center_before[1] - center_after[1]) * grid_size

            if pressed[pygame.K_LEFT]:
                view_position[0] -= move_rate * grid_size
                grids = None
            if pressed[pygame.K_RIGHT]:
                view_position[0] += move_rate * grid_size
                grids = None
            if pressed[pygame.K_UP]:
                view_position[1] -= move_rate * grid_size
                grids = None
            if pressed[pygame.K_DOWN]:
                view_position[1] += move_rate * grid_size
                grids = None

            if done:
                break

            # x_range: which vertical gridlines should be shown on the display
            # y_range: which horizontal gridlines should be shown on the display
            x_range = (
                max(0, int(math.floor(max(0, view_position[0]) / grid_size))),
                min(map_size[0], int(math.ceil(max(0, view_position[0] + resolution[0]) / grid_size)))
            )

            y_range = (
                max(0, int(math.floor(max(0, view_position[1]) / grid_size))),
                min(map_size[1], int(math.ceil(max(0, view_position[1] + resolution[1]) / grid_size)))
            )

            canvas.fill(background_rgb)

            if show_grid:
                if grids is None:
                    grids = pygame.Surface(resolution)
                    grids.set_alpha(grid_rgba[1])
                    grids.fill(background_rgb)

                    for i in range(x_range[0], x_range[1] + 1):
                        draw_line(
                            grids, grid_rgba[0],
                            (i * grid_size - view_position[0], max(0, view_position[1]) - view_position[1]),
                            (
                                i * grid_size - view_position[0],
                                min(view_position[1] + resolution[1], map_size[1] * grid_size) - view_position[1]
                            )
                        )
                    for i in range(y_range[0], y_range[1] + 1):
                        draw_line(
                            grids, grid_rgba[0],
                            (max(0, view_position[0]) - view_position[0], i * grid_size - view_position[1]),
                            (
                                min(view_position[0] + resolution[0], map_size[0] * grid_size) - view_position[0],
                                i * grid_size - view_position[1]
                            )
                        )
                canvas.blit(grids, (0, 0))

            if new_data is None or animation_progress > animation_total + animation_stop:
                buffered_new_data = server.get_data(
                    frame_id,
                    (view_position[0] / grid_size, (view_position[0] + resolution[0]) / grid_size),
                    (view_position[1] / grid_size, (view_position[1] + resolution[1]) / grid_size)
                )
                if buffered_new_data is None:
                    buffered_new_data = new_data
                else:
                    counter += 1
                    if add_counter and counter % stop_step == 0:
                        pause = True
                        add_counter -= 1
                old_data = new_data
                new_data = buffered_new_data
                frame_id += 1
                animation_progress = 0

            if new_data is not None:
                if old_data is None and animation_progress == 0:
                    animation_progress = animation_total
                rate = min(1.0, animation_progress / animation_total)
                for key in new_data[0]:
                    new_prop = new_data[0][key]
                    old_prop = old_data[0][key] if old_data is not None and key in old_data[0] else None
                    new_group = groups[new_prop[2]]
                    old_group = groups[old_prop[2]] if old_prop is not None else None
                    now_prop = [a * (1 - rate) + b * rate for a, b in
                                zip(old_prop, new_prop)] if old_prop is not None else new_prop
                    now_group = [a * (1 - rate) + b * rate for a, b in
                                 zip(old_group, new_group)] if old_group is not None else new_group

                    draw_rect(
                        canvas, pygame.Color(int(now_group[2]), int(now_group[3]), int(now_group[4])),
                        (
                            now_prop[0] * grid_size - view_position[0],
                            now_prop[1] * grid_size - view_position[1]
                        ),
                        now_group[0] * grid_size,
                        now_group[1] * grid_size
                    )

                for wall in walls:
                    x, y = wall[0], wall[1]
                    if x >= x_range[0] and x <= x_range[1] and y >= y_range[0] and y <= y_range[1]:
                        draw_rect(canvas, pygame.Color(127, 127, 127),
                                  (x *grid_size - view_position[0], y * grid_size - view_position[1]),
                                  grid_size, grid_size)

                for key, event_x, event_y in new_data[1]:
                    if not key in new_data[0]:
                        continue
                    new_prop = new_data[0][key]
                    old_prop = old_data[0][key] if old_data is not None and key in old_data[0] else None
                    new_group = groups[new_prop[2]]
                    old_group = groups[old_prop[2]] if old_prop is not None else None
                    now_prop = [a * (1 - rate) + b * rate for a, b in
                                zip(old_prop, new_prop)] if old_prop is not None else new_prop
                    now_group = [a * (1 - rate) + b * rate for a, b in
                                 zip(old_group, new_group)] if old_group is not None else new_group
                    draw_line(
                        canvas, attack_line_rgb,
                        (
                            now_prop[0] * grid_size - view_position[0] + now_group[0] / 2 * grid_size,
                            now_prop[1] * grid_size - view_position[1] + now_group[1] / 2 * grid_size
                        ),
                        (
                            event_x * grid_size - view_position[0] + grid_size / 2,
                            event_y * grid_size - view_position[1] + grid_size / 2
                        )
                    )
                    draw_rect(
                        canvas, attack_dot_rgb,
                        (
                            event_x * grid_size - view_position[0] + grid_size / 2 - attack_dot_size * grid_size / 2,
                            event_y * grid_size - view_position[1] + grid_size / 2 - attack_dot_size * grid_size / 2,
                        ),
                        attack_dot_size * grid_size,
                        attack_dot_size * grid_size
                    )

                if not pause or animation_progress < animation_total + animation_stop:
                    animation_progress += 1

                text_fps = text_formatter.render('FPS: {}'.format(int(clock.get_fps())), True, text_rgb)
                text_window = text_formatter.render(
                    'Window: (%.1f, %.1f, %.1f, %.1f)' % (
                        view_position[0], view_position[1],
                        view_position[0] + resolution[0],
                        view_position[1] + resolution[1]
                    ), True, text_rgb
                )

                text_grids = text_formatter.render('Numbers: %d' % len(new_data[0]), True, text_rgb)
                text_mouse = text_formatter.render('Mouse: (%d, %d)' % (mouse_x, mouse_y), True, text_rgb)
                text_please = banner_formatter.render('Please press a to add your agents', True, text_rgb)

                numbers = server.get_numbers()
                banner_red = banner_formatter.render('{}'.format(numbers[0]), True, pygame.Color(200, 0, 0))
                banner_vs = banner_formatter.render(' vs ', True, text_rgb)
                banner_blue = banner_formatter.render('{}'.format(numbers[1]), True, pygame.Color(0, 0, 200))

                canvas.blit(text_fps, (0, 0))
                canvas.blit(text_window, (0, (text_height + text_spacing) / 1.5))
                canvas.blit(text_grids, (0, (text_height + text_spacing) / 1.5 * 2))
                canvas.blit(text_mouse, (0, (text_height + text_spacing) / 1.5 * 3))
                if pause:
                    canvas.blit(text_please, (resolution[0] / 2 - 140, 32))

                canvas.blit(banner_blue, (resolution[0] / 2 - 45, 0))
                canvas.blit(banner_vs, (resolution[0] / 2, 0))
                canvas.blit(banner_red, (resolution[0] / 2 + 60, 0))

            pygame.display.update()
            clock.tick(fps_soft_bound)
