import time

import numpy as np
import random
import magent
from magent.builtin.tf_model import DeepQNetwork
from magent.renderer.server import BaseServer
from magent.utility import FontProvider


def remove_wall(d, cur_pos, wall_set, unit):
    if d == 0:
        for i in range(0, unit):
            for j in range(0, unit):
                temp = (cur_pos[0] + i, cur_pos[1] + unit + j)
                if temp in wall_set:
                    wall_set.remove(temp)
    elif d == 1:
        for i in range(0, unit):
            for j in range(0, unit):
                temp = (cur_pos[0] - unit + i, cur_pos[1] + j)
                if temp in wall_set:
                    wall_set.remove(temp)
    elif d == 2:
        for i in range(0, unit):
            for j in range(0, unit):
                temp = (cur_pos[0] + i, cur_pos[1] - unit + j)
                if temp in wall_set:
                    wall_set.remove(temp)
    elif d == 3:
        for i in range(0, unit):
            for j in range(0, unit):
                temp = (cur_pos[0] + unit + i, cur_pos[1] + j)
                if temp in wall_set:
                    wall_set.remove(temp)


def dfs(x, y, width, height, unit, wall_set):
    pos = set()
    trace = list()
    pos.add((x, y))
    trace.append((x, y))

    max_x = x + width
    max_y = y + height

    d = random.choice(range(4))
    pos_list = []
    flag = 0
    while len(trace) > 0:
        if flag == 4:
            cur_pos = trace[-1]
            trace.pop()
            if random.choice(range(2)) == 0:
                remove_wall(d, cur_pos, wall_set, unit)
            flag = 0
        if len(trace) == 0:
            break
        cur_pos = list(trace[-1])
        if d == 0:
            cur_pos[1] = max(y, cur_pos[1] - 2 * unit)
        elif d == 1:
            cur_pos[0] = min(max_x, cur_pos[0] + 2 * unit)
        elif d == 2:
            cur_pos[1] = min(max_y, cur_pos[1] + 2 * unit)
        elif d == 3:
            cur_pos[0] = max(x, cur_pos[0] - 2 * unit)
        if tuple(cur_pos) in pos:
            d = (d + 1) % 4
            flag += 1
        else:
            remove_wall(d, cur_pos, wall_set, unit)
            trace.append(tuple(cur_pos))
            pos.add(tuple(cur_pos))
            d = random.choice(range(4))


def clean_pos_set_convert_to_list(pos_set, pos_list):
    for v in pos_list:
        if v in pos_set:
            pos_set.remove(v)
    return list(pos_set)


def draw_line(x, y, width, height):
    pos_set = []
    for r in range(height):
        for c in range(width):
            pos_set.append((x + c, y + r))
    return pos_set


def open_the_door(x_s, y_s, w, h, unit):
    pos_list = []
    n_door = 15
    random_horizon_list_x = [x_s + (2 * np.random.choice(w // 2 // unit, n_door) + 1) * unit, x_s + (2 * np.random.choice(w // 2 // unit, n_door) - 1) * unit]
    random_vertical_list_y = [y_s + (2 * np.random.choice(h // 2 // unit, n_door) + 1) * unit, y_s + (2 * np.random.choice(h // 2 // unit, n_door) + 1) * unit]

    y_e = y_s + h - unit
    for v in random_horizon_list_x[0]:
        pos_list.extend([(v, y_s), (v + 1, y_s), (v, y_s + 1), (v + 1, y_s + 1)])
    for v in random_horizon_list_x[1]:
        pos_list.extend([(v, y_e), (v + 1, y_e), (v, y_e + 1), (v + 1, y_e + 1)])

    x_e = x_s + w - unit
    for v in random_vertical_list_y[0]:
        pos_list.extend([(x_s, v), (x_s, v + 1), (x_s + 1, v), (x_s + 1, v + 1)])
    for v in random_vertical_list_y[1]:
        pos_list.extend([(x_e, v), (x_e, v + 1), (x_e + 1, v), (x_e + 1, v + 1)])

    return pos_list


def create_maze(pos, width, height, unit, font_area):
    # draw block: with rect: left(x), top(y), width, height
    pos_set = []
    for i in range(height):
        if i % 2 == 0:
            pos_set.extend(draw_line(pos[0], pos[1] + i * unit, width * unit, unit))
            pos_set.extend(draw_line(pos[0], pos[1] + font_area[1] + i * unit, width * unit, unit))
            pos_set.extend(draw_line(pos[0] + i * unit, pos[1] + height * unit, unit, font_area[1]))
            pos_set.extend(draw_line(pos[0] + font_area[0] + i * unit, pos[1] + height * unit, unit, font_area[1]))

    for i in range(width):
        if i % 2 == 0:
            pos_set.extend(draw_line(pos[0] + i * unit, pos[1], unit, height * unit))
            pos_set.extend(draw_line(pos[0] + i * unit, pos[1] + font_area[1], unit, height * unit))
            pos_set.extend(draw_line(pos[0], pos[1] + i * unit, height * unit, unit))
            pos_set.extend(draw_line(pos[0] + font_area[0], pos[1] + i * unit, height * unit, unit))

    pos_set = set(pos_set)

    dfs(pos[0] + 2, pos[1] + 2, (width - 1) * unit, (height - 1) * unit, unit, pos_set)  # north
    dfs(pos[0] + 2, pos[1] + (height - 2) * unit, (height - 1) * unit, (width + 3) * unit, unit, pos_set)  # west
    dfs(pos[0] + height * unit, pos[1] + font_area[1] - unit, (width - height) * unit, (height - 1) * unit, unit, pos_set)  # south
    dfs(pos[0] + font_area[0] - unit, pos[1] + (height - 2) * unit, (height - 1) * unit, font_area[1] - (height + 1) * unit, unit, pos_set)  # east

    temp = []
    temp.extend(open_the_door(pos[0], pos[1], font_area[0] + height * unit, font_area[1] + height * unit, unit))
    res = clean_pos_set_convert_to_list(pos_set, temp)
    return res


def load_config(map_size):
    gw = magent.gridworld
    cfg = gw.Config()

    cfg.set({"map_width": map_size, "map_height": map_size})
    cfg.set({"minimap_mode": True})
    cfg.set({"embedding_size": 12})

    goal = cfg.register_agent_type(
        "goal",
        {'width': 1, 'length': 1,

         'can_absorb': True
         }
    )

    agent = cfg.register_agent_type(
        "agent",
        {'width': 1, 'length': 1, 'hp': 10, 'speed': 2,
         'view_range': gw.CircleRange(6),
         'damage': 2, 'step_recover': -10.0/400,

         'step_reward': 0,
         })

    g_goal = cfg.add_group(goal)
    g_agent = cfg.add_group(agent)

    g = gw.AgentSymbol(g_goal, 'any')
    a = gw.AgentSymbol(g_agent, 'any')

    cfg.add_reward_rule(gw.Event(a, 'collide', g), receiver=a, value=10)

    return cfg


def generate_map(mode, env, map_size, goal_handle, handles, messages, font):
    # pre-process message
    max_len = 8
    new = []
    for msg in messages:
        if len(msg) > max_len:
            for i in range(0, len(msg), max_len):
                new.append(msg[i:i+max_len])
        else:
            new.append(msg)
    messages = new

    center_x, center_y = map_size // 2, map_size // 2

    # create maze
    if mode == 1:
        radius = 90
        pos_list = create_maze([center_x - radius, center_y - radius], radius + 1, 15, 2, font_area=[radius * 2 - 28, radius * 2 - 28])
        env.add_walls(method="custom", pos=pos_list)

    def add_square(pos, side, gap):
        side = int(side)
        for x in range(center_x - side//2, center_x + side//2 + 1, gap):
            pos.append([x, center_y - side//2])
            pos.append([x, center_y + side//2])
        for y in range(center_y - side//2, center_y + side//2 + 1, gap):
            pos.append([center_x - side//2, y])
            pos.append([center_x + side//2, y])

    def draw(base_x, base_y, scale, data):
        w, h = len(data), len(data[0])
        pos = []
        for i in range(w):
            for j in range(h):
                if data[i][j] == 1:
                    start_x = i * scale + base_y
                    start_y = j * scale + base_x
                    for x in range(start_x, start_x + scale):
                        for y in range(start_y, start_y + scale):
                            pos.append([y, x])

        env.add_agents(goal_handle, method="custom", pos=pos)

    base_y = (map_size - len(messages) * font.height) // 2
    for message in messages:
        base_x = (map_size - len(message) * font.width) // 2
        scale = 1
        for x in message:
            data = font.get(x)
            draw(base_x, base_y, scale, data)
            base_x += font.width
        base_y += font.height + 1

    alpha_goal_num = env.get_num(goal_handle)

    # agent
    pos = []

    add_square(pos, map_size * 0.95, 1)
    add_square(pos, map_size * 0.90, 1)
    add_square(pos, map_size * 0.85, 1)
    add_square(pos, map_size * 0.80, 1)

    pos = np.array(pos)
    pos = pos[np.random.choice(np.arange(len(pos)), int(alpha_goal_num * 1.6), replace=False)]

    env.add_agents(handles[0], method="custom", pos=pos)


class ArrangeServer(BaseServer):
    def get_banners(self, frame_id, resolution):
        return []

    def keydown(self, frame_id, key, mouse_x, mouse_y):
        return False

    def get_status(self, frame_id):
        if self.done:
            return None
        else:
            return True

    def get_endscreen(self, frame_id):
        return []

    def mousedown(self, frame_id, key, mouse_x, mouse_y):
        return False

    def get_info(self):
        ret = self.env._get_groups_info()
        ret[1] = ret[0]
        return (self.map_size, self.map_size), ret, {'wall': self.env._get_walls_info()}

    def __init__(self, path="data/arrange_model", messages=None, mode=1):
        # some parameter
        map_size = 250
        eps = 0.15

        # init the game
        env = magent.GridWorld(load_config(map_size))
        font = FontProvider('data/font_8x8/basic.txt')

        handles = env.get_handles()
        food_handle, handles = handles[0], handles[1:]
        models = []
        models.append(DeepQNetwork(env, handles[0], 'arrange', use_conv=True))

        # load model
        models[0].load(path, 10)

        # init environment
        env.reset()
        generate_map(mode, env, map_size, food_handle, handles, messages, font)

        # save to member variable
        self.env = env
        self.food_handle = food_handle
        self.handles = handles
        self.eps = eps
        self.models = models
        self.done = False
        self.map_size = map_size
        self.new_rule_ct = 0
        self.pos_reward_ct = set()
        self.num = None

        self.ct = 0

    def step(self):
        handles = self.handles
        models = self.models
        env = self.env

        center_x = self.map_size // 2
        center_y = self.map_size

        for j in range(2):
            obs = [env.get_observation(handle) for handle in handles]
            ids = [env.get_agent_id(handle) for handle in handles]

            for i in range(len(handles)):
                if self.new_rule_ct > 0:
                    obs[i][1][:, 10:12] = 0
                else:
                    obs[i][1][:, 10:12] = 1
                acts = models[i].infer_action(obs[i], ids[i], 'e_greedy', eps=self.eps)
                env.set_action(handles[i], acts)

            done = env.step()

            goal_num = env.get_num(self.food_handle)
            rewards = env.get_reward(handles[0])

            for id_, r in zip(ids[0], rewards):
                if r > 0.05 and id_ not in self.pos_reward_ct:
                    self.pos_reward_ct.add(id_)

            if 1.0 * len(self.pos_reward_ct) / goal_num >= 0.99:
                self.new_rule_ct += 1
            
            self.num = [env.get_num(handle) for handle in [self.food_handle] + handles]
            env.clear_dead()

            if done:
                break

        return done

    def get_data(self, frame_id, x_range, y_range):
        start = time.time()

        if not self.done:
            self.done = self.step()
        print(self.done)
        if self.done:
            print("done!")

        pos, event = self.env._get_render_info(x_range, y_range)
        print(" fps ", 1 / (time.time() - start))
        return pos, event

    def add_agents(self, x, y, g):
        pos = []
        for i in range(-3, 3):
            for j in range(-3, 3):
                pos.append((x + i, y + j))
        self.env.add_agents(self.handles[g], method="custom", pos=pos)
        
    def get_map_size(self):
        return self.map_size, self.map_size

    def get_numbers(self):
        return self.num
