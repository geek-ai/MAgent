import math
import time
import random
import numpy as np

import magent
from magent.server import BaseServer
from magent.builtin.tf_model import DeepQNetwork
from magent.utility import FontProvider


def load_config(map_size):
    gw = magent.gridworld
    cfg = gw.Config()

    cfg.set({"map_width": map_size, "map_height": map_size})
    cfg.set({"minimap_mode": True})
    cfg.set({"embedding_size": 10})

    goal = cfg.register_agent_type(
        "goal",
        {'width': 1, 'length': 1, 'hp': 1,

         'can_absorb': True
         }
    )

    agent = cfg.register_agent_type(
        "agent",
        {'width': 1, 'length': 1, 'hp': 10, 'speed': 2,
         'view_range': gw.CircleRange(6),
         'damage': 2, 'step_recover': -10.0/350,

         'step_reward': -1,
         })

    g_goal = cfg.add_group(goal)
    g_agent = cfg.add_group(agent)

    g = gw.AgentSymbol(g_goal, 'any')
    a = gw.AgentSymbol(g_agent, 'any')

    cfg.add_reward_rule(gw.Event(a, 'collide', g), receiver=a, value=10)

    return cfg


def generate_map(env, map_size, goal_handle, handles, messages, font):
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

    def add_square(pos, side, gap):
        side = int(side)
        for x in range(center_x - side//2, center_x + side//2 + 1, gap):
            pos.append([x, center_y - side//2])
            pos.append([x, center_y + side//2])
        for y in range(center_y - side//2, center_y + side//2 + 1, gap):
            pos.append([center_x - side//2, y])
            pos.append([center_x + side//2, y])

    # goal
    pos = []
    add_square(pos, map_size * 0.6,  10)
    add_square(pos, map_size * 0.55, 10)
    add_square(pos, map_size * 0.5,  10)
    add_square(pos, map_size * 0.45, 10)
    add_square(pos, map_size * 0.4, 10)
    add_square(pos, map_size * 0.3, 10)
    env.add_agents(goal_handle, method="custom", pos=pos)
    circle_goal_num = env.get_num(goal_handle)

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

    base_y = (map_size - len(messages) * font.height) / 2
    for message in messages:
        base_x = (map_size - len(message) * font.width) / 2
        scale = 1
        for x in message:
            data = font.get(x)
            draw(base_x, base_y, scale, data)
            base_x += font.width
        base_y += font.height + 1

    alpha_goal_num = env.get_num(goal_handle) - circle_goal_num

    # agent
    pos = []

    add_square(pos, map_size * 0.9, 2)
    add_square(pos, map_size * 0.8, 2)
    add_square(pos, map_size * 0.7, 2)
    add_square(pos, map_size * 0.65, 2)

    pos = np.array(pos)
    pos = pos[np.random.choice(np.arange(len(pos)), int(circle_goal_num + alpha_goal_num * 1.1), replace=False)]

    env.add_agents(handles[0], method="custom", pos=pos)


class ArrangeServer(BaseServer):
    def __init__(self, path="data/arrange_model", messages=None):
        # some parameter
        map_size = 200
        eps = 0.15

        # init the game
        env = magent.GridWorld(load_config(map_size))
        font = FontProvider('data/font_8x8/basic.txt')

        handles = env.get_handles()
        food_handle, handles = handles[0], handles[1:]
        models = []
        models.append(DeepQNetwork(env, handles[0], 'arrange', use_conv=True))

        # load model
        models[0].load(path, 2)

        # init environment
        env.reset()
        generate_map(env, map_size, food_handle, handles, messages, font)

        # save to member variable
        self.env = env
        self.food_handle = food_handle
        self.handles = handles
        self.eps = eps
        self.models = models
        self.done = False
        self.map_size = map_size

        self.ct = 0

    def get_group_info(self):
        ret = self.env._get_groups_info()
        ret[1] = ret[0]
        return ret

    def get_static_info(self):
        ret = self.env._get_walls_info()
        return {'wall': ret}

    def step(self):
        handles = self.handles
        models = self.models
        env = self.env

        for j in range(2):
            obs = [env.get_observation(handle) for handle in handles]
            ids = [env.get_agent_id(handle) for handle in handles]

            for i in range(len(handles)):
                acts = models[i].infer_action(obs[i], ids[i], 'e_greedy', eps=self.eps)
                env.set_action(handles[i], acts)

            done = env.step()
            num = [env.get_num(handle) for handle in [self.food_handle] + handles]
            env.clear_dead()

        return done

    def get_data(self, frame_id, x_range, y_range):
        start = time.time()

        if not self.done:
            self.done = self.step()

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
