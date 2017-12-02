import math
import time

import matplotlib.pyplot as plt
import numpy as np

import magent
from magent.builtin.tf_model import DeepQNetwork
from magent.renderer.server import BaseServer


def load_config(map_size):
    gw = magent.gridworld
    cfg = gw.Config()

    cfg.set({"map_width": map_size, "map_height": map_size})
    cfg.set({"minimap_mode": True})

    cfg.set({"embedding_size": 10})

    small = cfg.register_agent_type(
        "small",
        {'width': 1, 'length': 1, 'hp': 10, 'speed': 2,
         'view_range': gw.CircleRange(6), 'attack_range': gw.CircleRange(1.5),
         'damage': 2, 'step_recover': 0.1,
         'step_reward': -0.001, 'kill_reward': 100, 'dead_penalty': -0.05, 'attack_penalty': -1,
         })

    g0 = cfg.add_group(small)
    g1 = cfg.add_group(small)

    a = gw.AgentSymbol(g0, index='any')
    b = gw.AgentSymbol(g1, index='any')

    cfg.add_reward_rule(gw.Event(a, 'attack', b), receiver=a, value=2)
    cfg.add_reward_rule(gw.Event(b, 'attack', a), receiver=b, value=2)

    return cfg


def generate_map(env, map_size, handles):
    width = map_size
    height = map_size

    init_num = 20

    gap = 3
    leftID, rightID = 0, 1

    # left
    pos = []
    for y in range(10, 45):
        pos.append((width / 2 - 5, y))
        pos.append((width / 2 - 4, y))
    for y in range(50, height // 2 + 25):
        pos.append((width / 2 - 5, y))
        pos.append((width / 2 - 4, y))

    for y in range(height // 2 - 25, height - 50):
        pos.append((width / 2 + 5, y))
        pos.append((width / 2 + 4, y))
    for y in range(height - 45, height - 10):
        pos.append((width / 2 + 5, y))
        pos.append((width / 2 + 4, y))
    env.add_walls(pos=pos, method="custom")

    n = init_num
    side = int(math.sqrt(n)) * 2
    pos = []
    for x in range(width // 2 - gap - side, width // 2 - gap - side + side, 2):
        for y in range((height - side) // 2, (height - side) // 2 + side, 2):
            pos.append([x, y, 0])
    env.add_agents(handles[leftID], method="custom", pos=pos)

    # right
    n = init_num
    side = int(math.sqrt(n)) * 2
    pos = []
    for x in range(width // 2 + gap, width // 2 + gap + side, 2):
        for y in range((height - side) // 2, (height - side) // 2 + side, 2):
            pos.append([x, y, 0])
    env.add_agents(handles[rightID], method="custom", pos=pos)


class BattleServer(BaseServer):
    def __init__(self, path="data/battle_model", total_step=1000, add_counter=10, add_interval=50):
        # some parameter
        map_size = 125
        eps = 0.05

        # init the game
        env = magent.GridWorld(load_config(map_size))

        handles = env.get_handles()
        models = []
        models.append(DeepQNetwork(env, handles[0], 'trusty-battle-game-l', use_conv=True))
        models.append(DeepQNetwork(env, handles[1], 'trusty-battle-game-r', use_conv=True))

        # load model
        models[0].load(path, 0, 'trusty-battle-game-l')
        models[1].load(path, 0, 'trusty-battle-game-r')

        # init environment
        env.reset()
        generate_map(env, map_size, handles)

        # save to member variable
        self.env = env
        self.handles = handles
        self.eps = eps
        self.models = models
        self.map_size = map_size
        self.total_step = total_step
        self.add_interval = add_interval
        self.add_counter = add_counter
        self.done = False
        print(env.get_view2attack(handles[0]))
        plt.show()

    def get_info(self):
        return (self.map_size, self.map_size), self.env._get_groups_info(), {'wall': self.env._get_walls_info()}

    def step(self):
        handles = self.handles
        models = self.models
        env = self.env

        obs = [env.get_observation(handle) for handle in handles]
        ids = [env.get_agent_id(handle) for handle in handles]

        counter = []
        for i in range(len(handles)):
            acts = models[i].infer_action(obs[i], ids[i], 'e_greedy', eps=self.eps)
            env.set_action(handles[i], acts)
            counter.append(np.zeros(shape=env.get_action_space(handles[i])))
            for j in acts:
                counter[-1][j] += 1
        # plt.clf()
        # for c in counter:
        #    plt.bar(range(len(c)), c / np.sum(c))
        # plt.draw()
        # plt.pause(1e-8)

        # code for checking the correctness of observation
        # for channel in range(7):
        #     x = magent.round(list(obs[1][0][0][:,:,channel]), 2)
        #     for row in x:
        #         print row
        #     print("-------------")
        # input()

        done = env.step()
        env.clear_dead()

        return done

    def get_data(self, frame_id, x_range, y_range):
        start = time.time()
        if self.done:
            return None
        self.done = self.step()
        pos, event = self.env._get_render_info(x_range, y_range)
        print(" fps ", 1 / (time.time() - start))
        return pos, event

    def add_agents(self, x, y, g):
        pos = []
        for i in range(-5, 5):
            for j in range(-5, 5):
                pos.append((x + i, y + j))
        self.env.add_agents(self.handles[g], method="custom", pos=pos)

        pos = []
        x = np.random.randint(0, self.map_size - 1)
        y = np.random.randint(0, self.map_size - 1)
        for i in range(-5, 5):
            for j in range(-5, 6):
                pos.append((x + i, y + j))
        self.env.add_agents(self.handles[g ^ 1], method="custom", pos=pos)

    def get_map_size(self):
        return self.map_size, self.map_size

    def get_banners(self, frame_id, resolution):
        red = '{}'.format(self.env.get_num(self.handles[0])), (200, 0, 0)
        vs = ' vs ', (0, 0, 0)
        blue = '{}'.format(self.env.get_num(self.handles[1])), (0, 0, 200)
        result = [(red, vs, blue)]

        tmp = '{} chance(s) remained'.format(
            max(0, self.add_counter)), (0, 0, 0)
        result.append((tmp,))

        tmp = '{} / {} steps'.format(frame_id, self.total_step), (0, 0, 0)
        result.append((tmp,))
        if frame_id % self.add_interval == 0 and frame_id < self.total_step and self.add_counter > 0:
            tmp = 'Please press your left mouse button to add agents', (0, 0, 0)
            result.append((tmp,))
        return result

    def get_status(self, frame_id):
        if frame_id % self.add_interval == 0 and self.add_counter > 0:
            return False
        elif frame_id >= self.total_step or self.done:
            return None
        else:
            return True

    def keydown(self, frame_id, key, mouse_x, mouse_y):
        return False

    def mousedown(self, frame_id, pressed, mouse_x, mouse_y):
        if frame_id % self.add_interval == 0 and frame_id < self.total_step and pressed[0] \
                and self.add_counter > 0 and not self.done:
            self.add_counter -= 1
            pos = []
            for i in range(-5, 5):
                for j in range(-5, 5):
                    pos.append((mouse_x + i, mouse_y + j))
            self.env.add_agents(self.handles[0], method="custom", pos=pos)

            pos = []
            x = np.random.randint(0, self.map_size - 1)
            y = np.random.randint(0, self.map_size - 1)
            for i in range(-5, 6):
                for j in range(-5, 5):
                    pos.append((x + i, y + j))
            self.env.add_agents(self.handles[1], method="custom", pos=pos)
            return True
        return False

    def get_endscreen(self, frame_id):
        if frame_id == self.total_step or self.done:
            if self.env.get_num(self.handles[0]) > self.env.get_num(self.handles[1]):
                return [(("You", (200, 0, 0)), (" win! :)", (0, 0, 0)))]
            else:
                return [(("You", (200, 0, 0)), (" lose. :(", (0, 0, 0)))]
        else:
            return []
