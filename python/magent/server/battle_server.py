import math
import time
import numpy as np

import magent
from magent.server import BaseServer
from magent.builtin.tf_model import DeepQNetwork
import matplotlib.pyplot as plt


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
         'step_reward': -0.001,  'kill_reward': 100, 'dead_penalty': -0.05, 'attack_penalty': -1,
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
    for y in range(10, height // 2 + 25):
        pos.append((width / 2 - 5, y))
        pos.append((width / 2 - 4, y))
    for y in range(height // 2 - 25, height - 10):
        pos.append((width / 2 + 5, y))
        pos.append((width / 2 + 4, y))
    env.add_walls(pos=pos, method="custom")

    n = init_num
    side = int(math.sqrt(n)) * 2
    pos = []
    for x in range(width//2 - gap - side, width//2 - gap - side + side, 2):
        for y in range((height - side)//2, (height - side)//2 + side, 2):
            pos.append([x, y, 0])
    env.add_agents(handles[leftID], method="custom", pos=pos)

    # right
    n = init_num
    side = int(math.sqrt(n)) * 2
    pos = []
    for x in range(width//2 + gap, width//2 + gap + side, 2):
        for y in range((height - side)//2, (height - side)//2 + side, 2):
            pos.append([x, y, 0])
    env.add_agents(handles[rightID], method="custom", pos=pos)


class BattleServer(BaseServer):
    def __init__(self, path="data/battle_model"):
        # some parameter
        map_size = 125
        eps = 0.05

        # init the game
        env = magent.GridWorld(load_config(map_size))

        handles = env.get_handles()
        models = []
        models.append(DeepQNetwork(env, handles[0], 'trusty-l', use_conv=True))
        models.append(DeepQNetwork(env, handles[1], 'trusty-r', use_conv=True))

        # load model
        models[0].load(path, 0, 'trusty-l')
        models[1].load(path, 0, 'trusty-r')
        
        # init environment
        env.reset()
        generate_map(env, map_size, handles)

        # save to member variable
        self.env = env
        self.handles = handles
        self.eps = eps
        self.models = models
        self.map_size = map_size
        print(env.get_view2attack(handles[0]))
        plt.show()

    def get_group_info(self):
        ret = self.env._get_groups_info()
        return ret

    def get_static_info(self):
        ret = self.env._get_walls_info()
        return {'wall': ret}

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
        #plt.clf()
        #for c in counter:
        #    plt.bar(range(len(c)), c / np.sum(c))
        #plt.draw()
        #plt.pause(1e-8)

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
        done = self.step()

        if done:
            return None

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
        for i in range(-5, 6):
            for j in range(-5, 6):
                pos.append((x + i, y + j))
        self.env.add_agents(self.handles[g ^ 1], method="custom", pos=pos)

    def get_map_size(self):
        return self.map_size, self.map_size

    def get_numbers(self):
        return self.env.get_num(self.handles[0]), self.env.get_num(self.handles[1])
