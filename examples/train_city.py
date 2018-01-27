"""
Simulating a city with many cars. Every car has a goal.
"""

import argparse
import logging as log
import random
import time

import numpy as np

import magent

from magent.builtin.tf_model import DeepQNetwork, DeepRecurrentQNetwork
from magent.builtin.rule_model import RandomActor

random.seed(0)
np.random.seed(0)


def get_config(map_size):
    gw = magent.gridworld
    cfg = gw.Config()

    cfg.set({"map_width": map_size, "map_height": map_size})
    cfg.set({"view_width": 13, "view_height": 13})
    cfg.set({"embedding_size": 10})
    cfg.set({"reward_scale": 2})

    return cfg


def generate_map(env, map_size, handles):
    """ generate a map, which consists of two squares of agents and vertical lines"""
    build_width = 10
    build_height = 10

    road_width = 4
    road_height = road_width
    num_park = 12
    car_dense = 0.1

    width_margin = -1
    for width in range(3, 5):
        current_margin = (map_size - 2 - road_width) % (width + road_width)
        if current_margin % 2 == 0:
            if current_margin <= 2:
                build_width = width
                width_margin = current_margin
                break
            if width_margin == -1 or width_margin > current_margin:
                width_margin = current_margin
                build_width = width

    height_margin = -1
    for height in range(3, 5):
        current_margin = (map_size - 2 - road_width) % (height + road_width)
        if current_margin % 2 == 0:
            if current_margin <= 2:
                build_height = height
                height_margin = current_margin
                break
            if height_margin == -1 or height_margin > current_margin:
                height_margin = current_margin
                build_height = height

    light_pos = []
    width_margin /= 2
    height_margin /= 2
    width_num = 0
    height_num = 0
    banned = dict()
    build_pos = dict()
    for x in range(width_margin + road_width + 1, map_size - road_width - width_margin, build_width + road_width):
        for y in range(height_margin + road_height + 1,
                       map_size - road_height - height_margin, build_height + road_height):
            build_pos.setdefault((x, y, build_width, build_height), True)
            banned.setdefault((x - 1 + build_width, y - 1 + build_height), [False] * 8)
            banned[(x - 1 + build_width, y - 1 + build_height)][4] = True
            banned.setdefault((x - (build_width + road_width) - 1 + build_width, y - 1 + build_height), [False] * 8)
            banned[(x - (build_width + road_width) - 1 + build_width, y - 1 + build_height)][5] = True
            banned.setdefault((x - 1 + build_width, y - (build_height + road_height) - 1 + build_height), [False] * 8)
            banned[(x - 1 + build_width, y - (build_height + road_height) - 1 + build_height)][7] = True
            banned.setdefault((x - (build_width + road_width) - 1 + build_width,
                               y - (build_height + road_height) - 1 + build_height), [False] * 8)
            banned[(x - (build_width + road_width) - 1 + build_width,
                    y - (build_height + road_height) - 1 + build_height)][6] = True
            if height_num == 0:
                width_num += 1
        height_num += 1

    def delblock(x, y):
        if (x, y, build_width, build_height) in build_pos:
            del build_pos[(x, y, build_width, build_height)]
        banned.setdefault((x - 1 + build_width, y - 1 + build_height), [False] * 8)
        banned[(x - 1 + build_width, y - 1 + build_height)][4] = False
        banned.setdefault((x - (build_width + road_width) - 1 + build_width, y - 1 + build_height), [False] * 8)
        banned[(x - (build_width + road_width) - 1 + build_width, y - 1 + build_height)][5] = False
        banned.setdefault((x - 1 + build_width, y - (build_height + road_height) - 1 + build_height), [False] * 8)
        banned[(x - 1 + build_width, y - (build_height + road_height) - 1 + build_height)][7] = False
        banned.setdefault((x - (build_width + road_width) - 1 + build_width,
                           y - (build_height + road_height) - 1 + build_height), [False] * 8)
        banned[(x - (build_width + road_width) - 1 + build_width,
                y - (build_height + road_height) - 1 + build_height)][6] = False

    for y_index in range(width_num):
        x = (width_num - 1) // 3 * (build_width + road_width) + width_margin + road_width + 1
        y = y_index * (build_height + road_height) + height_margin + road_height + 1
        delblock(x, y)
        x = (width_num - 1) // 3 * 2 * (build_width + road_width) + width_margin + road_width + 1
        y = y_index * (build_height + road_height) + height_margin + road_height + 1
        delblock(x, y)

    for x_index in range(width_num):
        x = x_index * (build_width + road_width) + width_margin + road_width + 1
        y = (height_num - 1) // 3 * (build_height + road_height) + height_margin + road_height + 1
        delblock(x, y)
        x = x_index * (build_width + road_width) + width_margin + road_width + 1
        y = (height_num - 1) // 3 * 2 * (build_height + road_height) + height_margin + road_height + 1
        delblock(x, y)

    extra_block = []
    block_num = int(len(build_pos.keys()) * 0.3)
    for _ in range(block_num):
        while True:
            x = random.randint(0, width_num - 2) * (build_width + road_width) + width_margin + road_width + 1
            y = random.randint(0, height_num - 2) * (build_height + road_height) + height_margin + road_height + 1
            if (x, y, build_width, build_height) in build_pos:
                break
        if random.random() <= 0.5:
            extra_block.append([x + build_width, y, road_width, build_height])
            banned.setdefault((x - 1 + build_width, y - 1 + build_height), [False] * 8)
            banned[(x - 1 + build_width, y - 1 + build_height)][0] = True
            y -= road_height + build_height
            banned.setdefault((x - 1 + build_width, y - 1 + build_height), [False] * 8)
            banned[(x - 1 + build_width, y - 1 + build_height)][2] = True
        else:
            extra_block.append([x, y + build_height, build_width, road_height])
            banned.setdefault((x - 1 + build_width, y - 1 + build_height), [False] * 8)
            banned[(x - 1 + build_width, y - 1 + build_height)][3] = True
            x -= road_width + build_width
            banned.setdefault((x - 1 + build_width, y - 1 + build_height), [False] * 8)
            banned[(x - 1 + build_width, y - 1 + build_height)][1] = True

    def get_status(block_status):
        status = [True] * 4
        if block_status[0]:
            status[0] = False
        if block_status[1]:
            status[1] = False
        if block_status[2]:
            status[2] = False
        if block_status[3]:
            status[3] = False
        if not block_status[4] or not block_status[5]:
            status[0] = False
        if not block_status[5] or not block_status[6]:
            status[1] = False
        if not block_status[6] or not block_status[7]:
            status[2] = False
        if not block_status[7] or not block_status[4]:
            status[3] = False
        return status

    for x in range(width_margin + road_width + 1, map_size - road_width - width_margin, build_width + road_width):
        for y in range(height_margin + road_height + 1,
                       map_size - road_height - height_margin, build_height + road_height):
            if x - 1 + build_width * 2 + road_width * 2 + width_margin < map_size and \
                    y - 1 + build_height * 2 + road_height * 2 + height_margin < map_size:
                status = get_status(banned[(x - 1 + build_width, y - 1 + build_height)])
                if sum(status) == 0:
                    continue
                res = 0
                for i in range(len(status)):
                    res |= status[i] << i
                assert res != 0
                light_pos.append([x - 1 + build_width, y - 1 + build_height, road_width + 1, road_height + 1, res])

    build_pos = build_pos.keys() + extra_block
    parks_id = np.random.choice(range(len(build_pos)), size=num_park, replace=False)

    def f(loc):
        return loc[0] + loc[2] < map_size and loc[1] + loc[3] < map_size

    build_pos = filter(f, build_pos)
    light_pos = filter(f, light_pos)

    filled = set()
    for pos in build_pos + light_pos:
        x0, y0, w, h = pos[:4]
        for x in range(x0, x0 + w):
            for y in range(y0, y0 + h):
                filled.add((x, y))

    n = map_size * map_size * car_dense
    agent_pos = []
    i = 0
    while i < n:
        x, y = np.random.randint(1, map_size - 1), np.random.randint(1, map_size - 1)
        while (x, y) in filled:
            x, y = np.random.randint(1, map_size - 1), np.random.randint(1, map_size - 1)

        agent_pos.append([x, y])
        i += 1

    env.add_buildings(method="custom", pos=build_pos)
    env.add_parks(method='custom', pos=[build_pos[i] for i in parks_id])
    env.add_traffic_lights(method="custom", pos=light_pos)
    env.add_agents(method="custom", pos=agent_pos)


def play_a_round(env, map_size, handles, models, print_every, train=True, render=False, eps=None):
    env.reset()

    generate_map(env, map_size, handles)

    step_ct = 0
    done = False

    n = len(handles)
    obs = [[] for _ in range(n)]
    ids = [[] for _ in range(n)]
    acts = [[] for _ in range(n)]
    nums = [env.get_num(handle) for handle in handles]
    sample_buffer = magent.utility.EpisodesBuffer(capacity=1000)
    total_reward = [0 for _ in range(n)]

    print("===== sample =====")
    print("eps %.2f number %s" % (eps, nums))
    start_time = time.time()
    while not done:
        # take actions for every model
        for i in range(n):
            obs[i] = env.get_observation(handles[i])
            # pos = env.get_pos(handles[i])
            # id = 1
            # print(pos[id])
            # print(obs[i][0][id][:,:,0])
            ids[i] = env.get_agent_id(handles[i])
            acts[i] = models[i].infer_action(obs[i], ids[i], 'e_greedy', eps=eps)
            env.set_action(handles[i], acts[i])

        # simulate one step
        done = env.step()

        # sample
        step_reward = []
        for i in range(n):
            rewards = env.get_reward(handles[i])
            if train:
                alives = env.get_alive(handles[i])
                sample_buffer.record_step(ids[i], obs[i], acts[i], rewards, alives)
            s = sum(rewards)
            step_reward.append(s)
            total_reward[i] += s

        # render
        if render:
            env.render()

        # stat info
        nums = [env.get_num(handle) for handle in handles]

        # clear dead agents
        env.clear_dead()

        if step_ct % print_every == 0:
            print("step %3d,  nums: %s reward: %s,  total_reward: %s " %
                  (step_ct, nums, np.around(step_reward, 2), np.around(total_reward, 2)))
        step_ct += 1
        if step_ct > 300:
            break

    sample_time = time.time() - start_time
    print("steps: %d,  total time: %.2f,  step average %.2f" % (step_ct, sample_time, sample_time / step_ct))

    # train
    total_loss, value = 0, 0
    if train:
        print("===== train =====")
        start_time = time.time()
        total_loss, value = models[0].train(sample_buffer, 500)
        train_time = time.time() - start_time
        print("train_time %.2f" % train_time)

    def round_list(l):
        return [round(x, 2) for x in l]

    return total_loss, nums, round_list(total_reward), value


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--render_every", type=int, default=10)
    parser.add_argument("--n_round", type=int, default=2000)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--load_from", type=int)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--map_size", type=int, default=101)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--name", type=str, default="city")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument('--alg', default='dqn', choices=['dqn', 'drqn', 'a2c'])
    args = parser.parse_args()

    # set logger
    magent.utility.init_logger(args.name)

    # init the game
    env = magent.TransCity(get_config(args.map_size))
    env.set_render_dir("build/render")

    # init models
    batch_size = 256
    target_update = 1000
    train_freq = 5

    handles = [0]

    models = []
    models.append(DeepQNetwork(env, handles[0], "cars",
                               batch_size=batch_size,
                               memory_size=2 ** 20, target_update=target_update,
                               train_freq=train_freq))

    # load if
    savedir = 'save_model'
    if args.load_from is not None:
        start_from = args.load_from
        print("load ... %d" % start_from)
        for model in models:
            model.load(savedir, start_from)
    else:
        start_from = 0

    # print debug info
    print(args)
    print("view_space", env.get_view_space(handles[0]))
    print("feature_space", env.get_feature_space(handles[0]))

    # play
    start = time.time()
    for k in range(start_from, start_from + args.n_round):
        tic = time.time()
        eps = magent.utility.piecewise_decay(k, [0, 700, 1400], [1, 0.2, 0.05]) if not args.greedy else 0
        loss, num, reward, value = play_a_round(env, args.map_size, handles, models,
                                                train=args.train, print_every=50,
                                                render=args.render or (k + 1) % args.render_every == 0,
                                                eps=eps)  # for e-greedy

        log.info("round %d\t loss: %s\t num: %s\t reward: %s\t value: %s" % (k, loss, num, reward, value))
        print("round time %.2f  total time %.2f\n" % (time.time() - tic, time.time() - start))

        # save models
        if (k + 1) % args.save_every == 0 and args.train:
            print("save model... ")
            for model in models:
                model.save(savedir, k)
