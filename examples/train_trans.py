"""
train agents to walk through some walls, avoiding collide
"""

import argparse
import time
import os
import logging as log
import math
import random

import numpy as np

import magent
from magent.builtin.tf_model import DeepQNetwork, DeepRecurrentQNetwork


def get_config(map_size):
    gw = magent.gridworld
    cfg = gw.Config()

    cfg.set({"map_width": map_size * 2, "map_height": map_size})
    cfg.set({"minimap_mode": True})
    cfg.set({"embedding_size": 10})

    agent = cfg.register_agent_type(
        "agent",
        {'width': 1, 'length': 1, 'hp': 10, 'speed': 1,
         'view_range': gw.CircleRange(6),
         'damage': 2, 'step_recover': 0.1,

         'step_reward': -1,
         })

    g0 = cfg.add_group(agent)

    return cfg


leftID, rightID = 0, 1
def generate_map(env, map_size, handles):
    """ generate a map, which consists of two squares of agents and vertical lines"""
    width = map_size * 2
    height = map_size
    margin = map_size * 0.1
    line_num = 9
    wall_width = 4
    gap = 2
    road_height = 2
    road_num = 4
    init_num = margin * height * 0.8

    def random_add(x1, x2, y1, y2, n):
        added = set()
        ct = 0
        while ct < n:
            x = random.randint(x1, x2)
            y = random.randint(y1, y2)

            next = (x, y)
            if next in added:
                continue
            added.add(next)
            ct += 1
        return list(added)

    # left
    pos = random_add(0, margin, 0, height, init_num)
    env.add_agents(handles[leftID], method="custom", pos=pos)

    # right
    # pos = random_add(width - margin, width, 0, height, init_num)
    # env.add_agents(handles[rightID], method="custom", pos=pos)

    # wall
    lines = set()
    low, high = margin * 2 + wall_width, width - margin * 2 - wall_width
    ct = 0
    while ct < line_num:
        next = random.randint(low, high)
        collide = False
        for j in range(-wall_width - gap, wall_width+gap + 1):
            if next+j in lines:
                collide = True
                break

        if collide:
            continue
        lines.add(next)
        ct += 1

    lines = list(lines)
    walls = []
    for item in lines:
        road_skip = set()
        for i in range(road_num):
            road_start = random.randint(1, height-1 - road_height)
            for j in range(road_height):
                road_skip.add(road_start + j)

        for i in range(height):
            if i in road_skip:
                continue
            for j in range(-wall_width//2, wall_width//2 + 1):
                walls.append((item+j, i))

    env.add_walls(method="custom", pos=walls)



def play_a_round(env, map_size, handles, models, print_every, train=True, render=False, eps=None):
    env.reset()
    generate_map(env, map_size, handles)

    step_ct = 0
    done = False

    n = len(handles)
    obs  = [[] for _ in range(n)]
    ids  = [[] for _ in range(n)]
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
        if step_ct > 550:
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

    def round_list(l): return [round(x, 2) for x in l]
    return total_loss, nums, round_list(total_reward), value

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--render_every", type=int, default=10)
    parser.add_argument("--n_round", type=int, default=2000)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--load_from", type=int)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--map_size", type=int, default=60)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--name", type=str, default="battle")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument('--alg', default='dqn', choices=['dqn', 'drqn', 'a2c'])
    args = parser.parse_args()

    # set logger
    log.basicConfig(level=log.INFO, filename=args.name + '.log')
    console = log.StreamHandler()
    console.setLevel(log.INFO)
    log.getLogger('').addHandler(console)

    # init the game
    env = magent.GridWorld(get_config(args.map_size))
    env.set_render_dir("build/render")

    # two groups of agents
    names = [args.name + "-l", args.name + "-r"]
    handles = env.get_handles()

    # sample eval observation set
    eval_obs = None
    if args.eval:
        print("sample eval set...")
        env.reset()
        generate_map(env, args.map_size, handles)
        eval_obs = magent.utility.sample_observation(env, handles, 2048, 500)[0]

    # init models
    batch_size = 256
    unroll_step = 8
    target_update = 1000
    train_freq = 5

    models = []
    if args.alg == 'dqn':
        models.append(DeepQNetwork(env, handles[0], "selfplay",
                                   batch_size=batch_size,
                                   memory_size=2 ** 20, target_update=target_update,
                                   train_freq=train_freq, eval_obs=eval_obs))
    elif args.alg == 'drqn':
        models.append(DeepRecurrentQNetwork(env, handles[0], "selfplay",
                                            batch_size=batch_size/unroll_step, unroll_step=unroll_step,
                                            memory_size=2 * 8 * 625, target_update=target_update,
                                            train_freq=train_freq, eval_obs=eval_obs))
    else:
        raise NotImplementedError

    models.append(models[0])

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
                                                render=args.render or (k+1) % args.render_every == 0,
                                                eps=eps)  # for e-greedy

        log.info("round %d\t loss: %s\t num: %s\t reward: %s\t value: %s" % (k, loss, num, reward, value))
        print("round time %.2f  total time %.2f\n" % (time.time() - tic, time.time() - start))

        # save models
        if (k + 1) % args.save_every == 0 and args.train:
            print("save model... ")
            for model in models:
                model.save(savedir, k)
