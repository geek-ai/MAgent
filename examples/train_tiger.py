"""
Double attack, tigers get reward when they attack a same deer
"""

import argparse
import time
import logging as log

import numpy as np

import magent
from magent.builtin.rule_model import RandomActor


def generate_map(env, map_size, handles):
    env.add_walls(method="random", n=map_size*map_size*0.04)
    env.add_agents(handles[0], method="random", n=map_size*map_size*0.05)
    env.add_agents(handles[1], method="random", n=map_size*map_size*0.01)


def play_a_round(env, map_size, handles, models, print_every, train_id=1, step_batch_size=None, render=False, eps=None):
    env.reset()
    generate_map(env, map_size, handles)

    step_ct = 0
    total_reward = 0
    done = False
    total_loss = value = 0

    n = len(handles)
    obs  = [[] for _ in range(n)]
    ids  = [[] for _ in range(n)]
    acts = [[] for _ in range(n)]
    nums = [0 for _ in range(n)]
    sample_buffer = magent.utility.EpisodesBuffer(10000)
    n_transition = 0

    print("===== sample =====")
    print("eps %s" % eps)
    start_time = time.time()
    while not done:
        # take actions for every model
        for i in range(n):
            if i == 0:
                temp_num = env.get_num(handles[i])
                obs[i] = (np.empty(temp_num), np.empty(temp_num))
            else:
                obs[i] = env.get_observation(handles[i])
            ids[i] = env.get_agent_id(handles[i])
            acts[i] = models[i].infer_action(obs[i], ids[i], policy='e_greedy', eps=eps)
            env.set_action(handles[i], acts[i])

        # simulate one step
        done = env.step()

        # sample
        reward = 0
        if train_id != -1:
            rewards = env.get_reward(handles[train_id])
            alives  = env.get_alive(handles[train_id])
            total_reward += sum(rewards)
            sample_buffer.record_step(ids[train_id], obs[train_id], acts[train_id], rewards, alives)
            reward = sum(rewards)

        # render
        if render:
            env.render()

        # clear dead agents
        env.clear_dead()

        # stats info
        for i in range(n):
            nums[i] = env.get_num(handles[i])
        n_transition += nums[train_id]

        if step_ct % print_every == 0:
            print("step %3d,  deer: %5d,  tiger: %5d,  train_id: %d,  reward: %.2f,  total_reward: %.2f " %
                  (step_ct, nums[0], nums[1], train_id, reward, total_reward))
        step_ct += 1
        if step_ct > 1000:
            break

        if step_batch_size and n_transition > step_batch_size and train_id != -1:
            total_loss, value = models[train_id].train(sample_buffer, 500)
            sample_buffer.reset()
            n_transition = 0


    sample_time = time.time() - start_time
    print("steps: %d, total time: %.2f, step average %.2f" % (step_ct, sample_time, sample_time / step_ct))

    # train
    if train_id != -1:
        print("===== train =====")
        start_time = time.time()
        total_loss, value = models[train_id].train(sample_buffer)
        train_time = time.time() - start_time
        print("train_time %.2f" % train_time)

    return total_loss, total_reward, value


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--n_round", type=int, default=200)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--load_from", type=int)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--map_size", type=int, default=500)
    parser.add_argument("--name", type=str, default="tiger")
    parser.add_argument('--alg', default='dqn', choices=['dqn', 'drqn', 'a2c'])
    args = parser.parse_args()

    # init the game
    env = magent.GridWorld("double_attack", map_size=args.map_size)
    env.set_render_dir("build/render")

    # two groups of animal
    deer_handle, tiger_handle = env.get_handles()

    # init two models
    models = [
        RandomActor(env, deer_handle, tiger_handle),
    ]

    batch_size = 512
    unroll     = 8

    if args.alg == 'dqn':
        from magent.builtin.tf_model import DeepQNetwork
        models.append(DeepQNetwork(env, tiger_handle, "tiger",
                                   batch_size=batch_size,
                                   memory_size=2 ** 20, learning_rate=4e-4))
        step_batch_size = None
    elif args.alg == 'drqn':
        from magent.builtin.tf_model import DeepRecurrentQNetwork
        models.append(DeepRecurrentQNetwork(env, tiger_handle, "tiger",
                                   batch_size=batch_size/unroll, unroll_step=unroll,
                                   memory_size=20000, learning_rate=4e-4))
        step_batch_size = None
    elif args.alg == 'a2c':
        from magent.builtin.mx_model import AdvantageActorCritic
        step_batch_size = int(10 * args.map_size * args.map_size*0.01)
        models.append(AdvantageActorCritic(env, tiger_handle, "tiger",
                                   batch_size=step_batch_size,
                                   learning_rate=1e-2))
    else:
        raise NotImplementedError

    # load if
    savedir = 'save_model'
    if args.load_from is not None:
        start_from = args.load_from
        print("load ... %d" % start_from)
        for model in models:
            model.load(savedir, start_from)
    else:
        start_from = 0

    # init logger
    magent.utility.init_logger(args.name)

    # print debug info
    print(args)
    print("view_size", env.get_view_space(tiger_handle))

    # play
    train_id = 1 if args.train else -1
    start = time.time()
    for k in range(start_from, start_from + args.n_round):
        tic = time.time()
        eps = magent.utility.linear_decay(k, 10, 0.1) if not args.greedy else 0
        loss, reward, value = play_a_round(env, args.map_size, [deer_handle, tiger_handle], models,
                                           step_batch_size=step_batch_size, train_id=train_id,
                                           print_every=40, render=args.render,
                                           eps=eps)

        log.info("round %d\t loss: %s\t reward: %s\t value: %s" % (k, loss, reward, value))
        print("round time %.2f  total time %.2f\n" % (time.time() - tic, time.time() - start))

        if (k + 1) % args.save_every == 0:
            print("save model... ")
            for model in models:
                model.save(savedir, k)
