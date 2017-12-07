"""
Train a model to against existing benchmark
"""

import argparse
import time
import os
import logging as log
import math

import numpy as np

import magent
from magent.builtin.rule_model import RandomActor


def generate_map(env, map_size, handles):
    width = height = map_size
    init_num = map_size * map_size * 0.04

    gap = 3
    leftID, rightID = 0, 1

    # add left square of agents
    n = init_num
    side = int(math.sqrt(n)) * 2
    pos = []
    for x in range(width//2 - gap - side, width//2 - gap - side + side, 2):
        for y in range((height - side)//2, (height - side)//2 + side, 2):
            pos.append([x, y, 0])
    env.add_agents(handles[leftID], method="custom", pos=pos)

    # add right square of agents
    n = init_num
    side = int(math.sqrt(n)) * 2
    pos = []
    for x in range(width//2 + gap, width//2 + gap + side, 2):
        for y in range((height - side)//2, (height - side)//2 + side, 2):
            pos.append([x, y, 0])
    env.add_agents(handles[rightID], method="custom", pos=pos)


def play_a_round(env, map_size, handles, models, print_every, eps, step_batch_size=None, train=True,
                 train_id=1, render=False):
    """play a round of game"""
    env.reset()
    generate_map(env, map_size, handles)

    step_ct = 0
    done = False

    n = len(handles)
    obs  = [[] for _ in range(n)]
    ids  = [[] for _ in range(n)]
    acts = [[] for _ in range(n)]
    nums = [env.get_num(handle) for handle in handles]
    total_reward = [0 for _ in range(n)]
    n_transition = 0
    pos_reward_num = 0
    total_loss, value = 0, 0

    print("===== sample =====")
    print("eps %s number %s" % (eps, nums))
    start_time = time.time()
    while not done:
        # take actions for every model
        for i in range(n):
            obs[i] = env.get_observation(handles[i])
            ids[i] = env.get_agent_id(handles[i])
            # let models infer action in parallel (non-blocking)
            models[i].infer_action(obs[i], ids[i], 'e_greedy', eps[i], block=False)

        for i in range(n):
            acts[i] = models[i].fetch_action()  # fetch actions (blocking)
            env.set_action(handles[i], acts[i])

        # simulate one step
        done = env.step()

        # sample
        step_reward = []
        for i in range(n):
            rewards = env.get_reward(handles[i])
            if train and i == train_id:
                alives = env.get_alive(handles[train_id])
                # store samples in replay buffer (non-blocking)
                models[train_id].sample_step(rewards, alives, block=False)
                pos_reward_num += len(rewards[rewards > 0])
            s = sum(rewards)
            step_reward.append(s)
            total_reward[i] += s

        # render
        if render:
            env.render()

        # stat info
        nums = [env.get_num(handle) for handle in handles]
        n_transition += nums[train_id]

        # clear dead agents
        env.clear_dead()

        # check return message of previous called non-blocking function sample_step()
        if train:
            models[train_id].check_done()

        if step_ct % print_every == 0:
            print("step %3d,  nums: %s reward: %s,  total_reward: %s, pos_rewards %d" %
                  (step_ct, nums, np.around(step_reward, 2), np.around(total_reward, 2),
                      pos_reward_num))
        step_ct += 1
        if step_ct > args.n_step:
            break

        if step_batch_size and n_transition > step_batch_size and train:
            total_loss, value = models[train_id].train(500)
            n_transition = 0

    sample_time = time.time() - start_time
    print("steps: %d,  total time: %.2f,  step average %.2f" % (step_ct, sample_time, sample_time / step_ct))

    # train
    if train:
        print("===== train =====")
        start_time = time.time()
        total_loss, value = models[train_id].train(500)
        train_time = time.time() - start_time
        print("train_time %.2f" % train_time)

    return magent.round(total_loss), nums, magent.round(total_reward), magent.round(value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--render_every", type=int, default=10)
    parser.add_argument("--n_round", type=int, default=600)
    parser.add_argument("--n_step", type=int, default=550)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--load_from", type=int)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--map_size", type=int, default=125)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--name", type=str, default="against")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--opponent", type=int, default=0)
    parser.add_argument('--alg', default='dqn', choices=['dqn', 'drqn', 'a2c'])
    args = parser.parse_args()

    # download opponent model
    magent.utility.check_model('against')

    # set logger
    magent.utility.init_logger(args.name)

    # init the game
    env = magent.GridWorld("battle", map_size=args.map_size)
    env.set_render_dir("build/render")

    # two groups of agents
    handles = env.get_handles()

    # sample eval observation set
    if args.eval:
        print("sample eval set...")
        env.reset()
        generate_map(env, args.map_size, handles)
        eval_obs = magent.utility.sample_observation(env, handles, n_obs=2048, step=500)
    else:
        eval_obs = [None, None]

    # init models
    names = [args.name + "-a", "battle"]
    batch_size = 512
    unroll_step = 16
    train_freq = 5

    models = []

    # load opponent
    if args.opponent >= 0:
        from magent.builtin.tf_model import DeepQNetwork
        models.append(magent.ProcessingModel(env, handles[1], names[1], 20000, 0, DeepQNetwork))
        models[0].load("data/battle_model", args.opponent)
    else:
        models.append(magent.ProcessingModel(env, handles[1], names[1], 20000, 0, RandomActor))

    # load our model
    if args.alg == 'dqn':
        from magent.builtin.tf_model import DeepQNetwork
        models.append(magent.ProcessingModel(env, handles[0], names[0], 20001, 1000, DeepQNetwork,
                                   batch_size=batch_size,
                                   learning_rate=3e-4,
                                   memory_size=2 ** 20, train_freq=train_freq, eval_obs=eval_obs[0]))
                                   
        step_batch_size = None
    elif args.alg == 'drqn':
        from magent.builtin.tf_model import DeepRecurrentQNetwork
        models.append(magent.ProcessingModel(env, handles[0], names[0], 20001, 1000, DeepRecurrentQNetwork,
                                   batch_size=batch_size/unroll_step, unroll_step=unroll_step,
                                   learning_rate=3e-4,
                                   memory_size=4 * 625, train_freq=train_freq, eval_obs=eval_obs[0]))
        step_batch_size = None
    elif args.alg == 'a2c':
        from magent.builtin.mx_model import AdvantageActorCritic
        step_batch_size = 10 * args.map_size * args.map_size * 0.04
        models.append(magent.ProcessingModel(env, handles[0], names[0], 20001, 1000, AdvantageActorCritic,
                                             learning_rate=1e-3))

    # load if
    savedir = 'save_model'
    if args.load_from is not None:
        start_from = args.load_from
        print("load ... %d" % start_from)
        models[0].load(savedir, start_from)
    else:
        start_from = 0

    # print debug info
    print(args)
    print("view_size", env.get_view_space(handles[0]))
    print("feature_size", env.get_feature_space(handles[0]))

    # play
    start = time.time()
    for k in range(start_from, start_from + args.n_round):
        tic = time.time()
        start = 1 if args.opponent != -1 else 0.1
        train_eps = magent.utility.piecewise_decay(k, [0, 100, 250], [start, 0.1, 0.05]) if not args.greedy else 0
        opponent_eps = train_eps if k < 100 else 0.05  # can use curriculum learning in first 100 steps

        loss, num, reward, value = play_a_round(env, args.map_size, handles, models,
                                                eps=[opponent_eps, train_eps], step_batch_size=step_batch_size,
                                                train=args.train,
                                                print_every=50,
                                                render=args.render or (k+1) % args.render_every == 0)  # for e-greedy

        log.info("round %d\t loss: %s\t num: %s\t reward: %s\t value: %s" % (k, loss, num, reward, value))
        print("round time %.2f  total time %.2f\n" % (time.time() - tic, time.time() - start))

        # save models
        if (k + 1) % args.save_every == 0 and args.train:
            print("save model... ")
            if not os.path.exists(savedir):
                os.mkdir(savedir)
            for model in models:
                model.save(savedir, k)

    # close model processing
    for model in models:
        model.quit()
