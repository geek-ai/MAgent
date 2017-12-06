"""
Pursuit: predators get reward when they attack prey.
"""

import argparse
import time
import logging as log

import numpy as np

import magent
from magent.builtin.tf_model import DeepQNetwork


def play_a_round(env, map_size, handles, models, print_every, train=True, render=False, eps=None):
    env.reset()

    env.add_walls(method="random", n=map_size * map_size * 0.03)
    env.add_agents(handles[0], method="random", n=map_size * map_size * 0.0125)
    env.add_agents(handles[1], method="random", n=map_size * map_size * 0.025)

    step_ct = 0
    done = False

    n = len(handles)
    obs  = [[] for _ in range(n)]
    ids  = [[] for _ in range(n)]
    acts = [[] for _ in range(n)]
    nums = [env.get_num(handle) for handle in handles]
    total_reward = [0 for _ in range(n)]

    print("===== sample =====")
    print("eps %s number %s" % (eps, nums))
    start_time = time.time()
    while not done:
        # take actions for every model
        for i in range(n):
            obs[i] = env.get_observation(handles[i])
            ids[i] = env.get_agent_id(handles[i])
            # let models infer action in parallel (non-blocking)
            models[i].infer_action(obs[i], ids[i], 'e_greedy', eps, block=False)
        for i in range(n):
            acts[i] = models[i].fetch_action()  # fetch actions (blocking)
            env.set_action(handles[i], acts[i])

        # simulate one step
        done = env.step()

        # sample
        step_reward = []
        for i in range(n):
            rewards = env.get_reward(handles[i])
            if train:
                alives  = env.get_alive(handles[i])
                # store samples in replay buffer (non-blocking)
                models[i].sample_step(rewards, alives, block=False)
            s = sum(rewards)
            step_reward.append(s)
            total_reward[i] += s

        # render
        if render:
            env.render()

        # clear dead agents
        env.clear_dead()

        # check 'done' returned by 'sample' command
        if train:
            for model in models:
                model.check_done()

        if step_ct % print_every == 0:
            print("step %3d,  reward: %s,  total_reward: %s " %
                  (step_ct, np.around(step_reward, 2), np.around(total_reward, 2)))
        step_ct += 1
        if step_ct > 250:
            break

    sample_time = time.time() - start_time
    print("steps: %d,  total time: %.2f,  step average %.2f" % (step_ct, sample_time, sample_time / step_ct))

    # train
    total_loss, value = [0 for _ in range(n)], [0 for _ in range(n)]
    if train:
        print("===== train =====")
        start_time = time.time()

        # train models in parallel
        for i in range(n):
            models[i].train(print_every=2000, block=False)
        for i in range(n):
            total_loss[i], value[i] = models[i].fetch_train()

        train_time = time.time() - start_time
        print("train_time %.2f" % train_time)

    return magent.round(total_loss), magent.round(total_reward), magent.round(value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_every", type=int, default=2)
    parser.add_argument("--render_every", type=int, default=10)
    parser.add_argument("--n_round", type=int, default=500)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--load_from", type=int)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--map_size", type=int, default=1000)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--name", type=str, default="pursuit")
    args = parser.parse_args()

    # set logger
    magent.utility.init_logger(args.name)

    # init the game
    env = magent.GridWorld("pursuit", map_size=args.map_size)
    env.set_render_dir("build/render")

    # two groups of agents
    handles = env.get_handles()

    # load models
    names = ["predator", "prey"]
    models = []

    for i in range(len(names)):
        models.append(magent.ProcessingModel(
            env, handles[i], names[i], 20000+i, 4000, DeepQNetwork,
            batch_size=512, memory_size=2 ** 22,
            target_update=1000, train_freq=4
        ))

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
        eps = magent.utility.piecewise_decay(k, [0, 200, 400], [1, 0.2, 0.05]) if not args.greedy else 0

        loss, reward, value = play_a_round(env, args.map_size, handles, models,
                                           print_every=50, train=args.train,
                                           render=args.render or (k+1) % args.render_every == 0,
                                           eps=eps)  # for e-greedy
        log.info("round %d\t loss: %s\t reward: %s\t value: %s" % (k, loss, reward, value))
        print("round time %.2f  total time %.2f\n" % (time.time() - tic, time.time() - start))

        if (k + 1) % args.save_every == 0 and args.train:
            print("save model... ")
            for model in models:
                model.save(savedir, k)

    # send quit command
    for model in models:
        model.quit()
