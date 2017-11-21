import argparse
import time
import logging as log

import magent
from magent.builtin.tf_model import DeepQNetwork


def load_config(map_size):
    gw = magent.gridworld
    cfg = gw.Config()

    cfg.set({"map_width": map_size, "map_height": map_size})

    small = cfg.register_agent_type(
        "small",
        {'width': 1, 'length': 1, 'hp': 1, 'speed': 1,
         'view_range': gw.CircleRange(5), 'attack_range': gw.CircleRange(0),
         'step_reward': -1,
         })

    small = cfg.add_group(small)

    a = gw.AgentSymbol(small, index='any')
    in_line = gw.Event(a, 'align')
    cfg.add_reward_rule(in_line, receiver=a, value="auto")

    return cfg


def generate_map(env, map_size, handles):
    env.add_agents(handles[0], method="random", n=map_size * 4 - 4)


def play_a_round(env, map_size, handles, models, print_every, train_id=0, render=False, eps=None):
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
    total_reward = 0

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
        rewards = env.get_reward(handles[train_id])
        step_reward = sum(rewards)
        total_reward += step_reward
        if train_id != -1:
            alives = env.get_alive(handles[train_id])
            sample_buffer.record_step(ids[train_id], obs[train_id], acts[train_id], rewards, alives)

        # render
        if render:
            env.render()

        # clear dead agents
        env.clear_dead()

        if step_ct % print_every == 0:
            print("step %3d,  nums: %s reward: %s,  total_reward: %s " %
                  (step_ct, nums, round(step_reward, 2), round(total_reward, 2)))
        step_ct += 1

        if step_ct > 500:
            break

    sample_time = time.time() - start_time
    print("steps: %d,  total time: %.2f,  step average %.2f" % (step_ct, sample_time, sample_time / step_ct))

    # train
    total_loss, value = 0, 0
    if train_id != -1:
        print("===== train =====")
        start_time = time.time()
        total_loss, value = models[0].train(sample_buffer, 500)

        train_time = time.time() - start_time
        print("train_time %.2f" % train_time)

    return round(total_loss, 2), round(total_reward, 2), round(value, 2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--render_every", type=int, default=10)
    parser.add_argument("--n_round", type=int, default=1000)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--load_from", type=int)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--map_size", type=int, default=100)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--name", type=str, default="align")
    parser.add_argument("--eval", action="store_true")
    args = parser.parse_args()

    # set logger
    log.basicConfig(level=log.INFO, filename=args.name + '.log')
    console = log.StreamHandler()
    console.setLevel(log.INFO)
    log.getLogger('').addHandler(console)

    # init the game
    env = magent.GridWorld(load_config(args.map_size))
    env.set_render_dir("build/render")

    # two groups of agents
    handles = env.get_handles()

    # sample eval observation set
    if args.eval:
        print("sample eval set...")
        env.reset()
        generate_map(env, args.map_size, handles)
        eval_obs = magent.utility.sample_observation(env, handles, 2048, 500)
    else:
        eval_obs = [None]

    # load models
    models = []
    models.append(DeepQNetwork(env, handles[0], args.name,
                               batch_size=512, memory_size=2 ** 19,
                               target_update=1000, train_freq=6, eval_obs=eval_obs[0]))

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
    print("view_shape", env.get_view_space(handles[0]))
    print("feature_size", env.get_feature_space(handles[0]))

    # play
    train_id = 0 if args.train else -1
    for k in range(start_from, start_from + args.n_round):
        start_time = time.time()
        eps = magent.utility.piecewise_decay(k, [0, 200, 400], [1, 0.1, 0.05]) if not args.greedy else 0

        loss, reward, value = play_a_round(env, args.map_size, handles, models,
                                           train_id=train_id, print_every=250,
                                           render=args.render or (k+1) % args.render_every == 0,
                                           eps=eps)  # for e-greedy
        log.info("round %d\t loss: %s\t reward: %s\t value: %s" % (k, loss, reward, value))
        print("round time %.2f\n" % (time.time() - start_time))

        # save models
        if (k + 1) % args.save_every == 0 and args.train:
            print("save model... ")
            for model in models:
                model.save(savedir, k)
