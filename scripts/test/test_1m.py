"""test one million random agents"""

import time

import magent
import os
import math
import argparse
from magent.builtin.rule_model import RandomActor

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def load_forest(map_size):
    gw = magent.gridworld
    cfg = gw.Config()

    cfg.set({"map_width": map_size, "map_height": map_size})

    predator = cfg.register_agent_type(
        "predator",
        {
            'width': 2, 'length': 2, 'hp': 1, 'speed': 1,
            'view_range': gw.CircleRange(5), 'attack_range': gw.CircleRange(2),
            'attack_penalty': -0.2
        })

    prey = cfg.register_agent_type(
        "prey",
        {
            'width': 1, 'length': 1, 'hp': 1, 'speed': 1.5,
            'view_range': gw.CircleRange(4), 'attack_range': gw.CircleRange(0)
        })

    predator_group  = cfg.add_group(predator)
    prey_group = cfg.add_group(prey)

    a = gw.AgentSymbol(predator_group, index='any')
    b = gw.AgentSymbol(prey_group, index='any')

    cfg.add_reward_rule(gw.Event(a, 'attack', b), receiver=[a, b], value=[1, -1])

    return cfg


def measure_time(msg, func, *args, **kwargs):
    start_time = time.time()
    ret = func(*args, **kwargs)
    print("%-11s %.5f" % (msg, time.time() - start_time))
    return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_step", type=int, default=20)
    parser.add_argument("--agent_number", type=int, default=1000)
    parser.add_argument("--num_gpu", type=int, default=0)
    parser.add_argument('--frame', default='tf', choices=['tf', 'mx'])
    args = parser.parse_args()

    n_step = args.n_step
    agent_number = args.agent_number
    skip = 20    # warm up steps
    n_step += skip

    # init the game "forest" (or "battle" here)
    env = magent.GridWorld(load_forest(int(math.sqrt(agent_number * 20))))
    env.reset()

    # add two groups of animals
    deer_handle, tiger_handle = env.get_handles()

    env.add_walls(method="random", n=agent_number / 10)
    env.add_agents(deer_handle,  method="random", n=agent_number / 2)
    env.add_agents(tiger_handle, method="random", n=agent_number / 2)

    # init two models
    if args.num_gpu == 0:
        model1 = RandomActor(env, deer_handle, "deer")
        model2 = RandomActor(env, tiger_handle, "tiger")
    else:
        if args.frame == 'tf':
            from magent.builtin.tf_model import DeepQNetwork
        else:
            from magent.builtin.mx_model import DeepQNetwork
        model1 = DeepQNetwork(env, deer_handle, "deer", num_gpu=args.num_gpu, infer_batch_size=100000)
        model2 = DeepQNetwork(env, tiger_handle, "tiger", num_gpu=args.num_gpu, infer_batch_size=100000)

    total_reward = 0

    print(env.get_view_space(deer_handle))
    print(env.get_view_space(tiger_handle))

    total_time = 0

    for i in range(n_step):
        print("===== step %d =====" % i)
        start_time = time.time()

        obs_1 = measure_time("get obs 1", env.get_observation, deer_handle)
        acts_1 = measure_time("infer act 1", model1.infer_action, obs_1, None)
        measure_time("set act 1", env.set_action, deer_handle, acts_1)

        obs_2 = measure_time("get obs 2", env.get_observation, tiger_handle)
        acts_2 = measure_time("infer act 2", model2.infer_action, obs_2, None)
        measure_time("set act 2", env.set_action, tiger_handle, acts_2)

        # simulate one step
        done = measure_time("step", env.step)

        # get reward
        rewards = measure_time("get reward", env.get_reward, tiger_handle)
        total_reward += sum(rewards)
        measure_time("clear", env.clear_dead)

        step_time = time.time() - start_time
        if i >= skip:
            total_time += step_time
        print("all time: %.2f\n" % (step_time))

        # print info
        print("number of deer: %d"  % env.get_num(deer_handle))
        print("number of tiger: %d" % env.get_num(tiger_handle))
        print("total reward: %d" % total_reward)

        if done:
            print("game over")
            break

    print("FPS", (n_step - skip) / total_time)
