"""test one million random agents"""

import time

import magent.gridworld as gw
from magent.builtin.rule_model import RandomActor


def load_forest():
    cfg = gw.Config()

    cfg.set({"map_width": 10000, "map_height": 10000})
    cfg.set({"minimap_mode": 0})
    cfg.set({"turn_mode": 0})
    cfg.set({"embedding_size": 8})

    deer = cfg.register_agent_type(
        name="deer",
        attr={'width': 1, 'length': 1, 'hp': 10, 'speed': 1,
              'view_range': gw.CircleRange(7), 'attack_range': gw.CircleRange(0),
              'damage': 0, 'step_recover': 0,   'kill_supply': 8})

    tiger = cfg.register_agent_type(
        name="tiger",
        attr={'width': 1, 'length': 1, 'hp': 10, 'speed': 1,
              'view_range': gw.CircleRange(6), 'attack_range': gw.CircleRange(2),
              'damage': 10, 'step_recover': -0.01,   'kill_supply': 0})

    deer_g = cfg.add_group(agent_type=deer)
    tiger_g = cfg.add_group(agent_type=tiger)

    '''
    a = gw.AgentSymbol(group=deer_g, index='any')
    b = gw.AgentSymbol(group=tiger_g, index='any')

    a_kill = gw.Event(a, 'kill', b)
    b_kill = gw.Event(b, 'kill', a)

    cfg.add_reward_rule(on=a_kill, receiver=[a, b], value=[1, -1])
    cfg.add_reward_rule(on=b_kill, receiver=[a, b], value=[-1, 1])
    '''

    return cfg


def measure_time(msg, func, *args, **kwargs):
    start_time = time.time()
    ret = func(*args, **kwargs)
    print("%-11s %.3f" % (msg, time.time() - start_time))
    return ret

if __name__ == "__main__":
    n_step = 10

    # init the game "forest" (or "battle" here)
    env = gw.GridWorld(load_forest())
    env.reset()

    # add two groups of animals
    deer_handle, tiger_handle = env.get_handles()

    env.add_agents(deer_handle,  method="random", n=1000000)
    env.add_agents(tiger_handle, method="random", n=1000000)

    # init two models
    model1 = RandomActor(env, deer_handle)
    model2 = RandomActor(env, tiger_handle)

    total_reward = 0

    print(env.get_view_space(deer_handle))
    print(env.get_view_space(tiger_handle))

    for i in range(n_step):
        print("===== step %d =====" % i)
        start_time = time.time()

        obs_1 = measure_time("get obs 1", env.get_observation, deer_handle)
        acts_1 = measure_time("infer act 1", model1.infer_action, obs_1)
        measure_time("set act 1", env.set_action, deer_handle, acts_1)

        obs_2 = measure_time("get obs 2", env.get_observation, tiger_handle)
        acts_2 = measure_time("infer act 2", model2.infer_action, obs_2)
        measure_time("set act 2", env.set_action, tiger_handle, acts_2)

        # simulate one step
        done = measure_time("step", env.step)

        # update DQN
        rewards = measure_time("get reward", env.get_reward, tiger_handle)
        total_reward += sum(rewards)
        measure_time("clear", env.clear_dead)

        print("all time: %.2f\n" % (time.time() - start_time))
        # print info
        print("number of deer: %d"  % env.get_num(deer_handle))
        print("number of tiger: %d" % env.get_num(tiger_handle))
        print("total reward: %d" % total_reward)

        if done:
            print("game over")
            break

