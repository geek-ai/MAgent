"""
First demo, show the usage of API
"""

import magent
from magent.builtin.tf_model import DeepQNetwork
from magent.builtin.rule_model import RandomActor

if __name__ == "__main__":
    map_size = 100

    # init the game "double attack"  (config file are stored in python/magent/builtin/config/)
    env = magent.GridWorld("double_attack", map_size=map_size)
    env.set_render_dir("build/render")

    # get group handles
    deer_handle, tiger_handle = env.get_handles()

    # init env and agents
    env.reset()
    env.add_agents(deer_handle,  method="random", n=map_size * map_size * 0.05)
    env.add_agents(tiger_handle, method="random", n=map_size * map_size * 0.02)

    # init two models
    model1 = RandomActor(env, deer_handle, "deer")
    model2 = DeepQNetwork(env, tiger_handle, "tiger")

    # load trained model
    model2.load("data/demo_model", 1)

    done = False
    step_ct = 0
    while not done:
        # take action for deers
        obs_1 = env.get_observation(deer_handle)
        ids_1 = env.get_agent_id(deer_handle)
        acts_1 = model1.infer_action(obs_1, ids_1)
        env.set_action(deer_handle, acts_1)

        # take action for tigers
        obs_2  = env.get_observation(tiger_handle)
        ids_2  = env.get_agent_id(tiger_handle)
        acts_2 = model2.infer_action(obs_2, ids_1)
        env.set_action(tiger_handle, acts_2)

        # simulate one step
        done = env.step()
        reward = env.get_reward(tiger_handle)

        # render
        env.render()

        # clear dead agents
        env.clear_dead()

        # print info
        if step_ct % 10 == 0:
            print("step %d\t deer num: %d\t tiger num: %d\t tiger reward %d" %
                  (step_ct, env.get_num(deer_handle),
                   env.get_num(tiger_handle), sum(reward)))

        step_ct += 1
