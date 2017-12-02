"""let saved models to play tournament"""

import os
import numpy as np
import time
import re
import math

import magent
from magent.builtin.tf_model import DeepQNetwork
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def play(env, handles, models, map_size, leftID, rightID, eps=0.05):
    env.reset()

    # generate map
    width = height = map_size
    init_num = map_size * map_size * 0.04
    gap = 3

    # left
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

    step_ct = 0
    done = False

    n = 2
    obs  = [[] for _ in range(n)]
    ids  = [[] for _ in range(n)]
    acts = [[] for _ in range(n)]
    nums = [env.get_num(handle) for handle in handles]

    while not done:
        # take actions for every model
        for i in range(n):
            obs[i] = env.get_observation(handles[i])
            ids[i] = env.get_agent_id(handles[i])
            models[i].infer_action(obs[i], ids[i], 'e_greedy', eps, block=False)
        for i in range(n):
            acts[i] = models[i].fetch_action()
            env.set_action(handles[i], acts[i])

        done = env.step()
        nums = [env.get_num(handle) for handle in handles]
        env.clear_dead()

        step_ct += 1
        if step_ct > 550:
            break

    return nums


def extract_model_names(savedir, name, model_class, begin=0, pick_every=4):
    if model_class is DeepQNetwork:
        prefix = 'tfdqn'
    pattern = re.compile(prefix + '_(\d*).meta')

    ret = []
    for path in os.listdir(os.path.join(savedir, name)):
        match = pattern.match(path)
        if match and int(match.group(1)) > begin:
            ret.append((savedir, name, int(match.group(1)), model_class))

    ret.sort(key=lambda x: x[2])
    ret = [ret[i] for i in range(0, len(ret), pick_every)]

    return ret


if __name__ == '__main__':
    map_size = 125
    env = magent.GridWorld("battle", map_size=map_size)
    env.set_render_dir("build/render")

    # scan file names
    model_name = extract_model_names('save_model', 'battle', DeepQNetwork, begin=0, pick_every=5)

    print("total models = %d" % len(model_name))
    print("models", [x[:-1] for x in model_name])
    handles = env.get_handles()

    def play_wrapper(model_names, n_rounds):
        time_stamp = time.time()

        models = []
        for i, item in enumerate(model_names):
            models.append(magent.ProcessingModel(env, handles[i], item[1], 0, item[-1]))

        for i, item in enumerate(model_names):
            models[i].load(item[0], item[2])

        leftID, rightID = 0, 1
        result = 0
        total_num = np.zeros(2)
        for _ in range(n_rounds):
            round_num = play(env, handles, models, map_size, leftID, rightID)
            total_num += round_num
            leftID, rightID = rightID, leftID
            result += 1 if round_num[0] > round_num[1] else 0
        result = 1.0 * result

        for model in models:
            model.quit()

        return result / n_rounds, total_num / n_rounds, time.time() - time_stamp

    detail_file = open("detail.log", "w")
    winrate_file = open("win_rate.log", "w")

    rate = [[0.0 for j in range(len(model_name))] for i in range(len(model_name))]
    for i in range(len(model_name)):
        for j in range(i+1, len(model_name)):
            rate[i][j], nums, elapsed = play_wrapper([model_name[i], model_name[j]], 6)
            rate[j][i] = 1.0 - rate[i][j]
            round_res = ("model1: %s\t model2: %s\t rate: %.2f\t num: %s\t elapsed: %.2f" %
                        (model_name[i][:-1], model_name[j][:-1], rate[i][j], list(nums), elapsed))
            print(round_res)
            detail_file.write(round_res + "\n")

        winrate_file.write("model: %s\twin rate: %.2f\n" % (model_name[i],
                                                            1.0 * sum(np.asarray(rate[i])) / (len(model_name) - 1)))

        detail_file.flush()
        winrate_file.flush()

