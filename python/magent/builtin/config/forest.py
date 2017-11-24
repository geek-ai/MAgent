""" tigers eat deer to get health point and reward"""

import magent


def get_config(map_size):
    gw = magent.gridworld

    cfg = gw.Config()

    cfg.set({"map_width": map_size, "map_height": map_size})
    cfg.set({"embedding_size": 10})

    deer = cfg.register_agent_type(
        "deer",
        {'width': 1, 'length': 1, 'hp': 5, 'speed': 1,
         'view_range': gw.CircleRange(1), 'attack_range': gw.CircleRange(0),
         'damage': 0, 'step_recover': 0.2,
         'food_supply': 0, 'kill_supply': 8,
         })

    tiger = cfg.register_agent_type(
        "tiger",
        {'width': 1, 'length': 1, 'hp': 10, 'speed': 1,
         'view_range': gw.CircleRange(4), 'attack_range': gw.CircleRange(1),
         'damage': 3, 'step_recover': -0.5,
         'food_supply': 0, 'kill_supply': 0,
         'step_reward': 1, 'attack_penalty': -0.1,
         })

    deer_group  = cfg.add_group(deer)
    tiger_group = cfg.add_group(tiger)

    return cfg
