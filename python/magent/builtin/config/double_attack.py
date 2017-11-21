"""
A cooperation game, tigers must attack a same deer simultaneously to get reward
"""

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
         'step_recover': 0.2,
         'kill_supply': 8,
         })

    tiger = cfg.register_agent_type(
        "tiger",
        {'width': 1, 'length': 1, 'hp': 10, 'speed': 1,
         'view_range': gw.CircleRange(4), 'attack_range': gw.CircleRange(1),
         'damage': 1, 'step_recover': -0.2,
         })

    deer_group  = cfg.add_group(deer)
    tiger_group = cfg.add_group(tiger)

    a = gw.AgentSymbol(tiger_group, index='any')
    b = gw.AgentSymbol(tiger_group, index='any')
    c = gw.AgentSymbol(deer_group,  index='any')

    # tigers get reward when they attack a deer simultaneously
    e1 = gw.Event(a, 'attack', c)
    e2 = gw.Event(b, 'attack', c)
    cfg.add_reward_rule(e1 & e2, receiver=[a, b], value=[1, 1])

    return cfg
