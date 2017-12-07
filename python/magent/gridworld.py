"""gridworld interface"""
from __future__ import absolute_import

import ctypes
import os
import importlib

import numpy as np

from .c_lib import _LIB, as_float_c_array, as_int32_c_array
from .environment import Environment


class GridWorld(Environment):
    # constant
    OBS_INDEX_VIEW = 0
    OBS_INDEX_HP   = 1

    def __init__(self, config, **kwargs):
        """
        Parameters
        ----------
        config: str or Config Object
            if config is a string, then it is a name of builtin config,
                builtin config are stored in python/magent/builtin/config
                kwargs are the arguments to the config
            if config is a Config Object, then parameters are stored in that object
        """
        Environment.__init__(self)

        # if is str, load built in configuration
        if isinstance(config, str):
            # built-in config are stored in python/magent/builtin/config
            try:
                demo_game = importlib.import_module('magent.builtin.config.' + config)
                config = getattr(demo_game, 'get_config')(**kwargs)
            except AttributeError:
                raise BaseException('unknown built-in game "' + config + '"')

        # create new game
        game = ctypes.c_void_p()
        _LIB.env_new_game(ctypes.byref(game), b"GridWorld")
        self.game = game

        # set global configuration
        config_value_type = {
            'map_width': int, 'map_height': int,
            'food_mode': bool, 'turn_mode': bool, 'minimap_mode': bool,
            'revive_mode': bool, 'goal_mode': bool,
            'embedding_size': int,
            'render_dir': str,
        }

        for key in config.config_dict:
            value_type = config_value_type[key]
            if value_type is int:
                _LIB.env_config_game(self.game, key.encode("ascii"), ctypes.byref(ctypes.c_int(config.config_dict[key])))
            elif value_type is bool:
                _LIB.env_config_game(self.game, key.encode("ascii"), ctypes.byref(ctypes.c_bool(config.config_dict[key])))
            elif value_type is float:
                _LIB.env_config_game(self.game, key.encode("ascii"), ctypes.byref(ctypes.c_float(config.config_dict[key])))
            elif value_type is str:
                _LIB.env_config_game(self.game, key.encode("ascii"), ctypes.c_char_p(config.config_dict[key]))

        # register agent types
        for name in config.agent_type_dict:
            type_args = config.agent_type_dict[name]

            # special pre-process for view range and attack range
            for key in [x for x in type_args.keys()]:
                if key == "view_range":
                    val = type_args[key]
                    del type_args[key]
                    type_args["view_radius"] = val.radius
                    type_args["view_angle"]  = val.angle
                elif key == "attack_range":
                    val = type_args[key]
                    del type_args[key]
                    type_args["attack_radius"] = val.radius
                    type_args["attack_angle"]  = val.angle

            length = len(type_args)
            keys = (ctypes.c_char_p * length)(*[key.encode("ascii") for key in type_args.keys()])
            values = (ctypes.c_float * length)(*type_args.values())

            _LIB.gridworld_register_agent_type(self.game, name.encode("ascii"), length, keys, values)

        # serialize event expression, send to C++ engine
        self._serialize_event_exp(config)

        # init group handles
        self.group_handles = []
        for item in config.groups:
            handle = ctypes.c_int32()
            _LIB.gridworld_new_group(self.game, item.encode("ascii"), ctypes.byref(handle))
            self.group_handles.append(handle)

        # init observation buffer (for acceleration)
        self._init_obs_buf()

        # init view space, feature space, action space
        self.view_space = {}
        self.feature_space = {}
        self.action_space = {}
        buf = np.empty((3,), dtype=np.int32)
        for handle in self.group_handles:
            _LIB.env_get_info(self.game, handle, b"view_space",
                              buf.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
            self.view_space[handle.value] = (buf[0], buf[1], buf[2])
            _LIB.env_get_info(self.game, handle, b"feature_space",
                                  buf.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
            self.feature_space[handle.value] = (buf[0],)
            _LIB.env_get_info(self.game, handle, b"action_space",
                                  buf.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
            self.action_space[handle.value] = (buf[0],)

    def reset(self):
        """reset environment"""
        _LIB.env_reset(self.game)

    def add_walls(self, method, **kwargs):
        """add wall to environment

        Parameters
        ----------
        method: str
            can be 'random' or 'custom'
            if method is 'random', then kwargs["n"] is a int
            if method is 'custom', then kwargs["pos"] is a list of coordination

        Examples
        --------
        # add 1000 walls randomly
        >>> env.add_walls(method="random", n=1000)

        # add 3 walls to (1,2), (4,5) and (9, 8) in map
        >>> env.add_walls(method="custom", pos=[(1,2), (4,5), (9,8)])
        """
        # handle = -1 for walls
        kwargs["dir"] = 0
        self.add_agents(-1, method, **kwargs)

    # ====== AGENT ======
    def new_group(self, name):
        """register a new group into environment"""
        handle = ctypes.c_int32()
        _LIB.gridworld_new_group(self.game, ctypes.c_char_p(name.encode("ascii")), ctypes.byref(handle))
        return handle

    def add_agents(self, handle, method, **kwargs):
        """add agents to environment

        Parameters
        ----------
        handle: group handle
        method: str
            can be 'random' or 'custom'
            if method is 'random', then kwargs["n"] is a int
            if method is 'custom', then kwargs["pos"] is a list of coordination

        Examples
        --------
        # add 1000 walls randomly
        >>> env.add_agents(handle, method="random", n=1000)

        # add 3 agents to (1,2), (4,5) and (9, 8) in map
        >>> env.add_agents(handle, method="custom", pos=[(1,2), (4,5), (9,8)])
        """
        if method == "random":
            _LIB.gridworld_add_agents(self.game, handle, int(kwargs["n"]), b"random", 0, 0, 0)
        elif method == "custom":
            n = len(kwargs["pos"])
            pos = np.array(kwargs["pos"], dtype=np.int32)
            if len(pos) <= 0:
                return
            if pos.shape[1] == 3:  # if has dir
                xs, ys, dirs = pos[:, 0], pos[:, 1], pos[:, 2]
            else:                  # if do not has dir, use zero padding
                xs, ys, dirs = pos[:, 0], pos[:, 1], np.zeros((n,), dtype=np.int32)
            # copy again, to make these arrays continuous in memory
            xs, ys, dirs = np.array(xs), np.array(ys), np.array(dirs)
            _LIB.gridworld_add_agents(self.game, handle, n, b"custom", as_int32_c_array(xs),
                                      as_int32_c_array(ys), as_int32_c_array(dirs))
        elif method == "fill":
            x, y = kwargs["pos"][0], kwargs["pos"][1]
            width, height = kwargs["size"][0], kwargs["size"][1]
            dir = kwargs.get("dir", np.zeros_like(x))
            bind = np.array([x, y, width, height, dir], dtype=np.int32)
            _LIB.gridworld_add_agents(self.game, handle, 0,  b"fill", as_int32_c_array(bind),
                                      0, 0, 0)
        elif method == "maze":
            # TODO: implement maze add
            x_start, y_start, x_end, y_end = kwargs["pos"][0], kwargs["pos"][1], kwargs["pos"][2], kwargs["pos"][3]
            thick = kwargs["pos"][4]
            bind = np.array([x_start, y_start, x_end, y_end, thick], dtype=np.int32)
            _LIB.gridworld_add_agents(self.game, handle, 0, b"maze", as_int32_c_array(bind),
                                      0, 0, 0)
        else:
            print("Unknown type of position")
            exit(-1)

    # ====== RUN ======
    def _get_obs_buf(self, group, key, shape, dtype):
        """get buffer to receive observation from c++ engine"""
        obs_buf = self.obs_bufs[key]
        if group in obs_buf:
            ret = obs_buf[group]
            if shape != ret.shape:
                ret.resize(shape, refcheck=False)
        else:
            ret = obs_buf[group] = np.empty(shape=shape, dtype=dtype)

        return ret

    def _init_obs_buf(self):
        """init observation buffer"""
        self.obs_bufs = []
        self.obs_bufs.append({})
        self.obs_bufs.append({})

    def get_observation(self, handle):
        """ get observation of a whole group

        Parameters
        ----------
        handle : group handle

        Returns
        -------
        obs : tuple (views, features)
            views is a numpy array, whose shape is n * view_width * view_height * n_channel
            features is a numpy array, whose shape is n * feature_size
            for agent i, (views[i], features[i]) is its observation at this step
        """
        view_space = self.view_space[handle.value]
        feature_space = self.feature_space[handle.value]
        no = handle.value

        n = self.get_num(handle)
        view_buf = self._get_obs_buf(no, self.OBS_INDEX_VIEW, (n,) + view_space, np.float32)
        feature_buf = self._get_obs_buf(no, self.OBS_INDEX_HP, (n,) + feature_space, np.float32)

        bufs = (ctypes.POINTER(ctypes.c_float) * 2)()
        bufs[0] = as_float_c_array(view_buf)
        bufs[1] = as_float_c_array(feature_buf)
        _LIB.env_get_observation(self.game, handle, bufs)

        return view_buf, feature_buf

    def set_action(self, handle, actions):
        """ set actions for whole group

        Parameters
        ----------
        handle: group handle
        actions: numpy array
            the dtype of actions must be int32
        """
        assert isinstance(actions, np.ndarray)
        assert actions.dtype == np.int32
        _LIB.env_set_action(self.game, handle, actions.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))

    def step(self):
        """simulation one step after set actions

        Returns
        -------
        done: bool
            whether the game is done
        """
        done = ctypes.c_int32()
        _LIB.env_step(self.game, ctypes.byref(done))
        return bool(done)

    def get_reward(self, handle):
        """ get reward for a whole group

        Returns
        -------
        rewards: numpy array (float32)
            reward for all the agents in the group
        """
        n = self.get_num(handle)
        buf = np.empty((n,), dtype=np.float32)
        _LIB.env_get_reward(self.game, handle,
                            buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        return buf

    def clear_dead(self):
        """ clear dead agents in the engine
        must be called after step()
        """
        _LIB.gridworld_clear_dead(self.game)

    # ====== INFO ======
    def get_handles(self):
        """ get all group handles in the environment """
        return self.group_handles

    def get_num(self, handle):
        """ get the number of agents in a group"""
        num = ctypes.c_int32()
        _LIB.env_get_info(self.game, handle, b'num', ctypes.byref(num))
        return num.value

    def get_action_space(self, handle):
        """get action space

        Returns
        -------
        action_space : tuple
        """
        return self.action_space[handle.value]

    def get_view_space(self, handle):
        """get view space

        Returns
        -------
        view_space : tuple
        """
        return self.view_space[handle.value]

    def get_feature_space(self, handle):
        """ get feature space

        Returns
        -------
        feature_space : tuple
        """
        return self.feature_space[handle.value]

    def get_agent_id(self, handle):
        """ get agent id

        Returns
        -------
        ids : numpy array (int32)
            id of all the agents in the group
        """
        n = self.get_num(handle)
        buf = np.empty((n,), dtype=np.int32)
        _LIB.env_get_info(self.game, handle, b"id",
                          buf.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
        return buf

    def get_alive(self, handle):
        """ get alive status of agents in a group

        Returns
        -------
        alives: numpy array (bool)
            whether the agents are alive
        """
        n = self.get_num(handle)
        buf = np.empty((n,), dtype=np.bool)
        _LIB.env_get_info(self.game, handle, b"alive",
                          buf.ctypes.data_as(ctypes.POINTER(ctypes.c_bool)))
        return buf

    def get_pos(self, handle):
        """ get position of agents in a group

        Returns
        -------
        pos: numpy array (int)
            the shape of pos is (n, 2)
        """
        n = self.get_num(handle)
        buf = np.empty((n, 2), dtype=np.int32)
        _LIB.env_get_info(self.game, handle, b"pos",
                          buf.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
        return buf

    def get_mean_info(self, handle):
        """ deprecated """
        buf = np.empty(2 + self.action_space[handle.value][0], dtype=np.float32)
        _LIB.env_get_info(self.game, handle, b"mean_info",
                          buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        return buf

    def get_view2attack(self, handle):
        """ get a matrix with the same size of view_range,
            if element >= 0, then it means it is a attackable point, and the corresponding
                                    action number is the value of that element
        Returns
        -------
        attack_back: int
        buf: numpy array
            map attack action into view
        """
        size = self.get_view_space(handle)[0:2]
        buf = np.empty(size, dtype=np.int32)
        attack_base = ctypes.c_int32()
        _LIB.env_get_info(self.game, handle, b"view2attack",
                          buf.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
        _LIB.env_get_info(self.game, handle, b"attack_base",
                          ctypes.byref(attack_base))
        return attack_base.value, buf

    def get_global_minimap(self, height, width):
        """ compress global map into a minimap of given size
        Parameters
        ----------
        height: int
            the height of minimap
        width:  int
            the width of minimap

        Returns
        -------
        minimap : numpy array
            the shape (n_group + 1, height, width)
        """
        buf = np.empty((height, width, len(self.group_handles)), dtype=np.float32)
        buf[0, 0, 0] = height
        buf[0, 0, 1] = width
        _LIB.env_get_info(self.game, -1, b"global_minimap",
                          buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        return buf

    def set_seed(self, seed):
        """ set random seed of the engine"""
        _LIB.env_config_game(self.game, b"seed", ctypes.byref(ctypes.c_int(seed)))

    # ====== RENDER ======
    def set_render_dir(self, name):
        """ set directory to save render file"""
        if not os.path.exists(name):
            os.mkdir(name)
        _LIB.env_config_game(self.game, b"render_dir", name.encode("ascii"))

    def render(self):
        """ render a step """
        _LIB.env_render(self.game)

    def _get_groups_info(self):
        """ private method, for interactive application"""
        n = len(self.group_handles)
        buf = np.empty((n, 5), dtype=np.int32)
        _LIB.env_get_info(self.game, -1, b"groups_info",
                          buf.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
        return buf

    def _get_walls_info(self):
        """ private method, for interactive application"""
        n = 100 * 100
        buf = np.empty((n, 2), dtype=np.int32)
        _LIB.env_get_info(self.game, -1, b"walls_info",
                          buf.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
        n = buf[0, 0]  # the first line is the number of walls
        return buf[1:1+n]

    def _get_render_info(self, x_range, y_range):
        """ private method, for interactive application"""
        n = 0
        for handle in self.group_handles:
            n += self.get_num(handle)

        buf = np.empty((n+1, 4), dtype=np.int32)
        buf[0] = x_range[0], y_range[0], x_range[1], y_range[1]
        _LIB.env_get_info(self.game, -1, b"render_window_info",
                          buf.ctypes.data_as(ctypes.POINTER((ctypes.c_int32))))

        # the first line is for the number of agents in the window range
        info_line = buf[0]
        agent_ct, attack_event_ct = info_line[0], info_line[1]
        buf = buf[1:1 + info_line[0]]

        agent_info = {}
        for item in buf:
            agent_info[item[0]] = [item[1], item[2], item[3]]

        buf = np.empty((attack_event_ct, 3), dtype=np.int32)
        _LIB.env_get_info(self.game, -1, b"attack_event",
                          buf.ctypes.data_as(ctypes.POINTER((ctypes.c_int32))))
        attack_event = buf

        return agent_info, attack_event

    def __del__(self):
        _LIB.env_delete_game(self.game)

    # ====== SPECIAL RULE ======
    def set_goal(self, handle, method, *args, **kwargs):
        """ deprecated """
        if method == "random":
            _LIB.gridworld_set_goal(self.game, handle, b"random", 0, 0)
        else:
            raise NotImplementedError

    # ====== PRIVATE ======
    def _serialize_event_exp(self, config):
        """serialize event expression and sent them to game engine"""
        game = self.game

        # collect agent symbol
        symbol2int = {}
        config.symbol_ct = 0

        def collect_agent_symbol(node, config):
            for item in node.inputs:
                if isinstance(item, EventNode):
                    collect_agent_symbol(item, config)
                elif isinstance(item, AgentSymbol):
                    if item not in symbol2int:
                        symbol2int[item] = config.symbol_ct
                        config.symbol_ct += 1

        for rule in config.reward_rules:
            on = rule[0]
            receiver = rule[1]
            for symbol in receiver:
                if symbol not in symbol2int:
                    symbol2int[symbol] = config.symbol_ct
                    config.symbol_ct += 1
            collect_agent_symbol(on, config)

        # collect event node
        event2int = {}
        config.node_ct = 0

        def collect_event_node(node, config):
            if node not in event2int:
                event2int[node] = config.node_ct
                config.node_ct += 1
            for item in node.inputs:
                if isinstance(item, EventNode):
                    collect_event_node(item, config)

        for rule in config.reward_rules:
            collect_event_node(rule[0], config)

        # send to C++ engine
        for sym in symbol2int:
            no = symbol2int[sym]
            _LIB.gridworld_define_agent_symbol(game, no, sym.group, sym.index)

        for event in event2int:
            no = event2int[event]
            inputs = np.zeros_like(event.inputs, dtype=np.int32)
            for i, item in enumerate(event.inputs):
                if isinstance(item, EventNode):
                    inputs[i] = event2int[item]
                elif isinstance(item, AgentSymbol):
                    inputs[i] = symbol2int[item]
                else:
                    inputs[i] = item
            n_inputs = len(inputs)
            _LIB.gridworld_define_event_node(game, no, event.op, as_int32_c_array(inputs), n_inputs)

        for rule in config.reward_rules:
            # rule = [on, receiver, value, terminal]
            on = event2int[rule[0]]

            receiver = np.zeros_like(rule[1], dtype=np.int32)
            for i, item in enumerate(rule[1]):
                receiver[i] = symbol2int[item]
            if len(rule[2]) == 1 and rule[2][0] == 'auto':
                value = np.zeros(receiver, dtype=np.float32)
            else:
                value = np.array(rule[2], dtype=np.float32)
            n_receiver = len(receiver)
            _LIB.gridworld_add_reward_rule(game, on, as_int32_c_array(receiver),
                                           as_float_c_array(value), n_receiver, rule[3])


'''
the following classes are for reward description
'''
class EventNode:
    """an AST node of the event expression"""
    OP_AND = 0
    OP_OR  = 1
    OP_NOT = 2

    OP_KILL = 3
    OP_AT   = 4
    OP_IN   = 5
    OP_COLLIDE = 6
    OP_ATTACK  = 7
    OP_DIE  = 8
    OP_IN_A_LINE = 9
    OP_ALIGN = 10

    # can extend more operation below

    def __init__(self):
        # for non-leaf node
        self.op = None
        # for leaf node
        self.predicate = None

        self.inputs = []

    def __call__(self, subject, predicate, *args):
        node = EventNode()
        node.predicate = predicate
        if predicate == 'kill':
            node.op = EventNode.OP_KILL
            node.inputs = [subject, args[0]]
        elif predicate == 'at':
            node.op = EventNode.OP_AT
            coor = args[0]
            node.inputs = [subject, coor[0], coor[1]]
        elif predicate == 'in':
            node.op = EventNode.OP_IN
            coor = args[0]
            x1, y1 = min(coor[0][0], coor[1][0]), min(coor[0][1], coor[1][1])
            x2, y2 = max(coor[0][0], coor[1][0]), max(coor[0][1], coor[1][1])
            node.inputs = [subject, x1, y1, x2, y2]
        elif predicate == 'attack':
            node.op = EventNode.OP_ATTACK
            node.inputs = [subject, args[0]]
        elif predicate == 'kill':
            node.op = EventNode.OP_KILL
            node.inputs = [subject, args[0]]
        elif predicate == 'collide':
            node.op = EventNode.OP_COLLIDE
            node.inputs = [subject, args[0]]
        elif predicate == 'die':
            node.op = EventNode.OP_DIE
            node.inputs = [subject]
        elif predicate == 'in_a_line':
            node.op = EventNode.OP_IN_A_LINE
            node.inputs = [subject]
        elif predicate == 'align':
            node.op = EventNode.OP_ALIGN
            node.inputs = [subject]
        else:
            raise Exception("invalid predicate of event " + predicate)
        return node

    def __and__(self, other):
        node = EventNode()
        node.op = EventNode.OP_AND
        node.inputs = [self, other]
        return node

    def __or__(self, other):
        node = EventNode()
        node.op = EventNode.OP_OR
        node.inputs = [self, other]
        return node

    def __invert__(self):
        node = EventNode()
        node.op = EventNode.OP_NOT
        node.inputs = [self]
        return node

Event = EventNode()

class AgentSymbol:
    """symbol to represent some agents"""
    def __init__(self, group, index):
        """ define a agent symbol, it can be the object or subject of EventNode

        group: group handle
            it is the return value of cfg.add_group()
        index: int or str
            int: a deterministic integer id
            str: can be 'all' or 'any', represents all or any agents in a group
        """
        self.group = group if group is not None else -1
        if index == 'any':
            self.index = -1
        elif index == 'all':
            self.index = -2
        else:
            assert isinstance(self.index, int), "index must be a deterministic int"
            self.index = index

    def __str__(self):
        return 'agent(%d,%d)' % (self.group, self.index)


class Config:
    """configuration class of gridworld game"""
    def __init__(self):
        self.config_dict = {}
        self.agent_type_dict = {}
        self.groups = []
        self.reward_rules = []

    def set(self, args):
        """ set parameters of global configuration

        Parameters
        ----------
        args : dict
            key value pair of the configuration
        """
        for key in args:
            self.config_dict[key] = args[key]

    def register_agent_type(self, name, attr):
        """ register an agent type

        Parameters
        ----------
        name : str
            name of the type (should be unique)
        attr: dict
            key value pair of the agent type
            see notes below to know the available attributes

        Notes
        -----
        height: int, height of agent body
        width:  int, width of agent body
        speed:  float, maximum speed, i.e. the radius of move circle of the agent
        hp:     float, maximum health point of the agent
        view_range: gw.CircleRange or gw.SectorRange

        damage: float, attack damage
        step_recover: float, step recover of health point (can be negative)
        kill_supply: float, the hp gain when kill this type of agents

        step_reward: float, reward get in every step
        kill_reward: float, reward gain when kill this type of agent
        dead_penalty: float, reward get when dead
        attack_penalty: float, reward get when perform an attack (this is used to make agents do not attack blank grid)
        """
        if name in self.agent_type_dict:
            raise Exception("type name %s already exists" % name)
        self.agent_type_dict[name] = attr
        return name

    def add_group(self, agent_type):
        """ add a group to the configuration

        Returns
        -------
        group_handle : int
            a handle for the new added group
        """
        no = len(self.groups)
        self.groups.append(agent_type)
        return no

    def add_reward_rule(self, on, receiver, value, terminal=False):
        """ add a reward rule

        Some note:
        1. if the receiver is not a deterministic agent,
           it must be one of the agents involved in the triggering event

        Parameters
        ----------
        on: Expr
            a bool expression of the trigger event
        receiver:  (list of) AgentSymbol
            receiver of this reward rule
        value: (list of) float
            value to assign
        terminal: bool
            whether this event will terminate the game
        """
        if not (isinstance(receiver, tuple) or isinstance(receiver, list)):
            assert not (isinstance(value, tuple) or isinstance(value, tuple))
            receiver = [receiver]
            value = [value]
        if len(receiver) != len(value):
            raise Exception("the length of receiver and value should be equal")
        self.reward_rules.append([on, receiver, value, terminal])


class CircleRange:
    def __init__(self, radius):
        """ define a circle range for attack or view

        Parameters
        ----------
        radius : float
        """
        self.radius = radius
        self.angle  = 360

    def __str__(self):
        return 'circle(%g)' % self.radius


class SectorRange:
    def __init__(self, radius, angle):
        """ define a sector range for attack or view

        Parameters
        ----------
        radius : float
        angle :  float
            angle should be less than 180
        """
        self.radius = radius
        self.angle  = angle
        if self.angle >= 180:
            raise Exception("the angle of a sector should be smaller than 180 degree")

    def __str__(self):
        return 'sector(%g, %g)' % (self.radius, self.angle)
