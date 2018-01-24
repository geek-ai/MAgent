from __future__ import absolute_import

import ctypes
import os
import importlib

import numpy as np

from .c_lib import _LIB, as_float_c_array, as_int32_c_array
from .environment import Environment


class TransCity(Environment):
    OBS_VIEW_INDEX = 0
    OBS_FEATURE_INDEX = 1

    def __init__(self, config, **kwargs):
        Environment.__init__(self)

        # for global settings
        game = ctypes.c_void_p()
        _LIB.env_new_game(ctypes.byref(game), b"TransCity")
        self.game = game

        config_value_type = {
            'map_width': int, 'map_height': int,
            'view_width': int, 'view_height': int,
            'embedding_size': int, 'seed': int,
            'render_dir': str,
        }

        # config general setting
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

        # init observation buffer (for acceleration)
        self._init_obs_buf()

        # init view size, feature size, action space
        buf = np.empty((3,), dtype=np.int32)
        _LIB.env_get_info(self.game, 0, b"view_space",
                          buf.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
        self.view_space = (buf[0], buf[1], buf[2])
        _LIB.env_get_info(self.game, 0, b"feature_space",
                          buf.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
        self.feature_space = (buf[0],)
        _LIB.env_get_info(self.game, 0, b"action_space",
                          buf.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
        self.action_space = (buf[0],)

    def reset(self):
        _LIB.env_reset(self.game)

    def _add_object(self, obj_id, method, **kwargs):
        if method == "random":
            _LIB.trans_city_add_object(self.game, obj_id, int(kwargs["n"]), b"random", 0)
        elif method == "custom":
            n = len(kwargs["pos"])
            pos = np.array(kwargs["pos"], dtype=np.int32)
            _LIB.trans_city_add_object(self.game, obj_id, n, b"custom",
                                       as_int32_c_array(pos))
        else:
            print("unsupported type of method")
            exit(-1)

    def add_walls(self, method, **kwargs):
        # handle = -1 for walls
        self._add_object(-1, method, **kwargs)

    def add_traffic_lights(self, method, **kwargs):
        # handles = -2 for traffic light
        self._add_object(-2, method, **kwargs)

    def add_parks(self, method, **kwargs):
        # handle = -3 for parking
        self._add_object(-3, method, **kwargs)

    def add_buildings(self, method, **kwargs):
        # handle = -4 for building
        self._add_object(-4, method, **kwargs)

    def add_agents(self, method, **kwargs):
        self._add_object(0, method, **kwargs)

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

    def get_observation(self, handle=0):
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
        view_space = self.view_space
        feature_space = self.feature_space
        no = 0

        n = self.get_num(handle)
        view_buf = self._get_obs_buf(no, self.OBS_VIEW_INDEX, (n,) + view_space, np.float32)
        feature_buf = self._get_obs_buf(no, self.OBS_FEATURE_INDEX, (n, feature_space), np.float32)

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

    def get_reward(self, handle=0):
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
        _LIB.trans_city_clear_dead(self.game)

    # ====== INFO ======
    def get_num(self, handle=0):
        """get the number of agents in a group"""
        num = ctypes.c_int32()
        _LIB.env_get_info(self.game, handle, b'num', ctypes.byref(num))
        return num.value

    def get_action_space(self, handle=0):
        """get action space"""
        return self.action_space

    def get_view_space(self, handle=0):
        """get view space"""
        return self.view_space

    def get_feature_space(self, handle=0):
        """get feature space """
        return self.feature_space

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

    def render_next_file(self):
        """start a new file to store render frames"""
        _LIB.env_render_next_file(self.game)

    def __del__(self):
        _LIB.env_delete_game(self.game)


class Config:
    def __init__(self):
        self.config_dict = {}

    def set(self, args):
        for key in args:
            self.config_dict[key] = args[key]

