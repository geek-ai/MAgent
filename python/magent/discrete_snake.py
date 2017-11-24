""" Deprecated!! """

from __future__ import absolute_import

import ctypes
import os
import importlib

import numpy as np

from .c_lib import _LIB, as_float_c_array, as_int32_c_array
from .environment import Environment


class DiscreteSnake(Environment):
    """deprecated"""
    OBS_VIEW_INDEX = 0
    OBS_FEATURE_INDEX = 1

    def __init__(self, config, **kwargs):
        Environment.__init__(self)

        # for global settings
        game = ctypes.c_void_p()
        _LIB.env_new_game(ctypes.byref(game), b"DiscreteSnake")
        self.game = game

        config_value_type = {
            'map_width': int, 'map_height': int,
            'view_width': int, 'view_height': int,
            'max_dead_penalty': float, 'corpse_value': float,
            'embedding_size': int, 'total_resource': int,
            'render_dir': str,
        }

        # config general setting
        for key in config.config_dict:
            print("discrete_snake.py L37 : ", key, config.config_dict[key])
            value_type = config_value_type[key]
            if value_type is int:
                _LIB.env_config_game(self.game, key, ctypes.byref(ctypes.c_int(config.config_dict[key])))
            elif value_type is bool:
                _LIB.env_config_game(self.game, key, ctypes.byref(ctypes.c_bool(config.config_dict[key])))
            elif value_type is float:
                _LIB.env_config_game(self.game, key, ctypes.byref(ctypes.c_float(config.config_dict[key])))
            elif value_type is str:
                _LIB.env_config_game(self.game, key, ctypes.c_char_p(config.config_dict[key]))

        # init observation buffer (for acceleration)
        self._init_obs_buf()

        # init view size, feature size, action space
        buf = np.empty((3,), dtype=np.int32)
        _LIB.env_get_info(self.game, 0, b"view_space",
                          buf.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
        self.view_space = [buf[0], buf[1], buf[2]]
        _LIB.env_get_info(self.game, 0, b"feature_space",
                          buf.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
        self.feature_space = buf[0]
        _LIB.env_get_info(self.game, 0, b"action_space",
                          buf.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
        self.action_space = buf[0]

    def reset(self):
        _LIB.env_reset(self.game)

    def _add_object(self, obj_id, method, **kwargs):
        if method == "random":
            _LIB.discrete_snake_add_object(self.game, obj_id, int(kwargs["n"]), b"random", 0)
        else:
            print("unsupported type of method")
            exit(-1)

    def add_walls(self, method, **kwargs):
        # handle = -1 for walls
        self._add_object(-1, method, **kwargs)

    def add_food(self, method, **kwargs):
        # handles = -2 for food
        self._add_object(-2, method, **kwargs)

    def add_agent(self, method, *args, **kwargs):
        self._add_object(0, method, **kwargs)

    # ====== RUN ======
    def _get_obs_buf(self, key, shape, dtype):
        if self.obs_bufs[key] is None:
            group_buf = self.obs_bufs[key] = [1]  # (buf_id, buf1, buf2, ...)
            group_buf.append(np.zeros(shape=shape, dtype=dtype))
            group_buf.append(np.zeros(shape=shape, dtype=dtype))
            ret = group_buf[1]
        else:
            group_buf = self.obs_bufs[key]
            turn = group_buf[0]
            ret = group_buf[turn]
            if shape != ret.shape:
                ret.resize(shape, refcheck=False)
            group_buf[0] = (turn-1 + 1) % 2 + 1

        return ret

    def _init_obs_buf(self):
        self.obs_bufs = [None, None]

    def get_observation(self, handle=0):
        view_space = self.view_space
        feature_space = self.feature_space

        n = self.get_num(handle)
        view_buf = self._get_obs_buf(self.OBS_VIEW_INDEX, [n] + view_space, np.float32)
        feature_buf = self._get_obs_buf(self.OBS_FEATURE_INDEX, (n, feature_space), np.float32)

        bufs = (ctypes.POINTER(ctypes.c_float) * 2)()
        bufs[0] = as_float_c_array(view_buf)
        bufs[1] = as_float_c_array(feature_buf)
        _LIB.env_get_observation(self.game, handle, bufs)

        return view_buf, feature_buf

    def set_action(self, handle, actions):
        assert isinstance(actions, np.ndarray)
        assert actions.dtype == np.int32
        _LIB.env_set_action(self.game, handle, actions.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))

    def step(self):
        done = ctypes.c_int32()
        _LIB.env_step(self.game, ctypes.byref(done))
        return done

    def get_reward(self, handle=0):
        n = self.get_num(handle)
        buf = np.empty((n,), dtype=np.float32)
        _LIB.env_get_reward(self.game, handle,
                            buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        return buf

    def clear_dead(self):
        _LIB.discrete_snake_clear_dead(self.game)

    # ====== INFO ======
    def get_num(self, handle=0):
        num = ctypes.c_int32()
        _LIB.env_get_info(self.game, handle, "num", ctypes.byref(num))
        return num.value

    def get_action_space(self, handle=0):
        return self.action_space

    def get_view_space(self, handle=0):
        return self.view_space

    def get_feature_space(self, handle=0):
        return self.feature_space

    def get_agent_id(self, handle=0):
        n = self.get_num(handle)
        buf = np.empty((n,), dtype=np.int32)
        _LIB.env_get_info(self.game, handle, b"id",
                          buf.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
        return buf

    def get_head(self, handle=0):
        n = self.get_num(handle)
        buf = np.empty((n, 2), dtype=np.int32)
        _LIB.env_get_info(self.game, handle, b"head",
                          buf.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
        return buf

    def get_alive(self, handle=0):
        n = self.get_num(handle)
        buf = np.empty((n,), dtype=np.bool)
        _LIB.env_get_info(self.game, handle, b"alive",
                          buf.ctypes.data_as(ctypes.POINTER(ctypes.c_bool)))
        return buf

    def get_length(self, handle=0):
        n = self.get_num(handle)
        buf = np.empty((n, ), dtype=np.int32)
        _LIB.env_get_info(self.game, handle, b"length",
                          buf.ctypes.data_as(ctypes.POINTER(ctypes.c_int)))
        return buf

    def get_food_num(self):
        num = ctypes.c_int32()
        _LIB.env_get_info(self.game, -2, "num", ctypes.byref(num))  # -2 for food
        return num.value

    # ====== RENDER ======
    def set_render_dir(self, name):
        if not os.path.exists(name):
            os.mkdir(name)
        _LIB.env_config_game(self.game, b"render_dir", name)

    def render(self):
        _LIB.env_render(self.game)

    def render_next_file(self):
        _LIB.env_render_next_file(self.game)

    def __del__(self):
        _LIB.env_delete_game(self.game)


class Config:
    def __init__(self):
        self.config_dict = {}

    def set(self, args):
        for key in args:
            self.config_dict[key] = args[key]