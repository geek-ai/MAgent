""" some utility for call C++ code"""

from __future__ import absolute_import

import os
import ctypes
import platform


def _load_lib():
    """ Load library in build/lib. """
    cur_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    lib_path = os.path.join(cur_path, "../../build/")
    if platform.system() == 'Darwin':
        path_to_so_file = os.path.join(lib_path, "libmagent.dylib")
    elif platform.system() == 'Linux':
        path_to_so_file = os.path.join(lib_path, "libmagent.so")
    else:
        raise BaseException("unsupported system: " + platform.system())
    lib = ctypes.CDLL(path_to_so_file, ctypes.RTLD_GLOBAL)
    return lib


def as_float_c_array(buf):
    """numpy to ctypes array"""
    return buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def as_int32_c_array(buf):
    """numpy to ctypes array"""
    return buf.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))


def as_bool_c_array(buf):
    """numpy to ctypes array"""
    return buf.ctypes.data_as(ctypes.POINTER(ctypes.c_bool))


os.environ['OMP_NUM_THREADS'] = '8'   # performs best in our machines
_LIB = _load_lib()
