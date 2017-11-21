import os
import mxnet as mx

from magent.utility import has_gpu
from magent.model import BaseModel


class MXBaseModel(BaseModel):
    def __init__(self, env, handle, name, subclass_name):
        BaseModel.__init__(self, env, handle)
        self.name = name
        self.subclass_name = subclass_name

    def _get_ctx(self):
        """check whether has a gpu"""
        if has_gpu():
            return mx.gpu()
        else:
            return mx.cpu()

    def save(self, dir_name, epoch):
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        dir_name = os.path.join(dir_name, self.name, )
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        pre = os.path.join(dir_name, self.subclass_name)
        self.model.save_checkpoint(pre, epoch, save_optimizer_states=True)

    def load(self, dir_name, epoch):
        dir_name = os.path.join(dir_name, self.name)
        pre = os.path.join(dir_name, self.subclass_name)
        _, arg_params, aux_params = mx.model.load_checkpoint(pre, epoch)
        self.model.set_params(arg_params, aux_params, force_init=True)
