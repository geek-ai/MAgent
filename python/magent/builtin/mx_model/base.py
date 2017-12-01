import os
import mxnet as mx

from magent.utility import has_gpu
from magent.model import BaseModel


class MXBaseModel(BaseModel):
    def __init__(self, env, handle, name, subclass_name):
        """init a model

        Parameters
        ----------
        env: magent.Environment
        handle: handle (ctypes.c_int32)
        name: str
        subclass_name: str
            name of subclass
        """
        BaseModel.__init__(self, env, handle)
        self.name = name
        self.subclass_name = subclass_name

    def _get_ctx(self):
        """return correct context , priority: gpu > cpu

        Returns
        -------
        ctx: mx.context
        """
        if has_gpu():
            return mx.gpu()
        else:
            return mx.cpu()

    def save(self, dir_name, epoch):
        """save model to dir

        Parameters
        ----------
        dir_name: str
            name of the directory
        epoch: int
        """
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        dir_name = os.path.join(dir_name, self.name, )
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        pre = os.path.join(dir_name, self.subclass_name)
        self.model.save_checkpoint(pre, epoch, save_optimizer_states=True)

    def load(self, dir_name, epoch=0, name=None):
        """save model to dir

        Parameters
        ----------
        dir_name: str
            name of the directory
        epoch: int
        """
        name = name or self.name
        dir_name = os.path.join(dir_name, name)
        pre = os.path.join(dir_name, self.subclass_name)
        _, arg_params, aux_params = mx.model.load_checkpoint(pre, epoch)
        self.model.set_params(arg_params, aux_params, force_init=True)
