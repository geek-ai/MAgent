"""A random agent"""

import numpy as np

from magent.model import BaseModel


class RandomActor(BaseModel):
    def __init__(self, env, handle, *args, **kwargs):
        BaseModel.__init__(self, env, handle)

        self.env = env
        self.handle = handle
        self.n_action = env.get_action_space(handle)[0]

    def infer_action(self, obs, *args, **kwargs):
        num = len(obs[0])
        actions = np.random.randint(self.n_action, size=num, dtype=np.int32)
        return actions
