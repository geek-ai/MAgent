"""advantage actor critic"""

import os
import time

import numpy as np
import mxnet as mx

from .base import MXBaseModel


class AdvantageActorCritic(MXBaseModel):
    def __init__(self, env, handle, name, eval_obs=None,
                 batch_size=64, reward_decay=0.99, learning_rate=1e-3,
                 train_freq=1, value_coef=0.1, ent_coef=0.1,
                 custom_view_space=None, custom_feature_space=None,
                 *args, **kwargs):
        """init a model

        Parameters
        ----------
        env: Environment
            environment
        handle: Handle (ctypes.c_int32)
            handle of this group, can be got by env.get_handles
        name: str
            name of this model
        learning_rate: float
        batch_size: int
        reward_decay: float
            reward_decay in TD
        eval_obs: numpy array
            evaluation set of observation
        train_freq: int
            mean training times of a sample
        ent_coef: float
            weight of entropy loss in total loss
        value_coef: float
            weight of value loss in total loss
        custom_feature_space: tuple
            customized feature space
        custom_view_space: tuple
            customized feature space
        """
        MXBaseModel.__init__(self, env, handle, name, "mxa2c")
        # ======================== set config  ========================
        self.env = env
        self.handle = handle
        self.view_space = custom_view_space or env.get_view_space(handle)
        self.feature_space = custom_feature_space or env.get_feature_space(handle)
        self.num_actions  = env.get_action_space(handle)[0]
        self.reward_decay = reward_decay

        self.batch_size   = batch_size
        self.learning_rate= learning_rate
        self.train_freq   = train_freq   # train time of every sample (s,a,r,s')
        self.eval_obs     = eval_obs

        self.value_coef = value_coef
        self.ent_coef   = ent_coef

        self.train_ct = 0

        # ======================= build network =======================
        self.ctx = self._get_ctx()

        self.input_view = mx.sym.var("input_view")
        self.input_feature = mx.sym.var("input_feature")

        policy, value = self._create_network(self.input_view, self.input_feature)

        log_policy = mx.sym.log(policy)
        out_policy = mx.sym.BlockGrad(policy)

        neg_entropy = ent_coef * mx.sym.sum(policy * log_policy, axis=1)
        neg_entropy = mx.sym.MakeLoss(data=neg_entropy)

        self.sym = mx.sym.Group([log_policy, value, neg_entropy, out_policy])
        self.model = mx.mod.Module(self.sym, data_names=['input_view', 'input_feature'],
                                   label_names=None, context=self.ctx)

        # bind (set initial batch size)
        self.bind_size = batch_size
        self.model.bind(data_shapes=[('input_view', (batch_size,) + self.view_space),
                                     ('input_feature', (batch_size,) + self.feature_space)],
                        label_shapes=None)

        # init params
        self.model.init_params(initializer=mx.init.Xavier())
        self.model.init_optimizer(optimizer='adam', optimizer_params={
            'learning_rate': learning_rate,
            'clip_gradient': 10,
        })

        # init training buffers
        self.view_buf = np.empty((1,) + self.view_space)
        self.feature_buf = np.empty((1,) + self.feature_space)
        self.action_buf = np.empty(1, dtype=np.int32)
        self.advantage_buf, self.value_buf = np.empty(1), np.empty(1)
        self.terminal_buf = np.empty(1, dtype=np.bool)

        # print("parameters", self.model.get_params())
        # mx.viz.plot_network(self.output).view()

    def _create_network(self, input_view, input_feature):
        """define computation graph of network

        Parameters
        ----------
        view_space: tuple
        feature_space: tuple
            the input shape
        """
        kernel_num = [32, 32]
        hidden_size = [256]

        if False:
            h_conv1 = mx.sym.Convolution(data=input_view, kernel=(3, 3),
                                         num_filter=kernel_num[0], layout="NHWC")
            h_conv1 = mx.sym.Activation(data=h_conv1, act_type="relu")
            h_conv2 = mx.sym.Convolution(data=h_conv1, kernel=(3, 3),
                                         num_filter=kernel_num[1], layout="NHWC")
            h_conv2 = mx.sym.Activation(data=h_conv2, act_type="relu")
        else:
            input_view = mx.sym.flatten(data=input_view)
            h_conv2 = mx.sym.FullyConnected(input_view, num_hidden=hidden_size[0])
            h_conv2 = mx.sym.Activation(data=h_conv2, act_type="relu")

        flatten_view = mx.sym.flatten(data=h_conv2)
        h_view = mx.sym.FullyConnected(data=flatten_view, num_hidden=hidden_size[0])
        h_view = mx.sym.Activation(data=h_view, act_type="relu")

        h_emb = mx.sym.FullyConnected(data=input_feature, num_hidden=hidden_size[0])
        h_emb = mx.sym.Activation(data=h_emb, act_type="relu")

        dense = h_view + h_emb

        policy = mx.sym.FullyConnected(data=dense, num_hidden=self.num_actions, no_bias=True)
        policy = mx.sym.SoftmaxActivation(data=policy)
        policy = mx.sym.clip(data=policy, a_min=1e-5, a_max=1 - 1e-5)
        value  = mx.sym.FullyConnected(data=dense, num_hidden=1)

        return policy, value

    def infer_action(self, raw_obs, ids, policy="e_greedy", eps=0):
        """infer action for a batch of agents

        Parameters
        ----------
        raw_obs: tuple(numpy array, numpy array)
            raw observation of agents tuple(views, features)
        ids: numpy array
            ids of agents

        Returns
        -------
        acts: numpy array of int32
            actions for agents
        """
        view, feature = raw_obs[0], raw_obs[1]
        n = len(view)

        ret = np.empty(n, dtype=np.int32)
        self._reset_bind_size(n)
        data_batch = mx.io.DataBatch(data=[mx.nd.array(view), mx.nd.array(feature)])
        self.model.forward(data_batch, is_train=False)
        policy = self.model.get_outputs()[3].asnumpy()

        actions = np.arange(self.num_actions)
        for i in range(n):
            ret[i] = np.random.choice(actions, p=policy[i])
        return ret

    def train(self, sample_buffer, print_every=1000):
        """feed new data sample and train

        Parameters
        ----------
        sample_buffer: magent.utility.EpisodesBuffer
            buffer contains samples

        Returns
        -------
        loss: list
            policy gradient loss, critic loss, entropy loss
        value: float
            estimated state value
        """
        # calc buffer size
        n = 0
        for episode in sample_buffer.episodes():
            if episode.terminal:
                n += len(episode.rewards)
            else:
                n += len(episode.rewards) - 1

        if n == 0:
            return [0.0, 0.0, 0.0], 0.0

        # resize to the new size
        self.view_buf.resize((n,) + self.view_space)
        self.feature_buf.resize((n,) + self.feature_space)
        self.action_buf.resize(n)
        self.value_buf.resize(n)
        self.advantage_buf.resize(n)
        view, feature = self.view_buf, self.feature_buf
        action, value = self.action_buf, self.value_buf
        advantage     = self.advantage_buf

        ct = 0
        gamma = self.reward_decay
        # collect episodes from multiple separate buffers to a continuous buffer

        for episode in sample_buffer.episodes():
            v, f, a, r = episode.views, episode.features, episode.actions, episode.rewards

            m = len(episode.rewards)
            self._reset_bind_size(m)
            data_batch = mx.io.DataBatch(data=[mx.nd.array(v), mx.nd.array(f)])
            self.model.forward(data_batch, is_train=False)
            value = self.model.get_outputs()[1].asnumpy().flatten()

            delta_t = np.empty(m)
            if episode.terminal:
                delta_t[:m-1] = r[:m-1] + gamma * value[1:m] - value[:m-1]
                delta_t[m-1]  = r[m-1]  + gamma * 0          - value[m-1]
            else:
                delta_t[:m-1] = r[:m-1] + gamma * value[1:m] - value[:m-1]
                m -= 1
                v, f, a = v[:-1], f[:-1], a[:-1]

            if m == 0:
                continue

            # discount advantage
            keep = 0
            for i in reversed(range(m)):
                keep = keep * gamma + delta_t[i]
                advantage[ct+i] = keep

            view[ct:ct+m] = v
            feature[ct:ct+m] = f
            action[ct:ct+m]  = a
            ct += m
        assert n == ct

        n = len(advantage)
        neg_advantage = -advantage

        neg_advs_np = np.zeros((n, self.num_actions), dtype=np.float32)
        neg_advs_np[np.arange(n), action] = neg_advantage
        neg_advs = mx.nd.array(neg_advs_np)

        # the grads of values are exactly negative advantages
        v_grads = mx.nd.array(self.value_coef * (neg_advantage[:, np.newaxis]))
        data_batch = mx.io.DataBatch(data=[mx.nd.array(view), mx.nd.array(feature)])
        self._reset_bind_size(n)
        self.model.forward(data_batch, is_train=True)
        self.model.backward(out_grads=[neg_advs, v_grads])
        self.model.update()
        log_policy, value, entropy_loss, _ = self.model.get_outputs()

        value = mx.nd.mean(value).asnumpy()[0]
        log_policy = log_policy.asnumpy()[np.arange(n), action]
        pg_loss = np.mean(neg_advantage * log_policy)
        entropy_loss = np.mean(entropy_loss.asnumpy())
        value_loss = self.value_coef * np.mean(np.square(advantage))

        print("sample %d  %.4f %.4f %.4f %.4f" % (n, pg_loss, value_loss, entropy_loss, value))
        return [pg_loss, value_loss, entropy_loss], value

    def _reset_bind_size(self, new_size):
        """reset input shape of the model

        Parameters
        ----------
        new_size: int
            new batch size
        """
        if self.bind_size == new_size:
            return
        else:
            self.bind_size = new_size
            self.model.reshape(
                data_shapes=[
                    ('input_view',    (new_size,) + self.view_space),
                    ('input_feature', (new_size,) + self.feature_space)],
            )

    def get_info(self):
        """get information of the model

        Returns
        -------
        info: string
        """
        return "mx dqn train_time: %d" % (self.train_ct)
