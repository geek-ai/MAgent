import time

import numpy as np
import mxnet as mx

from .base import MXBaseModel
from ..common import ReplayBuffer
from ...utility import has_gpu


class DeepQNetwork(MXBaseModel):
    def __init__(self, env, handle, name,
                 batch_size=64, learning_rate=1e-4, reward_decay=0.99,
                 train_freq=1, target_update=2000, memory_size=2 ** 20, eval_obs=None,
                 use_dueling=True, use_double=True, infer_batch_size=8192,
                 custom_view_space=None, custom_feature_space=None, num_gpu=1):
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
        train_freq: int
            mean training times of a sample
        target_update: int
            target will update every target_update batches
        memory_size: int
            weight of entropy loss in total loss
        eval_obs: numpy array
            evaluation set of observation
        use_dueling: bool
            whether use dueling q network
        use_double: bool
            whether use double q network
        num_gpu: int
            number of gpu
        infer_batch_size: int
            batch size while inferring actions
        custom_feature_space: tuple
            customized feature space
        custom_view_space: tuple
            customized feature space
        """
        MXBaseModel.__init__(self, env, handle, name, "mxdqn")
        # ======================== set config  ========================
        self.env = env
        self.handle = handle
        self.view_space = custom_view_space or env.get_view_space(handle)
        self.feature_space = custom_feature_space or env.get_feature_space(handle)
        self.num_actions  = env.get_action_space(handle)[0]

        self.batch_size    = batch_size
        self.infer_batch_size = infer_batch_size
        self.learning_rate = learning_rate
        self.train_freq    = train_freq      # train time of every sample (s,a,r,s')
        self.target_update = target_update   # update frequency of target network
        self.eval_obs      = eval_obs
        self.num_gpu       = num_gpu

        self.use_dueling  = use_dueling
        self.use_double   = use_double

        self.train_ct = 0

        # ======================= build network =======================
        self.ctx = self._get_ctx()
        if self.num_gpu > 1 and self.ctx == mx.gpu():
            self.ctx = []
            for i in range(self.num_gpu):
                self.ctx.append(mx.gpu(i))

        self.input_view = mx.sym.var("input_view")
        self.input_feature = mx.sym.var("input_feature")
        self.mask   = mx.sym.var("mask")
        self.action = mx.sym.var("action")
        self.target = mx.sym.var("target")

        self.qvalues = self._create_network(self.input_view, self.input_feature)
        self.gamma = reward_decay
        self.action_onehot = mx.sym.one_hot(self.action, depth=self.num_actions)
        td_error = mx.sym.square(self.target - mx.sym.sum(self.qvalues * self.action_onehot, axis=1))
        self.loss = mx.sym.sum(td_error * self.mask) / mx.sym.sum(self.mask)
        self.loss = mx.sym.MakeLoss(data=self.loss)

        self.out_qvalues = mx.sym.BlockGrad(self.qvalues)
        self.output = mx.sym.Group([self.out_qvalues, self.loss])

        self.model = mx.mod.Module(self.output,
                                   data_names=['input_view', 'input_feature'],
                                   label_names=['action', 'target', 'mask'], context=self.ctx)

        self.target_model = mx.mod.Module(self.qvalues,
                                          data_names=['input_view', 'input_feature'],
                                          label_names=[], context=self.ctx)

        # bind (set initial batch size)
        self.bind_size = batch_size
        self.model.bind(data_shapes=[('input_view', (batch_size,) + self.view_space),
                                     ('input_feature', (batch_size,) + self.feature_space)],
                        label_shapes=[('action', (batch_size,)),
                                      ('target', (batch_size,)),
                                      ('mask', (batch_size,))])
        self.target_model.bind(data_shapes=[('input_view', (batch_size,) + self.view_space),
                                            ('input_feature', (batch_size,) + self.feature_space)])

        # init params
        self.model.init_params(initializer=mx.init.Xavier())
        self.model.init_optimizer(optimizer='adam', optimizer_params={
            'learning_rate': learning_rate,
            'clip_gradient': 10.0})

        self._copy_network(self.target_model, self.model)

        # init replay buffers
        self.replay_buf_len = 0
        self.memory_size = memory_size
        self.replay_buf_view     = ReplayBuffer(shape=(memory_size,) + self.view_space)
        self.replay_buf_feature  = ReplayBuffer(shape=(memory_size,) + self.feature_space)
        self.replay_buf_action   = ReplayBuffer(shape=(memory_size,), dtype=np.int32)
        self.replay_buf_reward   = ReplayBuffer(shape=(memory_size,))
        self.replay_buf_terminal = ReplayBuffer(shape=(memory_size,), dtype=np.bool)
        self.replay_buf_mask     = ReplayBuffer(shape=(memory_size,))
        # if mask[i] == 0, then the item is used for padding, not for training

        # print("parameters", self.model.get_params())
        # mx.viz.plot_network(self.loss).view()

    def _create_network(self, input_view, input_feature, use_conv=True):
        """define computation graph of network

        Parameters
        ----------
        input_view: mx.symbol
        input_feature: mx.symbol
            the input tensor
        """
        kernel_num = [32, 32]
        hidden_size = [256]

        if use_conv:
            input_view = mx.sym.transpose(data=input_view, axes=[0, 3, 1, 2])
            h_conv1 = mx.sym.Convolution(data=input_view, kernel=(3, 3),
                                         num_filter=kernel_num[0], layout="NCHW")
            h_conv1 = mx.sym.Activation(data=h_conv1, act_type="relu")
            h_conv2 = mx.sym.Convolution(data=h_conv1, kernel=(3, 3),
                                         num_filter=kernel_num[1], layout="NCHW")
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

        dense = mx.sym.concat(h_view, h_emb)

        if self.use_dueling:
            # state value
            value = mx.sym.FullyConnected(data=dense, num_hidden=1)
            advantage = mx.sym.FullyConnected(data=dense, num_hidden=self.num_actions)

            mean = mx.sym.mean(advantage, axis=1, keepdims=True)
            advantage = mx.sym.broadcast_sub(advantage, mean)
            qvalues = mx.sym.broadcast_add(advantage, value)
        else:
            qvalues = mx.sym.FullyConnected(data=dense, num_hidden=self.num_actions)

        return qvalues

    def infer_action(self, raw_obs, ids, policy="e_greedy", eps=0):
        """infer action for a batch of agents

        Parameters
        ----------
        raw_obs: tuple(numpy array, numpy array)
            raw observation of agents tuple(views, features)
        ids: numpy array
            ids of agents
        policy: str
            can be eps-greedy or greedy
        eps: float
            used when policy is eps-greedy

        Returns
        -------
        acts: numpy array of int32
            actions for agents
        """
        view, feature = raw_obs[0], raw_obs[1]

        if policy == 'e_greedy':
            eps = eps
        elif policy == 'greedy':
            eps = 0

        n = len(view)
        if n < self.num_gpu:
            view = np.tile(view, (self.num_gpu, 1, 1, 1))
            feature = np.tile(feature, (self.num_gpu, 1))

        batch_size = min(len(view), self.infer_batch_size)
        self._reset_bind_size(batch_size)
        best_actions = []
        infer_iter = mx.io.NDArrayIter(data=[view, feature], batch_size=batch_size)
        for batch in infer_iter:
            self.model.forward(batch, is_train=False)
            qvalue_batch = self.model.get_outputs()[0]
            batch_action = mx.nd.argmax(qvalue_batch, axis=1)
            best_actions.append(batch_action)
        best_actions = np.array([x.asnumpy() for x in best_actions]).flatten()
        best_actions = best_actions[:n]

        random = np.random.randint(self.num_actions, size=(n,))
        cond = np.random.uniform(0, 1, size=(n,)) < eps
        ret = np.where(cond, random, best_actions)

        return ret.astype(np.int32)

    def _calc_target(self, next_view, next_feature, rewards, terminal):
        """calculate target value"""
        n = len(rewards)

        data_batch = mx.io.DataBatch(data=[mx.nd.array(next_view), mx.nd.array(next_feature)])
        self._reset_bind_size(n)
        if self.use_double:
            self.target_model.forward(data_batch, is_train=False)
            self.model.forward(data_batch, is_train=False)
            t_qvalues = self.target_model.get_outputs()[0].asnumpy()
            qvalues   = self.model.get_outputs()[0].asnumpy()
            next_value = t_qvalues[np.arange(n), np.argmax(qvalues, axis=1)]
        else:
            self.target_model.forward(data_batch, is_train=False)
            t_qvalues = self.target_model.get_outputs()[0].asnumpy()
            next_value = np.max(t_qvalues, axis=1)

        target = np.where(terminal, rewards, rewards + self.gamma * next_value)

        return target

    def _add_to_replay_buffer(self, sample_buffer):
        """add samples in sample_buffer to replay buffer"""
        n = 0
        for episode in sample_buffer.episodes():
            v, f, a, r = episode.views, episode.features, episode.actions, episode.rewards

            m = len(r)

            mask = np.ones((m,))
            terminal = np.zeros((m,), dtype=np.bool)
            if episode.terminal:
                terminal[-1] = True
            else:
                mask[-1] = 0

            self.replay_buf_view.put(v)
            self.replay_buf_feature.put(f)
            self.replay_buf_action.put(a)
            self.replay_buf_reward.put(r)
            self.replay_buf_terminal.put(terminal)
            self.replay_buf_mask.put(mask)

            n += m

        self.replay_buf_len = min(self.memory_size, self.replay_buf_len + n)
        return n

    def train(self, sample_buffer, print_every=1000):
        """ add new samples in sample_buffer to replay buffer and train

        Parameters
        ----------
        sample_buffer: magent.utility.EpisodesBuffer
            buffer contains samples
        print_every: int
            print log every print_every batches

        Returns
        -------
        loss: float
            bellman residual loss
        value: float
            estimated state value
        """
        add_num = self._add_to_replay_buffer(sample_buffer)
        batch_size = self.batch_size
        total_loss = 0

        n_batches = int(self.train_freq * add_num / batch_size)
        if n_batches == 0:
            return 0, 0

        print("batch number: %d  add: %d  replay_len: %d/%d" %
              (n_batches, add_num, self.replay_buf_len, self.memory_size))

        start_time = time.time()
        ct = 0
        for i in range(n_batches):
            # fetch a batch
            index = np.random.choice(self.replay_buf_len - 1, batch_size)

            batch_view     = self.replay_buf_view.get(index)
            batch_feature  = self.replay_buf_feature.get(index)
            batch_action   = self.replay_buf_action.get(index)
            batch_reward   = self.replay_buf_reward.get(index)
            batch_terminal = self.replay_buf_terminal.get(index)
            batch_mask     = self.replay_buf_mask.get(index)

            batch_next_view    = self.replay_buf_view.get(index+1)
            batch_next_feature = self.replay_buf_feature.get(index+1)

            batch_target = self._calc_target(batch_next_view, batch_next_feature,
                                             batch_reward, batch_terminal)

            self._reset_bind_size(batch_size)
            batch = mx.io.DataBatch(data=[mx.nd.array(batch_view),
                                          mx.nd.array(batch_feature)],
                                    label=[mx.nd.array(batch_action),
                                           mx.nd.array(batch_target),
                                           mx.nd.array(batch_mask)])
            self.model.forward(batch, is_train=True)
            self.model.backward()
            self.model.update()
            loss = np.mean(self.model.get_outputs()[1].asnumpy())
            total_loss += loss

            if ct % self.target_update == 0:
                self._copy_network(self.target_model, self.model)

            if ct % print_every == 0:
                print("batch %5d,  loss %.6f, eval %.6f" % (ct, loss, self._eval(batch_target)))
            ct += 1
            self.train_ct += 1

        total_time = time.time() - start_time
        step_average = total_time / max(1.0, (ct / 1000.0))
        print("batches: %d,  total time: %.2f,  1k average: %.2f" % (ct, total_time, step_average))

        return total_loss / ct if ct != 0 else 0, self._eval(batch_target)

    def _reset_bind_size(self, new_size):
        """reset batch size"""
        if self.bind_size == new_size:
            return
        else:
            self.bind_size = new_size
            def _reshape(model, is_target):
                data_shapes = [('input_view',    (new_size,) + self.view_space),
                               ('input_feature', (new_size,) + self.feature_space)]
                label_shapes = [('action',        (new_size,)),
                                ('target',        (new_size,)),
                                ('mask',          (new_size,))]
                if is_target:
                    label_shapes = None
                model.reshape(data_shapes=data_shapes, label_shapes=label_shapes)
            _reshape(self.model, False)
            _reshape(self.target_model, True)

    def _copy_network(self, dest, source):
        """copy to target network"""
        arg_params, aux_params = source.get_params()
        dest.set_params(arg_params, aux_params)

    def _eval(self, target):
        """evaluate estimated q value"""
        if self.eval_obs is None:
            return np.mean(target)
        else:
            self._reset_bind_size(len(self.eval_obs[0]))
            with self.ctx:
                batch = mx.io.DataBatch(data=[mx.nd.array(self.eval_obs[0]),
                                              mx.nd.array(self.eval_obs[1])])
                self.model.forward(batch, is_train=False)
                return np.mean(self.model.get_outputs()[0].asnumpy())

    def get_info(self):
        return "mx dqn train_time: %d" % (self.train_ct)