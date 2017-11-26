"""Deep q network"""

import time

import numpy as np
import tensorflow as tf

from .base import TFBaseModel
from ..common import ReplayBuffer


class DeepQNetwork(TFBaseModel):
    def __init__(self, env, handle, name,
                 batch_size=64, learning_rate=1e-4, reward_decay=0.99,
                 train_freq=1, target_update=2000, memory_size=2 ** 20, eval_obs=None,
                 use_dueling=True, use_double=True, use_conv=True,
                 custom_view_space=None, custom_feature_space=None,
                 num_gpu=1, infer_batch_size=8192, network_type=0):
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
        use_conv: bool
            use convolution or fully connected layer as state encoder
        num_gpu: int
            number of gpu
        infer_batch_size: int
            batch size while inferring actions
        custom_feature_space: tuple
            customized feature space
        custom_view_space: tuple
            customized feature space
        """
        TFBaseModel.__init__(self, env, handle, name, "tfdqn")
        # ======================== set config  ========================
        self.env = env
        self.handle = handle
        self.view_space = custom_view_space or env.get_view_space(handle)
        self.feature_space = custom_feature_space or env.get_feature_space(handle)
        self.num_actions  = env.get_action_space(handle)[0]

        self.batch_size   = batch_size
        self.learning_rate= learning_rate
        self.train_freq   = train_freq     # train time of every sample (s,a,r,s')
        self.target_update= target_update  # target network update frequency
        self.eval_obs     = eval_obs
        self.infer_batch_size = infer_batch_size  # maximum batch size when infer actions,
        # change this to fit your GPU memory if you meet a OOM

        self.use_dueling  = use_dueling
        self.use_double   = use_double
        self.num_gpu      = num_gpu
        self.network_type = network_type

        self.train_ct = 0

        # ======================= build network =======================
        # input place holder
        self.target = tf.placeholder(tf.float32, [None])
        self.weight = tf.placeholder(tf.float32, [None])

        self.input_view    = tf.placeholder(tf.float32, (None,) + self.view_space)
        self.input_feature = tf.placeholder(tf.float32, (None,) + self.feature_space)
        self.action = tf.placeholder(tf.int32, [None])
        self.mask   = tf.placeholder(tf.float32, [None])
        self.eps = tf.placeholder(tf.float32)  # e-greedy

        # build graph
        with tf.variable_scope(self.name):
            with tf.variable_scope("eval_net_scope"):
                self.eval_scope_name   = tf.get_variable_scope().name
                self.qvalues = self._create_network(self.input_view, self.input_feature, use_conv)

            if self.num_gpu > 1:  # build inference graph for multiple gpus
                self._build_multi_gpu_infer(self.num_gpu)

            with tf.variable_scope("target_net_scope"):
                self.target_scope_name = tf.get_variable_scope().name
                self.target_qvalues = self._create_network(self.input_view, self.input_feature, use_conv)

        # loss
        self.gamma = reward_decay
        self.actions_onehot = tf.one_hot(self.action, self.num_actions)
        td_error = tf.square(self.target - tf.reduce_sum(tf.multiply(self.actions_onehot, self.qvalues), axis=1))
        self.loss = tf.reduce_sum(td_error * self.mask) / tf.reduce_sum(self.mask)

        # train op (clip gradient)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        gradients, variables = zip(*optimizer.compute_gradients(self.loss))
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        self.train_op = optimizer.apply_gradients(zip(gradients, variables))

        # output action
        def out_action(qvalues):
            best_action = tf.argmax(qvalues, axis=1)
            best_action = tf.to_int32(best_action)
            random_action = tf.random_uniform(tf.shape(best_action), 0, self.num_actions, tf.int32)
            should_explore = tf.random_uniform(tf.shape(best_action), 0, 1) < self.eps
            return tf.where(should_explore, random_action, best_action)

        self.output_action = out_action(self.qvalues)
        if self.num_gpu > 1:
            self.infer_out_action = [out_action(qvalue) for qvalue in self.infer_qvalues]

        # target network update op
        self.update_target_op = []
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.target_scope_name)
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.eval_scope_name)
        for i in range(len(t_params)):
            self.update_target_op.append(tf.assign(t_params[i], e_params[i]))

        # init tensorflow session
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

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

    def _create_network(self, input_view, input_feature, use_conv=True, reuse=None):
        """define computation graph of network

        Parameters
        ----------
        input_view: tf.tensor
        input_feature: tf.tensor
            the input tensor
        """
        kernel_num  = [32, 32]
        hidden_size = [256]

        if use_conv:  # convolution
            h_conv1 = tf.layers.conv2d(input_view, filters=kernel_num[0], kernel_size=3,
                                       activation=tf.nn.relu, name="conv1", reuse=reuse)
            h_conv2 = tf.layers.conv2d(h_conv1, filters=kernel_num[1], kernel_size=3,
                                       activation=tf.nn.relu, name="conv2", reuse=reuse)
            flatten_view = tf.reshape(h_conv2, [-1, np.prod([v.value for v in h_conv2.shape[1:]])])
            h_view = tf.layers.dense(flatten_view, units=hidden_size[0], activation=tf.nn.relu,
                                     name="dense_view", reuse=reuse)
        else:         # fully connected
            flatten_view = tf.reshape(input_view, [-1, np.prod([v.value for v in input_view.shape[1:]])])
            h_view = tf.layers.dense(flatten_view, units=hidden_size[0], activation=tf.nn.relu)

        h_emb = tf.layers.dense(input_feature,  units=hidden_size[0], activation=tf.nn.relu,
                                name="dense_emb", reuse=reuse)

        dense = tf.concat([h_view, h_emb], axis=1)

        if self.use_dueling:
            value = tf.layers.dense(dense, units=1, name="value", reuse=reuse)
            advantage = tf.layers.dense(dense, units=self.num_actions, use_bias=False,
                                        name="advantage", reuse=reuse)

            qvalues = value + advantage - tf.reduce_mean(advantage, axis=1, keep_dims=True)
        else:
            qvalues = tf.layers.dense(dense, units=self.num_actions, name="value", reuse=reuse)

        return qvalues

    def infer_action(self, raw_obs, ids, policy='e_greedy', eps=0):
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
        batch_size = min(n, self.infer_batch_size)

        if self.num_gpu > 1 and n > batch_size:   # infer by multi gpu in parallel
            ret = self._infer_multi_gpu(view, feature, ids, eps)
        else:                  # infer by splitting big batch in serial
            ret = []
            for i in range(0, n, batch_size):
                beg, end = i, i + batch_size
                ret.append(self.sess.run(self.output_action, feed_dict={
                    self.input_view: view[beg:end],
                    self.input_feature: feature[beg:end],
                    self.eps: eps}))
            ret = np.concatenate(ret)
        return ret

    def _calc_target(self, next_view, next_feature, rewards, terminal):
        """calculate target value"""
        n = len(rewards)
        if self.use_double:
            t_qvalues, qvalues = self.sess.run([self.target_qvalues, self.qvalues],
                                               feed_dict={self.input_view: next_view,
                                                          self.input_feature: next_feature})
            next_value = t_qvalues[np.arange(n), np.argmax(qvalues, axis=1)]
        else:
            t_qvalues = self.sess.run(self.target_qvalues, {self.input_view: next_view,
                                                            self.input_feature: next_feature})
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

            ret = self.sess.run([self.train_op, self.loss], feed_dict={
                self.input_view:    batch_view,
                self.input_feature: batch_feature,
                self.action:        batch_action,
                self.target:        batch_target,
                self.mask:          batch_mask
            })
            loss = ret[1]
            total_loss += loss

            if ct % self.target_update == 0:
                self.sess.run(self.update_target_op)

            if ct % print_every == 0:
                print("batch %5d,  loss %.6f, eval %.6f" % (ct, loss, self._eval(batch_target)))
            ct += 1
            self.train_ct += 1

        total_time = time.time() - start_time
        step_average = total_time / max(1.0, (ct / 1000.0))
        print("batches: %d,  total time: %.2f,  1k average: %.2f" % (ct, total_time, step_average))

        return total_loss / ct if ct != 0 else 0, self._eval(batch_target)

    def _eval(self, target):
        """evaluate estimated q value"""
        if self.eval_obs is None:
            return np.mean(target)
        else:
            return np.mean(self.sess.run([self.qvalues], feed_dict={
                self.input_view: self.eval_obs[0],
                self.input_feature: self.eval_obs[1]
            }))

    def clear_buffer(self):
        """clear replay buffer"""
        self.replay_buf_len = 0
        self.replay_buf_view.clear()
        self.replay_buf_feature.clear()
        self.replay_buf_action.clear()
        self.replay_buf_reward.clear()
        self.replay_buf_terminal.clear()
        self.replay_buf_mask.clear()

    def _build_multi_gpu_infer(self, num_gpu):
        """build inference graph for multi gpus"""
        self.infer_qvalues = []
        self.infer_input_view = []
        self.infer_input_feature = []
        for i in range(num_gpu):
            self.infer_input_view.append(tf.placeholder(tf.float32, (None,) + self.view_space))
            self.infer_input_feature.append(tf.placeholder(tf.float32, (None,) + self.feature_space))
            with tf.variable_scope("eval_net_scope"), tf.device("/gpu:%d" % i):
                self.infer_qvalues.append(self._create_network(self.infer_input_view[i],
                                                               self.infer_input_feature[i], reuse=True))

    def _infer_multi_gpu(self, view, feature, ids, eps):
        """infer action by multi gpu in parallel """
        ret = []
        beg = 0
        while beg < len(view):
            feed_dict = {self.eps: eps}
            for i in range(self.num_gpu):
                end = beg + self.infer_batch_size
                feed_dict[self.infer_input_view[i]] = view[beg:end]
                feed_dict[self.infer_input_feature[i]] = feature[beg:end]
                beg += self.infer_batch_size

            ret.extend(self.sess.run(self.infer_out_action, feed_dict=feed_dict))
        return np.concatenate(ret)
