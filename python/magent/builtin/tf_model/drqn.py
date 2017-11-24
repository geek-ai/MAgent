"""Deep recurrent Q network"""

import time
import os
import collections

import numpy as np
import tensorflow as tf

from .base import TFBaseModel


class DeepRecurrentQNetwork(TFBaseModel):
    def __init__(self, env, handle, name,
                 batch_size=32, unroll_step=8, reward_decay=0.99, learning_rate=1e-4,
                 train_freq=1, memory_size=20000, target_update=2000, eval_obs=None,
                 use_dueling=True, use_double=True, use_episode_train=False,
                 custom_view_space=None, custom_feature_space=None):
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
        custom_feature_space: tuple
            customized feature space
        custom_view_space: tuple
            customized feature space
        """
        TFBaseModel.__init__(self, env, handle, name, "tfdrqn")
        # ======================== set config  ========================
        self.env = env
        self.handle = handle
        self.view_space = custom_view_space or env.get_view_space(handle)
        self.feature_space = custom_feature_space or env.get_feature_space(handle)
        self.num_actions = env.get_action_space(handle)[0]

        self.batch_size   = batch_size
        self.unroll_step  = unroll_step
        self.handle = handle
        self.name = name
        self.learning_rate= learning_rate
        self.train_freq   = train_freq   # train time of every sample (s,a,r,s')
        self.target_update= target_update # target network update frequency
        self.eval_obs = eval_obs

        self.use_dueling  = use_dueling
        self.use_double   = use_double
        self.use_episode_train = use_episode_train
        self.skip_error   = 0
        self.pad_before_len = unroll_step - 1

        self.agent_states = {}
        self.train_ct = 0

        # ======================= build network =======================
        # input place holder
        self.target = tf.placeholder(tf.float32, [None])

        self.input_view    = tf.placeholder(tf.float32, (None,) + self.view_space, name="input_view")
        self.input_feature = tf.placeholder(tf.float32, (None,) + self.feature_space, name="input_feature")
        self.action = tf.placeholder(tf.int32, [None], name="action")
        self.mask   = tf.placeholder(tf.float32, [None], name="mask")

        self.batch_size_ph   = tf.placeholder(tf.int32, [])
        self.unroll_step_ph = tf.placeholder(tf.int32, [])

        # build graph
        with tf.variable_scope(self.name):
            with tf.variable_scope("eval_net_scope"):
                self.eval_scope_name   = tf.get_variable_scope().name
                self.qvalues, self.state_in, self.rnn_state = \
                    self._create_network(self.input_view, self.input_feature)

            with tf.variable_scope("target_net_scope"):
                self.target_scope_name = tf.get_variable_scope().name
                self.target_qvalues, self.target_state_in, self.target_rnn_state = \
                    self._create_network(self.input_view, self.input_feature)

        # loss
        self.gamma = reward_decay
        self.actions_onehot = tf.one_hot(self.action, self.num_actions)
        self.td_error = tf.square(
                self.target - tf.reduce_sum(tf.multiply(self.actions_onehot, self.qvalues), axis=1)
            )
        #self.loss = tf.reduce_mean(self.td_error)
        self.loss = tf.reduce_sum(self.td_error * self.mask) / tf.reduce_sum(self.mask)

        # train op (clip gradient)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        gradients, variables = zip(*optimizer.compute_gradients(self.loss))
        gradients, _ = tf.clip_by_global_norm(gradients, 10.0)
        self.train_op = optimizer.apply_gradients(zip(gradients, variables))

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

        # init memory buffers
        self.memory_size = memory_size
        self.replay_buffer_lens = collections.deque(maxlen=memory_size)
        self.replay_buffer = collections.deque(maxlen=memory_size)
        # item format [views, features, actions, rewards, terminals, masks, len]

        # init training buffers
        self.view_buf = np.empty((1,) + self.view_space)
        self.feature_buf = np.empty((1,) + self.feature_space)
        self.action_buf, self.reward_buf = np.empty(1, dtype=np.int32), np.empty(1)
        self.terminal_buf = np.empty(1, dtype=np.bool)

    def _create_network(self, input_view, input_feature, reuse=None):
        """define computation graph of network

        Parameters
        ----------
        input_view: tf.tensor
        input_feature: tf.tensor
            the input tensor
        """
        kernel_num  = [32, 32]
        hidden_size = [256]

        # conv
        h_conv1 = tf.layers.conv2d(input_view, filters=kernel_num[0], kernel_size=3,
                                   activation=tf.nn.relu, name="conv1", reuse=reuse)
        h_conv2 = tf.layers.conv2d(h_conv1, filters=kernel_num[1], kernel_size=3,
                                   activation=tf.nn.relu, name="conv2", reuse=reuse)
        flatten_view = tf.reshape(h_conv2, [-1, np.prod([v.value for v in h_conv2.shape[1:]])])
        h_view = tf.layers.dense(flatten_view, units=hidden_size[0], activation=tf.nn.relu,
                                 name="dense_view", reuse=reuse)

        h_emb = tf.layers.dense(input_feature,  units=hidden_size[0], activation=tf.nn.relu,
                                name="dense_emb", reuse=reuse)

        dense = tf.concat([h_view, h_emb], axis=1)

        # RNN
        state_size = hidden_size[0] * 2
        rnn_cell = tf.contrib.rnn.GRUCell(num_units=state_size)

        rnn_in = tf.reshape(dense, shape=[self.batch_size_ph, self.unroll_step_ph, state_size])
        state_in = rnn_cell.zero_state(self.batch_size_ph, tf.float32)
        rnn, rnn_state = tf.nn.dynamic_rnn(
            cell=rnn_cell, inputs=rnn_in, dtype=tf.float32, initial_state=state_in
        )
        rnn = tf.reshape(rnn, shape=[-1, state_size])

        if self.use_dueling:
            value = tf.layers.dense(dense, units=1, name="dense_value", reuse=reuse)
            advantage = tf.layers.dense(dense, units=self.num_actions, use_bias=False,
                                        name="dense_advantage", reuse=reuse)

            qvalues = value + advantage - tf.reduce_mean(advantage, axis=1, keep_dims=True)
        else:
            qvalues = tf.layers.dense(rnn, units=self.num_actions)

        self.state_size = state_size
        return qvalues, state_in, rnn_state

    def _get_agent_states(self, ids):
        """get hidden state of agents"""
        n = len(ids)
        states = np.empty([n, self.state_size])
        default = np.zeros([self.state_size])
        for i in range(n):
            states[i] = self.agent_states.get(ids[i], default)
        return states

    def _set_agent_states(self, ids, states):
        """set hidden state for agents"""
        if len(ids) <= len(self.agent_states) * 0.5:
            self.agent_states = {}
        for i in range(len(ids)):
            self.agent_states[ids[i]] = states[i]

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
        n = len(ids)

        states = self._get_agent_states(ids)
        qvalues, states = self.sess.run([self.qvalues, self.rnn_state], feed_dict={
            self.input_view:    view,
            self.input_feature: feature,
            self.state_in:      states,
            self.batch_size_ph: n,
            self.unroll_step_ph: 1
        })
        self._set_agent_states(ids, states)
        best_actions = np.argmax(qvalues, axis=1)

        if policy == 'e_greedy':
            random = np.random.randint(self.num_actions, size=(n,))
            cond = np.random.uniform(0, 1, size=(n,)) < eps
            ret = np.where(cond, random, best_actions)
        elif policy == 'greedy':
            ret = best_actions

        return ret.astype(np.int32)

    def _calc_target(self, next_view, next_feature, rewards, terminal, batch_size, unroll_step):
        """calculate target value"""
        n = len(rewards)
        if self.use_double:
            t_qvalues, qvalues = self.sess.run([self.target_qvalues, self.qvalues], feed_dict={
                self.input_view:      next_view,
                self.input_feature:   next_feature,
                # self.state_in:        state_in,
                # self.target_state_in: state_in,
                self.batch_size_ph:   batch_size,
                self.unroll_step_ph:  unroll_step})
            # ignore the first value (the first value is for computing correct hidden state)
            # t_qvalues = t_qvalues.reshape([-1, unroll_step, self.num_actions])
            # t_qvalues = t_qvalues[:, 1:, :].reshape([-1, self.num_actions])
            # qvalues = qvalues.reshape([-1, unroll_step, self.num_actions])
            # qvalues = qvalues[:, 1:, :].reshape([-1, self.num_actions])
            next_value = t_qvalues[np.arange(n), np.argmax(qvalues, axis=1)]
        else:
            t_qvalues = self.sess.run(self.target_qvalues, feed_dict={
                self.input_view:      next_view,
                self.input_feature:   next_feature,
                # self.target_state_in: state_in,
                self.batch_size_ph:   batch_size,
                self.unroll_step_ph:  unroll_step})
            # t_qvalues = t_qvalues.reshape([-1, unroll_step, self.num_actions])
            # t_qvalues = t_qvalues[:,1:,:].reshape([-1, self.num_actions])
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

            item = [v, f, a, r, terminal, mask, m]
            self.replay_buffer_lens.append(m)
            self.replay_buffer.append(item)

            n += m
        return n

    def train(self, sample_buffer, print_every=500):
        """ add new samples in sample_buffer to replay buffer and train
        do not keep hidden state (split episode into short sequences)

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
        unroll_step = self.unroll_step

        # calc sample weight of episodes (i.e. their lengths)
        replay_buffer = self.replay_buffer
        replay_lens_sum = np.sum(self.replay_buffer_lens)
        weight = np.array(self.replay_buffer_lens, dtype=np.float32) / replay_lens_sum

        n_batches = self.train_freq * add_num / (batch_size * (unroll_step - self.skip_error))
        if n_batches == 0:
            return 0, 0

        max_ = batch_size * unroll_step
        batch_view = np.zeros((max_+1,) + self.view_space, dtype=np.float32)
        batch_feature = np.zeros((max_+1,) + self.feature_space, dtype=np.float32)
        batch_action = np.zeros((max_,), dtype=np.int32)
        batch_reward = np.zeros((max_,), dtype=np.float32)
        batch_terminal = np.zeros((max_,), dtype=np.bool)
        batch_mask = np.zeros((max_,), dtype=np.float32)

        # calc batch number
        n_batches = int(self.train_freq * add_num / (batch_size * (unroll_step - self.skip_error)))
        print("batches: %d  add: %d  replay_len: %d/%d" %
              (n_batches, add_num, len(self.replay_buffer), self.memory_size))

        ct = 0
        total_loss = 0
        start_time = time.time()
        # train batches
        for i in range(n_batches):
            indexes = np.random.choice(len(replay_buffer), self.batch_size, p=weight)

            batch_mask[:] = 0

            for j in range(batch_size):
                item = replay_buffer[indexes[j]]
                v, f, a, r, t = item[0], item[1], item[2], item[3], item[4]
                length = len(v)

                start = np.random.randint(length)
                real_step = min(length - start, unroll_step)

                beg = j * unroll_step
                batch_view[beg:beg+real_step]     = v[start:start+real_step]
                batch_feature[beg:beg+real_step]  = f[start:start+real_step]
                batch_action[beg:beg+real_step]   = a[start:start+real_step]
                batch_reward[beg:beg+real_step]   = r[start:start+real_step]
                batch_terminal[beg:beg+real_step] = t[start:start+real_step]
                batch_mask[beg:beg+real_step]     = 1.0

                if not t[start+real_step-1]:
                    batch_mask[beg+real_step-1] = 0

            # collect trajectories from different IDs to a single buffer
            target = self._calc_target(batch_view[1:], batch_feature[1:],
                                       batch_reward, batch_terminal, batch_size, unroll_step)

            ret = self.sess.run([self.train_op, self.loss], feed_dict={
                self.input_view:      batch_view[:-1],
                self.input_feature:   batch_feature[:-1],
                self.action:          batch_action,
                self.target:          target,
                self.mask:            batch_mask,
                self.batch_size_ph:   batch_size,
                self.unroll_step_ph:  unroll_step,
            })
            loss = ret[1]
            total_loss += loss

            if ct % self.target_update == 0:
                self.sess.run(self.update_target_op)

            if ct % print_every == 0:
                print("batch %5d, loss %.6f, qvalue %.6f" % (ct, loss, self._eval(target)))
            ct += 1
            self.train_ct += 1

        total_time = time.time() - start_time
        step_average = total_time / max(1.0, (ct / 1000.0))
        print("batches: %d,  total time: %.2f,  1k average: %.2f" % (ct, total_time, step_average))

        return total_loss / ct if ct != 0 else 0, self._eval(target)

    def train_keep_hidden(self, sample_buffer, print_every=500):
        """ add new samples in sample_buffer to replay buffer and train
            keep hidden state (split episode into small sequence, but keep hidden states)
            this means must train some episodes continuously not fully random.
            to use this training scheme, you should also modify self._calc_target

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
        unroll_step = self.unroll_step

        # calc sample weight of episodes (i.e. their lengths)
        replay_buffer = self.replay_buffer
        replay_lens_sum = np.sum(self.replay_buffer_lens)
        weight = np.array(self.replay_buffer_lens, dtype=np.float32) / replay_lens_sum
   
        max_len  = self._div_round(np.max(self.replay_buffer_lens), unroll_step)
        n_batches = self.train_freq * add_num / (batch_size * unroll_step)
        if n_batches == 0:
            return 0, 0

        # allocate buffer
        max_ = batch_size * max_len
        batch_view     = np.zeros((max_+1,) + self.view_space, dtype=np.float32)
        batch_feature  = np.zeros((max_+1,) + self.feature_space, dtype=np.float32)
        batch_action   = np.zeros((max_,), dtype=np.int32)
        batch_reward   = np.zeros((max_,), dtype=np.float32)
        batch_terminal = np.zeros((max_,), dtype=np.bool)
        batch_mask     = np.zeros((max_,), dtype=np.float32)
        batch_hidden   = np.zeros((batch_size, self.state_size), dtype=np.float32)
        batch_pick     = np.zeros((batch_size, max_len), dtype=np.bool)
        pick_buffer    = np.arange(max_, dtype=np.int32)

        print("batches: %d  add: %d  replay_len: %d, %d/%d" %
              (n_batches, add_num, replay_lens_sum, len(self.replay_buffer), self.memory_size))

        start_time = time.time()
        total_loss = 0
        ct = 0
        while ct < n_batches:
            # random sample agent episodes (sequence)
            indexs = np.random.choice(len(replay_buffer), self.batch_size, p=weight)
            train_length = 0
            to_sort = []

            # collect length and sort
            for j, index in enumerate(indexs):
                length = replay_buffer[index][-1]
                length = self._div_round(length, unroll_step)
                train_length = max(train_length, length)
                to_sort.append([index, length])
            to_sort.sort(key=lambda x: -x[1])

            # merge short episodes to long episodes (use greedy method)
            merged = [False for _ in range(batch_size)]
            rows = []
            for j in range(len(to_sort)):
                if merged[j]:
                    continue
                row = [to_sort[j][0]]
                now_len = to_sort[j][1]
                if True: # use compress
                    for k in range(j+1, batch_size):
                        if now_len + to_sort[k][1] <= train_length:
                            row.append(to_sort[k][0])
                            now_len += to_sort[k][1]
                            merged[k] = True
                    rows.append(row)
            n_rows = len(rows)

            batch_reset = np.zeros((train_length, batch_size), dtype=np.bool)
            batch_mask[:] = 0

            # copy from replay buffer to batch buffer
            for j, row in enumerate(rows):
                beg = j * max_len
                init_beg = beg
                # fill a row
                for index in row:
                    v, f, a, r, terminal, mask, x = replay_buffer[index]

                    batch_reset[(beg - init_beg)/unroll_step, j] = True
                    batch_view[beg:beg+x] = v
                    batch_feature[beg:beg+x] = f
                    batch_action[beg:beg+x] = a
                    batch_reward[beg:beg+x] = r
                    batch_terminal[beg:beg+x] = terminal
                    batch_mask[beg:beg+x] = mask

                    beg += self._div_round(x, unroll_step)

            # train steps
            for j in range((train_length + unroll_step - 1) / unroll_step):
                batch_pick[:] = False
                batch_pick[:n_rows, j * unroll_step:(j+1) * unroll_step] = True

                pick = pick_buffer[batch_pick.reshape(-1)].reshape(n_rows, unroll_step)
                next_pick = np.empty((n_rows, unroll_step + 1), dtype=np.int32)  # next pick choose one more state than pick
                next_pick[:, :unroll_step] = pick
                next_pick[:, unroll_step]  = pick[:, -1] + 1
                pick = pick.reshape(-1)
                next_pick = next_pick.reshape(-1)

                steps = len(pick) / n_rows
                assert steps > 0
                if np.sum(batch_mask[pick]) < 1:
                    continue

                batch_hidden[batch_reset[j]] = np.zeros_like(batch_hidden[0])

                batch_target = self._calc_target(batch_view[next_pick], batch_feature[next_pick],
                                                 batch_reward[pick], batch_terminal[pick], batch_hidden[:n_rows],
                                                 n_rows, steps + 1)
                ret = self.sess.run(
                    [self.train_op, self.loss, self.rnn_state],
                    feed_dict={
                        self.input_view:      batch_view[pick],
                        self.input_feature:   batch_feature[pick],
                        self.action:          batch_action[pick],
                        self.target:          batch_target,
                        self.mask:            batch_mask[pick],
                        self.state_in:        batch_hidden[:n_rows],
                        self.batch_size_ph:   n_rows,
                        self.unroll_step_ph:  steps
                })
                loss, batch_hidden[:n_rows] = ret[1], ret[2]
                total_loss += loss

                if ct % self.target_update == 0:
                    self.sess.run(self.update_target_op)

                if ct % print_every == 0:
                    print("batches %5d, mask %d/%d (%d), loss %.6f, qvalue %.6f" %
                          (ct, sum(batch_mask), n_rows * train_length, n_rows, loss, self._eval(batch_target)))
                ct += 1
                self.train_ct += 1

        total_time = time.time() - start_time
        step_average = total_time / max(1.0, (ct / 1000.0))
        print("batches: %d,  total time: %.2f,  1k average: %.2f" % (ct, total_time, step_average))

        return round(total_loss / ct if ct != 0 else 0, 6), self._eval(batch_target)

    @staticmethod
    def _div_round(x, divisor):
        """round up to nearest integer that are divisible by divisor"""
        return (x + divisor - 1) / divisor * divisor

    def _eval(self, target):
        """evaluate estimated q value"""
        if self.eval_obs is None:
            return np.mean(target)
        else:
            return np.mean(self.sess.run(self.target_qvalues, feed_dict={
                self.input_view:      self.eval_obs[0],
                self.input_feature:   self.eval_obs[1],
                self.batch_size_ph:   self.eval_obs[0].shape[0],
                self.unroll_step_ph: 1
            }))

    def get_info(self):
        """get information of model"""
        return "tfdrqn train_time: %d" % (self.train_ct)
