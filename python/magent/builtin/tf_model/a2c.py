""" advantage actor critic """
import os

import numpy as np
import tensorflow as tf

from .base import TFBaseModel


class AdvantageActorCritic(TFBaseModel):
    def __init__(self, env, handle, name, learning_rate=1e-3,
                 batch_size=64, reward_decay=0.99, eval_obs=None,
                 train_freq=1, value_coef=0.1, ent_coef=0.08, use_comm=False,
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
        eval_obs: numpy array
            evaluation set of observation
        train_freq: int
            mean training times of a sample
        ent_coef: float
            weight of entropy loss in total loss
        value_coef: float
            weight of value loss in total loss
        use_comm: bool
            whether use CommNet
        custom_feature_space: tuple
            customized feature space
        custom_view_space: tuple
            customized feature space
        """
        TFBaseModel.__init__(self, env, handle, name, "tfa2c")
        # ======================== set config  ========================
        self.env = env
        self.handle = handle
        self.name = name
        self.view_space = custom_view_space or env.get_view_space(handle)
        self.feature_space = custom_feature_space or env.get_feature_space(handle)
        self.num_actions  = env.get_action_space(handle)[0]
        self.reward_decay = reward_decay

        self.batch_size   = batch_size
        self.learning_rate= learning_rate
        self.train_freq   = train_freq   # train time of every sample (s,a,r,s')

        self.value_coef = value_coef     # coefficient of value in the total loss
        self.ent_coef = ent_coef         # coefficient of entropy in the total loss

        self.train_ct = 0
        self.use_comm = use_comm

        # ======================= build network =======================
        with tf.name_scope(self.name):
            self._create_network(self.view_space, self.feature_space)

        # init tensorflow session
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

        # init training buffers
        self.view_buf = np.empty((1,) + self.view_space)
        self.feature_buf = np.empty((1,) + self.feature_space)
        self.action_buf = np.empty(1, dtype=np.int32)
        self.reward_buf = np.empty(1, dtype=np.float32)

    def _commnet_block(self, n, hidden, skip, name, hidden_size):
        """a block of CommNet

        Parameters
        ----------
        n: int
            number of agent
        hidden: tf.tensor
            hidden layer input
        skip: tf.tensor
            skip connection
        name: str
        hidden_size: int
        """
        mask = (tf.ones((n, n)) - tf.eye(n))
        mask *= tf.where(n > 1, 1.0 / (tf.cast(n, tf.float32) - 1.0), 0)

        C = tf.get_variable(name + "_C", shape=(hidden_size, hidden_size))
        H = tf.get_variable(name + "_H", shape=(hidden_size, hidden_size))

        message = tf.matmul(mask, hidden)

        return tf.tanh(tf.matmul(message, C) + tf.matmul(hidden, H) + skip)

    def _commnet(self, n, dense, hidden_size, n_step=2):
        """ CommNet Learning Multiagent Communication with Backpropagation by S. Sukhbaatar et al. NIPS 2016

        Parameters
        ----------
        n: int
            number of agent
        hidden_size: int
        n_step: int
            communication step

        Returns
        -------
        h: tf.tensor
            hidden units after CommNet
        """
        skip = dense

        h = dense
        for i in range(n_step):
            h = self._commnet_block(n, h, skip, "step_%d" % i, hidden_size)

        return h

    def _create_network(self, view_space, feature_space):
        """define computation graph of network

        Parameters
        ----------
        view_space: tuple
        feature_space: tuple
            the input shape
        """
        # input
        input_view    = tf.placeholder(tf.float32, (None,) + view_space)
        input_feature = tf.placeholder(tf.float32, (None,) + feature_space)
        action = tf.placeholder(tf.int32, [None])
        reward = tf.placeholder(tf.float32, [None])
        num_agent = tf.placeholder(tf.int32, [])

        kernel_num = [32, 32]
        hidden_size = [256]

        # fully connected
        flatten_view = tf.reshape(input_view, [-1, np.prod([v.value for v in input_view.shape[1:]])])
        h_view = tf.layers.dense(flatten_view, units=hidden_size[0], activation=tf.nn.relu)

        h_emb = tf.layers.dense(input_feature,  units=hidden_size[0], activation=tf.nn.relu)

        dense = tf.concat([h_view, h_emb], axis=1)
        dense = tf.layers.dense(dense, units=hidden_size[0] * 2, activation=tf.nn.relu)

        if self.use_comm:
            dense = self._commnet(num_agent, dense, dense.shape[-1].value)

        policy = tf.layers.dense(dense, units=self.num_actions, activation=tf.nn.softmax)
        policy = tf.clip_by_value(policy, 1e-10, 1-1e-10)
        value = tf.layers.dense(dense, units=1)
        value = tf.reshape(value, (-1,))
        advantage = tf.stop_gradient(reward - value)

        action_mask = tf.one_hot(action, self.num_actions)

        log_policy = tf.log(policy + 1e-6)
        log_prob = tf.reduce_sum(log_policy * action_mask, axis=1)
        pg_loss = -tf.reduce_mean(advantage * log_prob)
        vf_loss = self.value_coef * tf.reduce_mean(tf.square(reward - value))
        neg_entropy = self.ent_coef * tf.reduce_mean(tf.reduce_sum(policy * log_policy, axis=1))
        total_loss = pg_loss + vf_loss + neg_entropy

        # train op (clip gradient)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        gradients, variables = zip(*optimizer.compute_gradients(total_loss))
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        self.train_op = optimizer.apply_gradients(zip(gradients, variables))

        train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(total_loss)

        self.input_view = input_view
        self.input_feature = input_feature
        self.action     = action
        self.reward     = reward
        self.num_agent  = num_agent

        self.policy, self.value = policy, value
        self.train_op = train_op
        self.pg_loss, self.vf_loss, self.reg_loss = pg_loss, vf_loss, neg_entropy
        self.total_loss = total_loss

    def infer_action(self, raw_obs, ids, *args, **kwargs):
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

        policy = self.sess.run(self.policy, {self.input_view: view,
                                             self.input_feature: feature,
                                             self.num_agent: n})
        actions = np.arange(self.num_actions)

        ret = np.empty(n, dtype=np.int32)
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
            n += len(episode.rewards)

        # resize to the new size
        self.view_buf.resize((n,) + self.view_space)
        self.feature_buf.resize((n,) + self.feature_space)
        self.action_buf.resize(n)
        self.reward_buf.resize(n)
        view, feature  = self.view_buf, self.feature_buf
        action, reward = self.action_buf, self.reward_buf

        ct = 0
        gamma = self.reward_decay
        # collect episodes from multiple separate buffers to a continuous buffer
        for episode in sample_buffer.episodes():
            v, f, a, r = episode.views, episode.features, episode.actions, episode.rewards
            m = len(episode.rewards)

            r = np.array(r)
            keep = self.sess.run(self.value, feed_dict={
                self.input_view: [v[-1]],
                self.input_feature: [f[-1]],
                self.num_agent: 1
            })[0]
            for i in reversed(range(m)):
                keep = keep * gamma + r[i]
                r[i] = keep

            view[ct:ct+m] = v
            feature[ct:ct+m] = f
            action[ct:ct+m]  = a
            reward[ct:ct+m] = r
            ct += m

        assert n == ct

        # train
        _, pg_loss, vf_loss, ent_loss, state_value = self.sess.run(
            [self.train_op, self.pg_loss, self.vf_loss, self.reg_loss, self.value], feed_dict={
                self.input_view:    view,
                self.input_feature: feature,
                self.action:        action,
                self.reward:        reward,
                self.num_agent:     len(reward)
        })
        print("sample", n, pg_loss, vf_loss, ent_loss)

        return [pg_loss, vf_loss, ent_loss], np.mean(state_value)

    def get_info(self):
        """get information of the model

        Returns
        -------
        info: string
        """
        return "a2c train_time: %d" % (self.train_ct)
