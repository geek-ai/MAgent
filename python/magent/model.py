""" base model classes"""

try:
    import thread
except ImportError:
    import _thread as thread

import multiprocessing
import multiprocessing.connection
import sys

import numpy as np

class BaseModel:
    def __init__(self, env, handle, *args, **kwargs):
        """ init

        Parameters
        ----------
        env: Environment
            env
        handle: GroupHandle
            handle of this group, handles are returned by env.get_handles()
        """
        pass

    def infer_action(self, raw_obs, ids, *args, **kwargs):
        """ infer action for a group of agents

        Parameters
        ----------
        raw_obs: tuple
            raw_obs is a tuple of (view, feature)
            view is a numpy array, its shape is n * view_width * view_height * n_channel
                                   it contains the spatial local observation for all the agents
            feature is a numpy array, its shape is n * feature_size
                                   it contains the non-spatial feature for all the agents
        ids: numpy array of int32
            the unique id of every agents
        args:
            additional custom args
        kwargs:
            additional custom args
        """
        pass

    def train(self, sample_buffer, **kwargs):
        """ feed new samples and train

        Parameters
        ----------
        sample_buffer: EpisodesBuffer
            a buffer contains transitions of agents

        Returns
        -------
        loss and estimated mean state value
        """
        return 0, 0    # loss, mean value

    def save(self, *args, **kwargs):
        """ save the model """
        pass

    def load(self, *args, **kwargs):
        """ load the model """
        pass


class NDArrayPackage:
    """wrapper for transferring numpy arrays by bytes"""
    def __init__(self, *args):
        if isinstance(args[0], np.ndarray):
            self.data = args
            self.info = [(x.shape, x.dtype) for x in args]
        else:
            self.data = None
            self.info = args[0]

        self.max_len = (1 << 30) / 4

    def send_to(self, conn, use_thread=False):
        assert self.data is not None

        def send_thread():
            for x in self.data:
                if np.prod(x.shape)  > self.max_len:
                    seg = int(self.max_len // np.prod(x.shape[1:]))
                    for pt in range(0, len(x), seg):
                        conn.send_bytes(x[pt:pt+seg])
                else:
                    conn.send_bytes(x)

        if use_thread:
            thread.start_new_thread(send_thread, ())
        else:
            send_thread()

    def recv_from(self, conn):
        bufs = []
        for info in self.info:
            buf = np.empty(shape=(int(np.prod(info[0])),), dtype=info[1])

            item_size = int(np.prod(info[0][1:]))
            if np.prod(info[0]) > self.max_len:
                seg = int(self.max_len // item_size)
                for pt in range(0, int(np.prod(info[0])), seg * item_size):
                    conn.recv_bytes_into(buf[pt:pt+seg * item_size])
            else:
               conn.recv_bytes_into(buf)
            bufs.append(buf.reshape(info[0]))
        return bufs


class ProcessingModel(BaseModel):
    """
    start a sub-processing to host a model,
    use pipe or socket for communication
    """
    def __init__(self, env, handle, name, port, sample_buffer_capacity=1000,
                 RLModel=None, **kwargs):
        """
        Parameters
        ----------
        env: environment
        handle: group handle
        name: str
            name of the model (be used when store model)
        port: int
            port of socket or suffix of pipe
        sample_buffer_capacity: int
            the maximum number of samples (s,r,a,s') to collect in a game round
        RLModel: BaseModel
            the RL algorithm class
        kwargs: dict
            arguments for RLModel
        """
        BaseModel.__init__(self, env, handle)

        assert RLModel is not None

        kwargs['env'] = env
        kwargs['handle'] = handle
        kwargs['name'] = name
        addr = 'magent-pipe-' + str(port)  # named pipe
        # addr = ('localhost', port) # socket
        proc = multiprocessing.Process(
            target=model_client,
            args=(addr, sample_buffer_capacity, RLModel, kwargs),
        )

        self.client_proc = proc
        proc.start()
        listener = multiprocessing.connection.Listener(addr)
        self.conn = listener.accept()

    def sample_step(self, rewards, alives, block=True):
        """record a step (should be followed by check_done)

        Parameters
        ----------
        block: bool
            if it is True, the function call will block
            if it is False, the caller must call check_done() afterward
                            to check/consume the return message
        """
        package = NDArrayPackage(rewards, alives)
        self.conn.send(["sample", package.info])
        package.send_to(self.conn)

        if block:
            self.check_done()

    def infer_action(self, raw_obs, ids, policy='e_greedy', eps=0, block=True):
        """ infer action

        Parameters
        ----------
        policy: str
            can be 'e_greedy' or 'greedy'
        eps: float
            used when policy is 'e_greedy'
        block: bool
            if it is True, the function call will block, and return actions
            if it is False, the function call won't block, the caller
                            must call fetch_action() to get actions

        Returns
        -------
        actions: numpy array (int32)
            see above
        """

        package = NDArrayPackage(raw_obs[0], raw_obs[1], ids)
        self.conn.send(["act", policy, eps, package.info])
        package.send_to(self.conn, use_thread=True)

        if block:
            info = self.conn.recv()
            return NDArrayPackage(info).recv_from(self.conn)[0]
        else:
            return None

    def fetch_action(self):
        """ fetch actions , fetch action after calling infer_action(block=False)

        Returns
        -------
        actions: numpy array (int32)
        """
        info = self.conn.recv()
        return NDArrayPackage(info).recv_from(self.conn)[0]

    def train(self, print_every=5000, block=True):
        """ train new data samples according to the model setting

        Parameters
        ----------
        print_every: int
            print training log info every print_every batches

        """
        self.conn.send(['train', print_every])

        if block:
            return self.fetch_train()

    def fetch_train(self):
        """ fetch result of train after calling train(block=False)

        Returns
        -------
        loss: float
            mean loss
        value: float
            mean state value
        """
        return self.conn.recv()

    def save(self, save_dir, epoch, block=True):
        """ save model

        Parameters
        ----------
        block: bool
            if it is True, the function call will block
            if it is False, the caller must call check_done() afterward
                            to check/consume the return message
        """

        self.conn.send(["save", save_dir, epoch])
        if block:
            self.check_done()

    def load(self, save_dir, epoch, name=None, block=True):
        """ load model

        Parameters
        ----------
        name: str
            name of the model (set when stored name is not the same as self.name)
        block: bool
            if it is True, the function call will block
            if it is False, the caller must call check_done() afterward
                            to check/consume the return message
        """
        self.conn.send(["load", save_dir, epoch, name])
        if block:
            self.check_done()

    def check_done(self):
        """ check return message of sub processing """
        assert self.conn.recv() == 'done'

    def quit(self):
        """ quit """
        proc = self.client_proc
        self.client_proc = None
        self.conn.send(["quit"])
        proc.join()

    def __del__(self):
        """ quit in destruction """
        if self.client_proc is not None:
            quit()


def model_client(addr, sample_buffer_capacity, RLModel, model_args):
    """target function for sub-processing to host a model

    Parameters
    ----------
    addr: socket address
    sample_buffer_capacity: int
        the maximum number of samples (s,r,a,s') to collect in a game round
    RLModel: BaseModel
        the RL algorithm class
    args: dict
        arguments to RLModel
    """
    import magent.utility

    model = RLModel(**model_args)
    sample_buffer = magent.utility.EpisodesBuffer(capacity=sample_buffer_capacity)

    conn = multiprocessing.connection.Client(addr)

    while True:
        cmd = conn.recv()
        if cmd[0] == 'act':
            policy = cmd[1]
            eps = cmd[2]
            array_info = cmd[3]

            view, feature, ids = NDArrayPackage(array_info).recv_from(conn)
            obs = (view, feature)

            acts = model.infer_action(obs, ids, policy=policy, eps=eps)
            package = NDArrayPackage(acts)
            conn.send(package.info)
            package.send_to(conn)
        elif cmd[0] == 'train':
            print_every = cmd[1]
            total_loss, value = model.train(sample_buffer, print_every=print_every)
            sample_buffer = magent.utility.EpisodesBuffer(sample_buffer_capacity)
            conn.send((total_loss, value))
        elif cmd[0] == 'sample':
            array_info = cmd[1]
            rewards, alives = NDArrayPackage(array_info).recv_from(conn)
            sample_buffer.record_step(ids, obs, acts, rewards, alives)
            conn.send("done")
        elif cmd[0] == 'save':
            savedir = cmd[1]
            n_iter = cmd[2]
            model.save(savedir, n_iter)
            conn.send("done")
        elif cmd[0] == 'load':
            savedir = cmd[1]
            n_iter = cmd[2]
            name = cmd[3]
            model.load(savedir, n_iter, name)
            conn.send("done")
        elif cmd[0] == 'quit':
            break
        else:
            print("Error: Unknown command %s" % cmd[0])
            break
