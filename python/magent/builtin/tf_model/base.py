import os
import tensorflow as tf

from magent.model import BaseModel


class TFBaseModel(BaseModel):
    """base model for tensorflow model"""
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
        dir_name = os.path.join(dir_name, self.name)
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name)
        saver = tf.train.Saver(model_vars)
        saver.save(self.sess, os.path.join(dir_name, (self.subclass_name + "_%d") % epoch))

    def load(self, dir_name, epoch=0, name=None):
        """save model to dir

        Parameters
        ----------
        dir_name: str
            name of the directory
        epoch: int
        """
        if name is None or name == self.name:  # the name of saved model is the same as ours
            dir_name = os.path.join(dir_name, self.name)
            model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name)
            saver = tf.train.Saver(model_vars)
            saver.restore(self.sess, os.path.join(dir_name, (self.subclass_name + "_%d") % epoch))
        else:  # load a checkpoint with different name
            backup_graph = tf.get_default_graph()
            kv_dict = {}

            # load checkpoint from another saved graph
            with tf.Graph().as_default(), tf.Session() as sess:
                tf.train.import_meta_graph(os.path.join(dir_name, name, (self.subclass_name + "_%d") % epoch + ".meta"))
                dir_name = os.path.join(dir_name, name)
                model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, name)
                sess.run(tf.global_variables_initializer())
                saver = tf.train.Saver(model_vars)
                saver.restore(sess, os.path.join(dir_name, (self.subclass_name + "_%d") % epoch))
                for item in tf.global_variables():
                    kv_dict[item.name] = sess.run(item)

            # assign to now graph
            backup_graph.as_default()
            model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name)
            for item in model_vars:
                old_name = item.name.replace(self.name, name)
                self.sess.run(tf.assign(item, kv_dict[old_name]))

