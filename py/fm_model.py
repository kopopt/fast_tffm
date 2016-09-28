import os, threading, random, time, sys
import tensorflow as tf
from tensorflow.python.framework import ops
from py.fm_ops import fm_ops

class ModelStat:
  def __init__(self, name):
    self.int_delta = tf.placeholder(dtype = tf.int32)
    self.float_delta = tf.placeholder(dtype = tf.float32)
    self.total_loss = tf.Variable(0.0, name = name + '_loss', trainable = False)
    self.incre_total_loss = self.total_loss.assign_add(self.float_delta, True)
    self.total_example_num = tf.Variable(0, name = name + '_example_num', trainable = False)
    self.incre_total_example_num = self.total_example_num.assign_add(self.int_delta, True)

  def update(self, sess, loss_delta, example_num_delta):
    return sess.run([self.incre_total_loss, self.incre_total_example_num], feed_dict = {self.float_delta: loss_delta, self.int_delta: example_num_delta})

  def eval(self, sess):
    return sess.run([self.total_loss, self.total_example_num])

class FmModelBase:
  def __init__(self, queue_size, epoch_num, vocabulary_size, vocabulary_block_num, factor_num, init_value_range, optimizer, batch_size, factor_lambda, bias_lambda):
    with self.main_ps_device():
      self.filename_queue = tf.FIFOQueue(queue_size, [tf.int32, tf.bool, tf.string], shared_name = 'global_queue')

    with self.default_device():
      self.finished_worker_num = tf.Variable(0)
      self.incre_finshed_worker_num = self.finished_worker_num.assign_add(1, True)
      self.model_loaded = tf.Variable(False)
      self.set_model_loaded = self.model_loaded.assign(True)
      self.training_stat = []
      self.validation_stat = []
      for i in range(epoch_num):
        self.training_stat.append(ModelStat('training_%d'%i))
        self.validation_stat.append(ModelStat('validation_%d'%i))

      self.epoch_id = tf.placeholder(dtype = tf.int32)
      self.is_training = tf.placeholder(dtype = tf.bool)
      self.filename = tf.placeholder(dtype = tf.string)
      self.filename_enqueue_op = self.filename_queue.enqueue((self.epoch_id, self.is_training, self.filename))
      self.filename_dequeue_op = self.filename_queue.dequeue()
      self.filename_close_queue_op = self.filename_queue.close()

      self.vocab_blocks = []                   
      vocab_size_per_block = vocabulary_size / vocabulary_block_num + 1
      for i in range(vocabulary_block_num):
        self.vocab_blocks.append(tf.Variable(tf.random_uniform([vocab_size_per_block, factor_num + 1], -init_value_range, init_value_range), name = 'vocab_block_%d'%i))
      self.file_id = tf.placeholder(dtype = tf.int32)
      self.file_name = tf.placeholder(dtype = tf.string)
      labels, ori_ids, feature_ids, feature_vals, feature_poses = fm_ops.fm_parser(self.file_id, self.file_name, batch_size)
      self.example_num = tf.size(labels)
      local_params = tf.nn.embedding_lookup(self.vocab_blocks, ori_ids)
      self.pred_score, reg_score = fm_ops.fm_scorer(feature_ids, local_params, feature_vals, feature_poses, factor_lambda, bias_lambda)
      self.loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(self.pred_score, labels))
      if optimizer != None:
        self.opt = optimizer.minimize(self.loss + reg_score)
      self.init_vars = tf.initialize_all_variables()
      self.saver = tf.train.Saver(self.vocab_blocks)
  
  def main_ps_device(self):
    raise NotImplementedError("Subclasses should implement this!")

  def default_device(self):
    raise NotImplementedError("Subclasses should implement this!")

class DistFmModel(FmModelBase):
  def __init__(self, queue_size, cluster, task_index, epoch_num, vocabulary_size, vocabulary_block_num, factor_num, init_value_range, optimizer, batch_size, factor_lambda, bias_lambda):
    self.task_index = task_index
    self.cluster = cluster
    FmModelBase.__init__(self, queue_size, epoch_num, vocabulary_size, vocabulary_block_num, factor_num, init_value_range, optimizer, batch_size, factor_lambda, bias_lambda)

  def main_ps_device(self):
    return tf.device('/job:ps/task:0')

  def default_device(self):
    return tf.device(tf.train.replica_device_setter(worker_device = '/job:worker/task:%d'%self.task_index, ps_device = "/job:ps", cluster = self.cluster))

class LocalFmModel(FmModelBase):
  def main_ps_device(self):
    return tf.device('/cpu:0')

  def default_device(self):
    return tf.device('/cpu:0')
