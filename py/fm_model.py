import os, threading, random, time, sys
import tensorflow as tf
from tensorflow.python.framework import ops
from py.fm_ops import fm_ops

GLOBAL_PARAM_NAME = "global_params"
FM_VARIALBE_SCOPE = "fm"

class FmHandler:
    def __init__(self):
        self.lock = threading.Lock()
        self.loss = 0.0
        self.example_num = 0
        self.start_time = time.time()

    def log_stat(self, loss, example_num):
        self.lock.acquire()
        self.loss += loss
        self.example_num += example_num
        if self.example_num % 5000000 < example_num:
            self.print_stat()
        self.lock.release()

    def print_stat(self):
        t = time.time() - self.start_time
        print 'Ex num: %d; Avg. loss: %.5f; Time: %.4f; Speed: %.1f exs/sec.'%(self.example_num, self.loss / self.example_num, t, self.example_num / t)
        sys.stdout.flush()

class TrainHandler(FmHandler):
    def __init__(self, sess, model, qidx):
        FmHandler.__init__(self)
        self.sess = sess
        self.model = model
        self.qidx = qidx
    def run(self):
        _, loss, example_num = self.sess.run([self.model.opt, self.model.loss, self.model.line_num], feed_dict = {self.model.queue_idx: self.qidx})
        FmHandler.log_stat(self, loss, example_num)

class ValidationHandler(FmHandler):
    def __init__(self, sess, model, qidx):
        FmHandler.__init__(self)
        self.sess = sess
        self.model = model
        self.qidx = qidx
    def run(self):
        loss, example_num = self.sess.run([self.model.loss, self.model.line_num], feed_dict = {self.model.queue_idx: self.qidx})
        FmHandler.log_stat(self, loss, example_num)

class PredictHandler:
    def __init__(self, sess, model, score_file_handler):
        self.sess = sess
        self.model = model
        self.score_file_handler = score_file_handler
    def run(self):
        pred_score = self.sess.run(self.model.pred_score, feed_dict = {self.model.queue_idx: 0})
        for score in pred_score:
            self.score_file_handler.write(str(score) + '\n')

class FmModel:
    def __init__(self, vocabulary_size, factor_num, init_value_range):
        self.sess = tf.Session()
        with tf.variable_scope(FM_VARIALBE_SCOPE) as fm_scope:
            self.global_params = tf.get_variable(GLOBAL_PARAM_NAME, [vocabulary_size, factor_num + 1], tf.float32, tf.random_uniform_initializer(-init_value_range, init_value_range))
        self.saver = tf.train.Saver([self.global_params])
        self.sess.run(tf.initialize_all_variables())

    def train(self, train_files, validation_files, optimizer, epoch_num, thread_num, batch_size, factor_lambda, bias_lambda):
        queue_list = []
        for i in range(epoch_num):
            queue = tf.FIFOQueue(len(train_files), tf.string)
            random.shuffle(train_files)
            self.sess.run(queue.enqueue_many([train_files]))
            self.sess.run(queue.close())
            queue_list.append(queue)
        if validation_files != None:
            for i in range(epoch_num):
                queue = tf.FIFOQueue(len(validation_files), tf.string)
                random.shuffle(validation_files)
                self.sess.run(queue.enqueue_many([validation_files]))
                self.sess.run(queue.close())
                queue_list.append(queue)
        model = FmModelBase(queue_list, optimizer, batch_size, factor_lambda, bias_lambda)
        variables = set(tf.all_variables())
        variables.remove(self.global_params)
        self.sess.run(tf.initialize_variables(variables))
        for i in range(epoch_num):
            print '------  Epoch %d ------'%i

            print 'Start training ... '
            train_handler = TrainHandler(self.sess, model, i)
            model.multi_thread_run(thread_num, train_handler)
            print 'Finish training.'
            train_handler.print_stat()

            if validation_files != None:
                print 'Start validation ... '
                validation_handler = ValidationHandler(self.sess, model, epoch_num + i)
                model.multi_thread_run(thread_num, validation_handler)
                print 'Finish validation.'
                validation_handler.print_stat()

    def predict(self, pred_file, score_file):
        queue = tf.FIFOQueue(1, tf.string)
        self.sess.run(queue.enqueue(pred_file))
        self.sess.run(queue.close())
        model = FmModelBase([queue], None, 1000, 0, 0)
        with open(score_file, 'w') as f:
            predict_handler = PredictHandler(self.sess, model, f)
            model.single_thread_run(predict_handler)

    def save(self, model_file):
        self.saver.save(self.sess, model_file, write_meta_graph = False)

    def load(self, model_file):
        self.saver.restore(self.sess, model_file)

    def close(self):
        self.sess.close()

class FmModelBase:
    def __init__(self, queue_list, optimizer, batch_size, factor_lambda, bias_lambda):
        self.queue_idx = tf.placeholder(dtype = tf.int32)
        filename_queue = tf.QueueBase.from_list(self.queue_idx, queue_list)
        reader = tf.TextLineReader()
        keys, lines = reader.read_up_to(filename_queue, batch_size)
        self.line_num = tf.size(lines)
        labels, ori_ids, feature_ids, feature_vals, feature_poses = fm_ops.fm_parser(lines)
        with tf.variable_scope(FM_VARIALBE_SCOPE, reuse = True):
            global_params = tf.get_variable(GLOBAL_PARAM_NAME)
        local_params = tf.gather(global_params, ori_ids)
        self.pred_score, reg_score = fm_ops.fm_scorer(feature_ids, local_params, feature_vals, feature_poses, factor_lambda, bias_lambda)
        self.loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(self.pred_score, labels))
        if optimizer != None:
            self.opt = optimizer.minimize(self.loss + reg_score)

    def multi_thread_run(self, thread_num, handler):
        def thread_target(handler, coord):
          try:
            while not coord.should_stop():
                handler.run()
          except tf.errors.OutOfRangeError:
            pass
          except Exception, e:
            print str(e)
            raise
          finally:
            coord.request_stop()

        coord = tf.train.Coordinator()
        train_threads = [threading.Thread(target = thread_target, args = (handler,coord)) for i in range(thread_num)]
        for th in train_threads: th.start()
        coord.join(train_threads, stop_grace_period_secs=5)

    def single_thread_run(self, handler):
        try:
            while True:
                handler.run()
        except tf.errors.OutOfRangeError:
            pass
        except Exception, e:
            print str(e)
            raise
