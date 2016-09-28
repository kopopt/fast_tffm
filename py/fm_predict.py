import tensorflow as tf
import time, os
from py.fm_model import LocalFmModel, DistFmModel

PREDICT_BATCH_SIZE = 10000

def _predict(sess, supervisor, is_master_worker, model, model_file, predict_files, score_path, need_to_init):
  with sess as sess:
    if is_master_worker:
      if need_to_init:
        sess.run(model.init_vars)
      if not os.path.exists(score_path):
        os.mkdir(score_path)
      model.saver.restore(sess, model_file)
      for fname in predict_files:
        sess.run(model.filename_enqueue_op, feed_dict = {model.epoch_id: 0, model.is_training: False, model.filename: fname})
      sess.run(model.filename_close_queue_op)
      sess.run(model.set_model_loaded)
    try:
      while not sess.run(model.model_loaded):
        print 'Waiting for the model to be loaded.'
        time.sleep(1)
      fid = 0
      while True:
        _, _, fname = sess.run(model.filename_dequeue_op)
        score_file = score_path + '/' + os.path.basename(fname) + '.score'
        print 'Start processing %s, scores written to %s ...'%(fname, score_file)
        with open(score_file, 'w') as o:
          while True:
            pred_score, example_num = sess.run([model.pred_score, model.example_num], feed_dict = {model.file_id: fid, model.file_name: fname})
            if example_num == 0: break
            for score in pred_score:
              o.write(str(score) + '\n')
        fid += 1
        time.sleep(100)
    except tf.errors.OutOfRangeError:
      pass
    except Exception, ex:
      if supervisor != None:
        supervisor.request_stop(ex)
      raise  

def dist_predict(ps_hosts, worker_hosts, job_name, task_idx, predict_files, vocabulary_size, vocabulary_block_num, factor_num, model_file, score_path):
  cluster = tf.train.ClusterSpec({'ps': ps_hosts, 'worker': worker_hosts})
  server = tf.train.Server(cluster, job_name = job_name, task_index = task_idx)
  if job_name == 'ps':
    server.join()
  elif job_name == 'worker':      
    model = DistFmModel(len(predict_files), cluster, task_idx, 0, vocabulary_size, vocabulary_block_num, factor_num, 0, None, PREDICT_BATCH_SIZE, 0, 0)
    sv = tf.train.Supervisor(is_chief = (task_idx == 0), init_op = model.init_vars)
    _predict(sv.managed_session(server.target, config = tf.ConfigProto(log_device_placement=True)), sv, task_idx == 0, model, model_file, predict_files, score_path, False)
  else:
    sys.stderr.write('Invalid Job Name: %s'%job_name)
    raise

def local_predict(predict_files, vocabulary_size, vocabulary_block_num, factor_num, model_file, score_path):
  model = LocalFmModel(len(predict_files), 0, vocabulary_size, vocabulary_block_num, factor_num, 0, None, PREDICT_BATCH_SIZE, 0, 0)
  _predict(tf.Session(), None, True, model, model_file, predict_files, score_path, True)

