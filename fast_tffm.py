import ConfigParser, sys
from py.fm_ops import fm_ops
from py.fm_train import local_train, dist_train
from py.fm_predict import local_predict, dist_predict
import tensorflow as tf

cmd_instruction = '''Usage:
  1. Local training.
    python fast_tffm.py train <cfg_file>
  2. Distributed training.
    python fast_tffm.py dist_train <cfg_file> <job_name> <task_idx>
  3. Local predicting.
    python fast_tffm.py predict <cfg_file>
  4. Distributed predicting.
    python fast_tffm.py dist_predict <cfg_file> <job_name> <task_idx>
Arguments:
  <cfg_file>: configuartion file path. See sample.cfg for example.
  <job_name>: 'worker' or 'ps'. Launch as a worker or a parameter server
  <task_idx>: Task index.
'''

def check_argument_error(condition):
  if not condition:
    sys.stderr.write('''Invalid arguments\n''')
    sys.stderr.write(cmd_instruction)
    exit()

argc = len(sys.argv)
if argc == 1:
  print cmd_instruction,
  exit()

check_argument_error(argc >= 3)
mode = sys.argv[1]
cfg_file = sys.argv[2]
if mode == 'train' or mode == 'predict':
  check_argument_error(argc == 3)
elif mode == 'dist_train' or mode == 'dist_predict':
  check_argument_error(argc == 5)
  job_name = sys.argv[3]
  task_idx = int(sys.argv[4])
else:
  check_argument_error(False)

GENERAL_SECTION = 'General'
TRAIN_SECTION = 'Train'
PREDICT_SECTION = 'Predict'
CLUSTER_SPEC_SECTION = 'ClusterSpec'
STR_DELIMITER = ','

config = ConfigParser.ConfigParser()
config.read(cfg_file)
print 'Config: '
def read_config(section, option, not_null = True):
  if not config.has_option(section, option):
    if not_null:
      raise ValueError("%s is undefined."%option)
    else:
      return None
  else:
    value = config.get(section, option)
    print '  {0} = {1}'.format(option, value)
    return value
def read_strs_config(section, option, not_null = True):
  val = read_config(section, option, not_null)
  if val != None:
    return [s.strip() for s in val.split(STR_DELIMITER)]
  return None

factor_num = int(read_config(GENERAL_SECTION, 'factor_num'))
vocabulary_size = int(read_config(GENERAL_SECTION, 'vocabulary_size'))
vocabulary_block_num = int(read_config(GENERAL_SECTION, 'vocabulary_block_num'))
model_file = read_config(GENERAL_SECTION, 'model_file')
hash_feature_id = read_config(GENERAL_SECTION, 'hash_feature_id').strip().lower() == 'true'

if mode == 'dist_train' or mode == 'dist_predict':
  ps_hosts = read_strs_config(CLUSTER_SPEC_SECTION, 'ps_hosts')
  worker_hosts = read_strs_config(CLUSTER_SPEC_SECTION, 'worker_hosts')

if mode == 'train' or mode == 'dist_train':
  batch_size = int(read_config(TRAIN_SECTION, 'batch_size'))
  init_value_range = float(read_config(TRAIN_SECTION, 'init_value_range'))
  factor_lambda = float(read_config(TRAIN_SECTION, 'factor_lambda'))
  bias_lambda = float(read_config(TRAIN_SECTION, 'bias_lambda'))
  thread_num = int(read_config(TRAIN_SECTION, 'thread_num'))
  epoch_num = int(read_config(TRAIN_SECTION, 'epoch_num'))
  train_files = read_strs_config(TRAIN_SECTION, 'train_files')
  weight_files = read_strs_config(TRAIN_SECTION, 'weight_files', False)
  if weight_files != None and len(train_files) != len(weight_files):
    raise ValueError('The numbers of train files and weight files do not match.')
  validation_files = read_strs_config(TRAIN_SECTION, 'validation_files', False)
  learning_rate = float(read_config(TRAIN_SECTION, 'learning_rate'))
  adagrad_init_accumulator = float(read_config(TRAIN_SECTION, 'adagrad.initial_accumulator'))
  loss_type = read_config(TRAIN_SECTION, 'loss_type').strip().lower()
  if not loss_type in ['logistic', 'mse']:
    raise ValueError('Unsupported loss type: %s'%loss_type)

  optimizer = tf.train.AdagradOptimizer(learning_rate, adagrad_init_accumulator)

  if mode == 'train':
    local_train(train_files, weight_files, validation_files, epoch_num, vocabulary_size, vocabulary_block_num, hash_feature_id, factor_num, init_value_range, loss_type, optimizer, batch_size, factor_lambda, bias_lambda, thread_num, model_file)
  else:
    dist_train(ps_hosts, worker_hosts, job_name, task_idx, train_files, weight_files, validation_files, epoch_num, vocabulary_size, vocabulary_block_num, hash_feature_id, factor_num, init_value_range, loss_type, optimizer, batch_size, factor_lambda, bias_lambda, thread_num, model_file)
elif mode == 'predict' or mode == 'dist_predict':
  predict_files = read_config(PREDICT_SECTION, 'predict_files').split(',')
  score_path = read_config(PREDICT_SECTION, 'score_path')

  if mode == 'predict':
    local_predict(predict_files, vocabulary_size, vocabulary_block_num, hash_feature_id, factor_num, model_file, score_path)
  else:
    dist_predict(ps_hosts, worker_hosts, job_name, task_idx, predict_files, vocabulary_size, vocabulary_block_num, hash_feature_id, factor_num, model_file, score_path)

