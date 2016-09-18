from py.fm_ops import fm_ops
from py.fm_model import FmModel
import tensorflow as tf
import ConfigParser, sys

cmd_instruction = '''usage: python fast_tffm.py <mode> <cfg_file>
   <mode> -- train" or predict
   <cfg_file> -- configuartion file path. See sample.cfg for example.
'''

if len(sys.argv) != 3:
    sys.stderr.write('''Invalid number of arguments\n''')
    sys.stderr.write(cmd_instruction)
    exit()

mode = sys.argv[1]
cfg_file = sys.argv[2]

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

GENERAL_SECTION = 'General'
TRAIN_SECTION = 'Train'
PREDICT_SECTION = 'Predict'
STRING_DELIMITER = ','

model_file = read_config(GENERAL_SECTION, 'model_file')
factor_num = int(read_config(GENERAL_SECTION, 'factor_num'))
vocabulary_size = int(read_config(GENERAL_SECTION, 'vocabulary_size'))

if mode == 'train':
    batch_size = int(read_config(TRAIN_SECTION, 'batch_size'))
    init_value_range = float(read_config(TRAIN_SECTION, 'init_value_range'))
    factor_lambda = float(read_config(TRAIN_SECTION, 'factor_lambda'))
    bias_lambda = float(read_config(TRAIN_SECTION, 'bias_lambda'))
    thread_num = int(read_config(TRAIN_SECTION, 'thread_num'))
    epoch_num = int(read_config(TRAIN_SECTION, 'epoch_num'))
    train_files_str = read_config(TRAIN_SECTION, 'train_files')
    train_files = [s.strip() for s in train_files_str.split(STRING_DELIMITER)]
    validation_files_str = read_config(TRAIN_SECTION, 'validation_files', False)
    if validation_files_str == None:
        validation_files = None
    else:
        validation_files = [s.strip() for s in validation_files_str.split(STRING_DELIMITER)]
    learning_rate = float(read_config(TRAIN_SECTION, 'learning_rate'))
    adagrad_init_accumulator = float(read_config(TRAIN_SECTION, 'adagrad.initial_accumulator'))

    fm_model = FmModel(vocabulary_size, factor_num, init_value_range)
    optimizer = tf.train.AdagradOptimizer(learning_rate, adagrad_init_accumulator)
    fm_model.train(train_files, validation_files, optimizer, epoch_num, thread_num, batch_size, factor_lambda, bias_lambda)
    fm_model.save(model_file)
    fm_model.close()
elif mode == 'predict':
    predict_file = read_config(PREDICT_SECTION, 'predict_file')
    score_file = read_config(PREDICT_SECTION, 'score_file')
    fm_model = FmModel(vocabulary_size, factor_num, 1)
    fm_model.load(model_file)
    fm_model.predict(predict_file, score_file)
    fm_model.close()
else:
    raise ValueError("Invalid mode '%s'. Only 'train' or 'predict' are expected."%mode)