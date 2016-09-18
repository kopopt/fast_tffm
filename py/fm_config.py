class FmConfig:
    def __init__(self):
        self.batch_size = 10000
        self.factor_num = 10
        self.epoch_num = 5
        self.init_value_range = 0.0001
        self.factor_lambda = 1
        self.bias_lambda = 1
        self.vocabulary_size = 500000
        self.thread_num = 2
        self.train_file = ["./data/train.sample.txt"]
        self.test_file = ["./data/test.sample.txt"]