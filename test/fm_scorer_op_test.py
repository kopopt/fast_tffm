import sys
sys.path.append(sys.path[0] + '/../py/')
import tensorflow as tf
from fm_ops import fm_ops
import random
from tensorflow.python.platform import googletest

class FmScorerOpTest(tf.test.TestCase):
  def testBasic(self):
    batch_size = 2
    word_num_per_example = 6
    vocabulary_size = 10
    factor_num = 8
    feature_ids = [[1,2,3,2,4,9],[3,4,7,0,0,0]]
    feature_vals = [[9.0,2.0,3.6,4.0,-2.2,-10.7], [4.0,-3.4,2.0,0,0,0]]
    labels = [1.0, -1.0]
    factor_lambda = 1.2
    bias_lambda = 0.4
    params = [[0] * (factor_num + 1)]
    for i in range(vocabulary_size):
      params.append([random.uniform(0.01, 0.02) for j in range(factor_num + 1)])

    flat_feature_ids = []
    flat_feature_vals = []
    flat_feature_poses = [0]
    for i in range(len(feature_ids)):
      for j in range(len(feature_ids[i])):
        fid = feature_ids[i][j]
        fval = feature_vals[i][j]
        if fid == 0:
          continue
        flat_feature_ids.append(fid)
        flat_feature_vals.append(fval)
      flat_feature_poses.append(len(flat_feature_ids))

    params_var = tf.Variable(params)
    factor_map = tf.slice(params_var, [0, 1], [vocabulary_size, factor_num])
    bias_map = tf.reshape(tf.slice(params_var, [0, 0], [vocabulary_size, 1]), [vocabulary_size])

    factors = tf.gather(factor_map, feature_ids)
    biases = tf.gather(bias_map, feature_ids)
    fvals = tf.reshape(feature_vals, [batch_size, word_num_per_example, 1])
    factor_sum = tf.reduce_sum(factors * fvals,1)
    tf_pred = 0.5 * tf.reduce_sum(factor_sum * factor_sum, [1]) - 0.5 * tf.reduce_sum(fvals * fvals * factors * factors, [1,2]) + tf.reduce_sum(biases * feature_vals,[1]);
    tf_reg = 0.5 * (factor_lambda * tf.reduce_sum(factors * factors) + bias_lambda * tf.reduce_sum(biases * biases))
    tf_cost = tf.reduce_sum(tf.squared_difference(labels, tf_pred)) + tf_reg
    tf_grad = tf.gradients(tf_cost, params_var)

    my_pred, my_reg = fm_ops.fm_scorer(flat_feature_ids, params_var, flat_feature_vals, flat_feature_poses, factor_lambda, bias_lambda)
    my_cost = tf.reduce_sum(tf.squared_difference(labels, my_pred)) + my_reg
    my_grad = tf.gradients(my_cost, params_var)

    with self.test_session():
      tf.initialize_all_variables().run()
      self.assertAllClose(tf_pred.eval(), my_pred.eval())
      self.assertAllClose(tf_reg.eval(), my_reg.eval())
      self.assertAllClose(tf_grad[0].eval(), my_grad[0].eval())

if __name__ == "__main__":
  googletest.main()
