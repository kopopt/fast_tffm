import sys, threading, time
file_path = sys.path[0]
sys.path.append(file_path + '/../py')
import tensorflow as tf
from tensorflow.python.platform import googletest
from fm_ops import fm_ops

class FmParserOpTest(tf.test.TestCase):
  def testNoHashWithWeight(self):
    parser_op = fm_ops.fm_parser(0, file_path + '/sample_data_0', file_path + '/weight', 3, 1000, False)
    with self.test_session() as sess:
      labels, weights, ori_ids, feature_ids, feature_vals, feature_poses = sess.run(parser_op)
      self.assertAllClose([1, 0.8, -10], labels)
      self.assertAllClose([2, 3, 1], weights)
      self.assertAllEqual([234, 2, 0, 10], ori_ids)
      self.assertAllEqual([0, 1, 2, 2, 1, 3], feature_ids)
      self.assertAllClose([10, 1, 2, 1, 1, 20.3], feature_vals)
      self.assertAllEqual([0, 3, 6, 6], feature_poses)
      labels, weights, ori_ids, feature_ids, feature_vals, feature_poses = sess.run(parser_op)
      self.assertAllClose([2, 12], labels)
      self.assertAllClose([0.5, 0.2], weights)
      self.assertAllEqual([10, 0], ori_ids)
      self.assertAllEqual([0, 1], feature_ids)
      self.assertAllClose([21, 1], feature_vals)
      self.assertAllEqual([0, 1, 2], feature_poses)
      labels, weights, ori_ids, feature_ids, feature_vals, feature_poses = sess.run(parser_op)
      self.assertAllEqual([len(labels), len(weights)], [0, 0])

  def testHashWithNoWeight(self):
    parser_op = fm_ops.fm_parser(0, file_path + '/sample_data_1', '', 10, 1000, True)
    with self.test_session() as sess:
      labels, weights, ori_ids, feature_ids, feature_vals, feature_poses = sess.run(parser_op)
      self.assertAllClose([1, 0.2, -10], labels)
      self.assertAllClose([1, 1, 1], weights)
      self.assertAllEqual([819, 280, 545, 273, 542], ori_ids)
      self.assertAllEqual([0, 1, 2, 1, 2, 3, 4], feature_ids)
      self.assertAllClose([10, 1, 2, 1, 1, 20.3, 1], feature_vals)
      self.assertAllEqual([0, 3, 7, 7], feature_poses)
  
  def testError(self):
    with self.test_session() as sess:
      with self.assertRaisesOpError("The line number in data file and weight file do not match."):
        fm_ops.fm_parser(0, file_path + '/sample_data_1', file_path + '/weight', 10, 1000, True).labels.eval()

      with self.assertRaisesOpError("Invalid feature id feq321. Set hash_feature_id = True?"):
        fm_ops.fm_parser(0, file_path + '/sample_data_1', '', 10, 1000, False).labels.eval()
      
      with self.assertRaisesOpError("Label could not be read in example: aa 12 123"):
        fm_ops.fm_parser(0, file_path + '/wrong_data', '', 10, 1000, True).labels.eval()

      with self.assertRaisesOpError("Invalid weight: aa"):
        fm_ops.fm_parser(0, file_path + '/sample_data_0', file_path + '/wrong_weight', 3, 1000, False).labels.eval()

if __name__ == "__main__":
  googletest.main()
