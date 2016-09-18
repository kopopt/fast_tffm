import sys
sys.path.append(sys.path[0] + '/../py')
import tensorflow as tf
from tensorflow.python.platform import googletest
from fm_ops import fm_ops

class FmParserOpTest(tf.test.TestCase):
  def testBasic(self):
    with self.test_session():
      labels, ori_ids, feature_ids, feature_vals, feature_poses = fm_ops.fm_parser([" 1  1234:10 2:1  0:2 ", "0.8 0:1 2:1 10:20.3", "-10"])
      self.assertAllClose([1, 0.8, -10], labels.eval())
      self.assertAllEqual([1234, 2, 0, 10], ori_ids.eval())
      self.assertAllEqual([0, 1, 2, 2, 1, 3], feature_ids.eval())
      self.assertAllClose([10, 1, 2, 1, 1, 20.3], feature_vals.eval())
      self.assertAllEqual([0, 3, 6, 6], feature_poses.eval())

  def testError(self):
    with self.test_session():
      with self.assertRaisesOpError("Invalid format for example:  12 123"):
        fm_ops.fm_parser([" 12 123"]).labels.eval()
      with self.assertRaisesOpError("Invalid format for example:  12 123:312     0  "):
        fm_ops.fm_parser([" 12 123:312     0  "]).labels.eval()
      
if __name__ == "__main__":
  googletest.main()
