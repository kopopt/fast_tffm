import sys, threading, time
file_path = sys.path[0]
sys.path.append(file_path + '/../py')
import tensorflow as tf
from tensorflow.python.platform import googletest
from fm_ops import fm_ops

class FmParserOpTest(tf.test.TestCase):
  def testBasic(self):
    labels_op, ori_ids_op, feature_ids_op, feature_vals_op, feature_poses_op = fm_ops.fm_parser(0, file_path + '/correct_data', 3)
    with self.test_session() as sess:
      labels, ori_ids, feature_ids, feature_vals, feature_poses = sess.run([labels_op, ori_ids_op, feature_ids_op, feature_vals_op, feature_poses_op])
      self.assertAllClose([1, 0.8, -10], labels)
      self.assertAllEqual([1234, 2, 0, 10], ori_ids)
      self.assertAllEqual([0, 1, 2, 2, 1, 3], feature_ids)
      self.assertAllClose([10, 1, 2, 1, 1, 20.3], feature_vals)
      self.assertAllEqual([0, 3, 6, 6], feature_poses)
      labels, ori_ids, feature_ids, feature_vals, feature_poses = sess.run([labels_op, ori_ids_op, feature_ids_op, feature_vals_op, feature_poses_op])
      self.assertAllClose([2, 12], labels)
      self.assertAllEqual([10], ori_ids)
      self.assertAllEqual([0], feature_ids)
      self.assertAllClose([21], feature_vals)
      self.assertAllEqual([0, 1, 1], feature_poses)
  
  def testBatch(self):
    fid = tf.placeholder(dtype = tf.int32)
    fname = tf.placeholder(dtype = tf.string)
    batch = fm_ops.fm_parser(fid, fname, 97)
    with self.test_session() as sess:
      for i in range(5):
        self.nn = 0
        self.lock = threading.Lock()
        def run(self):
          while True:
            d = sess.run(batch, feed_dict = {fid: i, fname: file_path + '/../data/train_%d'%i})
            if len(d[0]) == 0: break
            self.lock.acquire()
            self.nn += len(d[0])
            self.lock.release()
        threads = [threading.Thread(target = run, args = (self,)) for j in range(10)]
        for th in threads: th.start()
        for th in threads: th.join()
        self.assertAllEqual(self.nn, 10000)

  def testError(self):
    with self.test_session():
      with self.assertRaisesOpError("Invalid format for example:  12 123"):
        fm_ops.fm_parser(0, file_path + '/wrong_data_0', 10).labels.eval()
      with self.assertRaisesOpError("Invalid format for example:  12 123:312     0  "):
        fm_ops.fm_parser(0, file_path + '/wrong_data_1', 10).labels.eval()
  
if __name__ == "__main__":
  googletest.main()
