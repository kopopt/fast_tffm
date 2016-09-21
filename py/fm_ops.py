import tensorflow as tf
import os
from tensorflow.python.framework import ops

fm_ops = tf.load_op_library(os.path.dirname(os.path.realpath(__file__)) + '/../lib/libfast_tffm.so')

@ops.RegisterGradient("FmScorer")
def _fm_scorer_grad(op, pred_grad, reg_grad):
    feature_ids = op.inputs[0]
    feature_params = op.inputs[1]
    feature_vals = op.inputs[2]
    feature_poses = op.inputs[3]
    factor_lambda = op.inputs[4]
    bias_lambda = op.inputs[5]
    with ops.control_dependencies([pred_grad.op, reg_grad.op]):
        return None, fm_ops.fm_grad(feature_ids, feature_params, feature_vals, feature_poses, factor_lambda, bias_lambda, pred_grad, reg_grad), None, None, None, None
