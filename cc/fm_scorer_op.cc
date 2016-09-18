#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <ctime>

REGISTER_OP("FmScorer")
    .Input("feature_ids: int32")
    .Input("feature_params: float32")
    .Input("feature_vals: float32")
    .Input("feature_poses: int32")
    .Input("factor_lambda: float32")
    .Input("bias_lambda: float32")
    .Output("pred_score: float32")
    .Output("reg_score: float32");

using namespace tensorflow;

class FmScorerOp : public OpKernel {
 public:
  explicit FmScorerOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor* feature_ids_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("feature_ids", &feature_ids_tensor));
    const Tensor* feature_params_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("feature_params", &feature_params_tensor));
    const Tensor* feature_vals_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("feature_vals", &feature_vals_tensor));
    const Tensor* feature_poses_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("feature_poses", &feature_poses_tensor));
    const Tensor* factor_lambda_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("factor_lambda", &factor_lambda_tensor));
    const Tensor* bias_lambda_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("bias_lambda", &bias_lambda_tensor));

    auto feature_ids = feature_ids_tensor->flat<int32>();
    auto feature_params = feature_params_tensor->flat<float>();
    auto feature_vals = feature_vals_tensor->flat<float>();
    auto feature_poses = feature_poses_tensor->flat<int32>();
    auto factor_lambda = factor_lambda_tensor->scalar<float>()();
    auto bias_lambda = bias_lambda_tensor->scalar<float>()();
    
    auto fp = feature_params_tensor->matrix<float>();

    int64 batch_size = feature_poses.size() - 1;
    int64 factor_num = feature_params_tensor->dim_size(1) - 1;

    Tensor* pred_score_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("pred_score", TensorShape({batch_size}), &pred_score_tensor));
    auto pred_score = pred_score_tensor->flat<float>();
    Tensor* reg_score_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("reg_score", TensorShape({}), &reg_score_tensor));
    auto reg_score = reg_score_tensor->scalar<float>();

    std::vector<float> factor_sum(factor_num, 0);
    float rscore = 0;
    for (size_t i = 0; i < batch_size; ++i) {
      float pscore = 0;
      std::fill(factor_sum.begin(), factor_sum.end(), 0);
      for (size_t j = feature_poses(i); j < feature_poses(i + 1); ++j) {
        int32 fid = feature_ids(j);
        float fval = feature_vals(j);
        size_t param_offset = fid * (factor_num + 1);
        float bias = feature_params(param_offset);
        float fsum_2 = 0;
        for (size_t k = 0; k < factor_num; ++k) {
          float t = feature_params(param_offset + k + 1);
          factor_sum[k] += fval * t;
          fsum_2 += t * t;
        }
        pscore += fval * bias;
        pscore -= 0.5 * fval * fval * fsum_2;

        rscore += 0.5 * factor_lambda * fsum_2;
        rscore += 0.5 * bias_lambda * bias * bias;
      }
      for (size_t k = 0; k < factor_num; ++k) {
        pscore += 0.5 * factor_sum[k] * factor_sum[k];
      }
      pred_score(i) = pscore;
    }
    reg_score() = rscore;

  }
};

REGISTER_KERNEL_BUILDER(Name("FmScorer").Device(DEVICE_CPU), FmScorerOp);
