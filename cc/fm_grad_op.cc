/*
 * fm_grad_op.cc
 *
 *  Created on: Sep 5, 2016
 *      Author: mianwei
 */

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

REGISTER_OP("FmGrad")
    .Input("feature_ids: int32")
    .Input("feature_params: float32")
    .Input("feature_vals: float32")
    .Input("feature_poses: int32")
    .Input("factor_lambda: float32")
    .Input("bias_lambda: float32")
    .Input("pred_grad: float32")
    .Input("reg_grad: float32")
    .Output("params_grad: float32");

using namespace tensorflow;

class FmGradOp : public OpKernel {
 public:
  explicit FmGradOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

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
    const Tensor* pred_grad_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("pred_grad", &pred_grad_tensor));
    const Tensor* reg_grad_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("reg_grad", &reg_grad_tensor));

    auto feature_ids = feature_ids_tensor->flat<int32>();
    auto feature_params = feature_params_tensor->flat<float>();
    auto feature_vals = feature_vals_tensor->flat<float>();
    auto feature_poses = feature_poses_tensor->flat<int32>();
    auto factor_lambda = factor_lambda_tensor->scalar<float>()();
    auto bias_lambda = bias_lambda_tensor->scalar<float>()();
    auto pred_grad = pred_grad_tensor->flat<float>();
    auto reg_grad = reg_grad_tensor->scalar<float>()();

    int64 batch_size = feature_poses.size() - 1;
    int64 factor_num = feature_params_tensor->dim_size(1) - 1;

    Tensor* param_grad_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("params_grad", feature_params_tensor->shape(), &param_grad_tensor));
    auto param_grad = param_grad_tensor->flat<float>();
    param_grad.setZero();

    std::vector<float> factor_sum(factor_num, 0);

    for (size_t i = 0; i < batch_size; ++i) {
      std::fill(factor_sum.begin(), factor_sum.end(), 0);
      auto pgrad = pred_grad(i);
      for (size_t j = feature_poses(i); j < feature_poses(i + 1); ++j) {
        auto fid = feature_ids(j);
        auto fval = feature_vals(j);
        size_t param_offset = fid * (factor_num + 1);
        for (size_t k = 0; k < factor_num; ++k) {
          float t = feature_params(param_offset + k + 1);
          factor_sum[k] += fval * t;
          param_grad(param_offset + k + 1) += reg_grad * factor_lambda * t - pgrad * fval * fval * t;
        }
        param_grad(param_offset) += pgrad * fval + reg_grad * bias_lambda * feature_params(param_offset);
      }
      for (size_t j = feature_poses(i); j < feature_poses(i + 1); ++j) {
        auto fid = feature_ids(j);
        auto fval = feature_vals(j);
        size_t param_offset = fid * (factor_num + 1);
        for (size_t k = 0; k < factor_num; ++k) {
          param_grad(param_offset + k + 1) += pgrad * fval * factor_sum[k];
        }
      }
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("FmGrad").Device(DEVICE_CPU), FmGradOp);
