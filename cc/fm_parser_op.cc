#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <ctime>
#include <cstdio>

REGISTER_OP("FmParser")
    .Input("lines: string")
    .Output("labels: float32")
    .Output("ori_ids: int64")
    .Output("feature_ids: int32")
    .Output("feature_vals: float32")
    .Output("feature_poses: int32");

using namespace tensorflow;

class FmParserOp : public OpKernel {
 public:
  explicit FmParserOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor* lines_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("lines", &lines_tensor));
    auto lines = lines_tensor->flat<string>();
    std::vector<float> labels;
    std::map<int64, int32> ori_id_map;
    std::vector<int32> feature_ids;
    std::vector<float> feature_vals;
    std::vector<int32> feature_poses;
    feature_poses.push_back(0);
    for (size_t i = 0; i < lines.size(); ++i) {
      ParseLine(ctx, lines(i), labels, ori_id_map, feature_ids, feature_vals, feature_poses);
    }

    std::vector<int64> ori_ids(ori_id_map.size(), 0);
    for (auto it = ori_id_map.begin(); it != ori_id_map.end(); ++it) {
      ori_ids[it->second] = it->first;
    }

    AllocateTensorForVector<float>(ctx, "labels", labels);
    AllocateTensorForVector<int64>(ctx, "ori_ids", ori_ids);
    AllocateTensorForVector<int32>(ctx, "feature_ids", feature_ids);
    AllocateTensorForVector<float>(ctx, "feature_vals", feature_vals);
    AllocateTensorForVector<int32>(ctx, "feature_poses", feature_poses);
  }

 private:
  void ParseLine(OpKernelContext* ctx, const string& line, std::vector<float>& labels,
      std::map<int64, int32>& ori_id_map, std::vector<int32>& feature_ids, std::vector<float>& feature_vals, std::vector<int32>& feature_poses) {
    const char* p = line.c_str();
    int64 ori_id;
    int32 fid;
    float fv;
    int offset;
    OP_REQUIRES(ctx, sscanf(p, "%f%n", &fv, &offset) == 1,
            errors::InvalidArgument("Label could not be read in example: ", line));
    labels.push_back(fv);
    p += offset;

    while (true) {
      size_t read_size = sscanf(p, " %lld:%f%n", &ori_id, &fv, &offset);
      if (read_size != 2) break;

      auto iter = ori_id_map.find(ori_id);
      if (iter == ori_id_map.end()) {
        fid = ori_id_map.size();
        ori_id_map[ori_id] = fid;
      } else {
        fid = iter->second;
      }
      feature_ids.push_back(fid);
      feature_vals.push_back(fv);
      p += offset;
    }
    feature_poses.push_back(feature_ids.size());

    char c;
    OP_REQUIRES(ctx, sscanf(p, "%*[ ]%c", &c) != 1,
        errors::InvalidArgument("Invalid format for example: ", line));
  }

  template<typename T>
  void AllocateTensorForVector(OpKernelContext* ctx, const string& name, const std::vector<T>& data) {
    Tensor* tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(name, TensorShape({static_cast<int64>(data.size())}), &tensor));
    auto tensor_data = tensor->flat<T>();
    for (size_t i = 0; i < data.size(); ++i) {
      tensor_data(i) = data[i];
    }
  }

};

REGISTER_KERNEL_BUILDER(Name("FmParser").Device(DEVICE_CPU), FmParserOp);
