#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <ctime>
#include <cstdio>
#include <fstream>

REGISTER_OP("FmParser")
    .Input("file_id: int32")
    .Input("file_name: string")
    .Input("batch_size: int32")
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
    const Tensor* fid_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("file_id", &fid_tensor));
    auto fid = fid_tensor->scalar<int32>()();
    OP_REQUIRES(ctx, fid >= 0, errors::InvalidArgument("file_id should be greater than 0."))
    const Tensor* fname_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("file_name", &fname_tensor));
    auto fname = fname_tensor->scalar<string>()();
    const Tensor* batch_size_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("batch_size", &batch_size_tensor));
    auto batch_size = batch_size_tensor->scalar<int32>()();

    std::vector<string> lines;
    {
      mutex_lock l(mu_);
      OP_REQUIRES(ctx, fid >= file_id_, errors::InvalidArgument("file_id is less than last file_id", file_id_))
      if (fid > file_id_) {
        if (file_stream_ != NULL) {
          file_stream_->close();
          delete file_stream_;
        }
        file_stream_ = new std::ifstream(fname);
        OP_REQUIRES(ctx, file_stream_->is_open(), errors::InvalidArgument("Fails to open file ", fname))
        file_name_ = fname;
        file_id_ = fid;
      } else {
        OP_REQUIRES(ctx, file_name_ == fname, errors::InvalidArgument("With the same file id, file name is different."))
      }
      string line;
      int k = 0;
      while (k < batch_size && std::getline(*file_stream_, line)) {
        lines.push_back(line);
        k += 1;
      }
    }

    std::vector<float> labels;
    std::map<int64, int32> ori_id_map;
    std::vector<int32> feature_ids;
    std::vector<float> feature_vals;
    std::vector<int32> feature_poses;
    feature_poses.push_back(0);
    for (size_t i = 0; i < lines.size(); ++i) {
      ParseLine(ctx, lines[i], labels, ori_id_map, feature_ids, feature_vals, feature_poses);
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
  mutex mu_;
  std::ifstream* file_stream_ = NULL;
  std::string file_name_ = "";
  int file_id_ = -1;


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
