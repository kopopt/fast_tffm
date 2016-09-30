#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/hash/hash.h"
#include <ctime>
#include <cstdio>
#include <fstream>

REGISTER_OP("FmParser")
    .Input("file_id: int32")
    .Input("data_file: string")
    .Input("weight_file: string")
    .Output("labels: float32")
    .Output("weights: float32")
    .Output("ori_ids: int64")
    .Output("feature_ids: int32")
    .Output("feature_vals: float32")
    .Output("feature_poses: int32")
    .Attr("batch_size: int")
    .Attr("vocab_size: int")
    .Attr("hash_feature_id: bool = false");

#define MAX_FEATURE_ID_LENGTH 100

using namespace tensorflow;

class FmParserOp : public OpKernel {
 public:

  explicit FmParserOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("batch_size", &batch_size_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("vocab_size", &vocab_size_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("hash_feature_id", &hash_feature_id_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor* fid_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("file_id", &fid_tensor));
    auto fid = fid_tensor->scalar<int32>()();
    OP_REQUIRES(ctx, fid >= 0, errors::InvalidArgument("file_id should be greater than 0."))
    const Tensor* data_file_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("data_file", &data_file_tensor));
    auto data_file = data_file_tensor->scalar<string>()();
    const Tensor* weight_file_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("weight_file", &weight_file_tensor));
    auto weight_file = weight_file_tensor->scalar<string>()();

    bool has_weight_file = (weight_file != "");
    std::vector<string> data_lines;
    std::vector<float> weights;
    {
      mutex_lock l(mu_);
      OP_REQUIRES(ctx, fid >= file_id_, errors::InvalidArgument("file_id is less than last file_id", file_id_))
      if (fid > file_id_) {
        if (data_file_stream_ != NULL) {
          data_file_stream_->close();
          delete data_file_stream_;
          if (has_weight_file) {
            weight_file_stream_->close();
            delete weight_file_stream_;
          }
        }
        data_file_stream_ = new std::ifstream(data_file);
        OP_REQUIRES(ctx, data_file_stream_->is_open(), errors::InvalidArgument("Fails to open data file: ", data_file))
        current_data_file_ = data_file;
        if (has_weight_file) {
          weight_file_stream_ = new std::ifstream(weight_file);
          OP_REQUIRES(ctx, weight_file_stream_->is_open(), errors::InvalidArgument("Fails to open weight file: ", weight_file))
          current_weight_file_ = weight_file;
        }
        file_id_ = fid;
      } else {
        OP_REQUIRES(ctx, current_data_file_ == data_file, errors::InvalidArgument("Data file is different with the same file id."))
        if (has_weight_file) {
            OP_REQUIRES(ctx, current_weight_file_ == weight_file, errors::InvalidArgument("Weight file is different with the same file id."))
        }
      }
      string data_line, weight_line;
      char *err;
      int k = 0;
      while (k < batch_size_) {
        std::getline(*data_file_stream_, data_line);
        if (has_weight_file) {
          std::getline(*weight_file_stream_, weight_line);
          OP_REQUIRES(ctx, data_file_stream_->eof() == weight_file_stream_->eof(), errors::InvalidArgument("The line number in data file and weight file do not match."))
        }
        if (data_file_stream_->eof()) {
          break;
        }
        data_lines.push_back(data_line);
        if (has_weight_file) {
          weights.push_back(strtof(weight_line.c_str(), &err));
          OP_REQUIRES(ctx, *err == 0 || isspace((unsigned char)*err), errors::InvalidArgument("Invalid weight: ", weight_line))
        } else {
          weights.push_back(1.0f);
        }
        k += 1;
      }
    }

    std::vector<float> labels;
    std::map<int64, int32> ori_id_map;
    std::vector<int32> feature_ids;
    std::vector<float> feature_vals;
    std::vector<int32> feature_poses;
    feature_poses.push_back(0);
    for (size_t i = 0; i < data_lines.size(); ++i) {
      ParseLine(ctx, data_lines[i], hash_feature_id_, vocab_size_, labels, ori_id_map, feature_ids, feature_vals, feature_poses);
    }

    std::vector<int64> ori_ids(ori_id_map.size(), 0);
    for (auto it = ori_id_map.begin(); it != ori_id_map.end(); ++it) {
      ori_ids[it->second] = it->first;
    }

    AllocateTensorForVector<float>(ctx, "labels", labels);
    AllocateTensorForVector<int64>(ctx, "ori_ids", ori_ids);
    AllocateTensorForVector<float>(ctx, "weights", weights);
    AllocateTensorForVector<int32>(ctx, "feature_ids", feature_ids);
    AllocateTensorForVector<float>(ctx, "feature_vals", feature_vals);
    AllocateTensorForVector<int32>(ctx, "feature_poses", feature_poses);
  }

 private:
  int32 batch_size_;
  int64 vocab_size_;
  bool hash_feature_id_;

  mutex mu_;
  std::ifstream* data_file_stream_ = NULL;
  std::ifstream* weight_file_stream_ = NULL;
  std::string current_data_file_ = "";
  std::string current_weight_file_ = "";
  int file_id_ = -1;

  void ParseLine(OpKernelContext* ctx, const string& line, bool hash_feature_id, int64 vocab_size, std::vector<float>& labels,
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

    size_t read_size;
    char ori_id_str[MAX_FEATURE_ID_LENGTH];
    char* err;
    while (true) {
      if (sscanf(p, " %[^: ]%n", ori_id_str, &offset) != 1) break;
      if (hash_feature_id) {
        ori_id = Hash64(ori_id_str, strlen(ori_id_str));
      } else {
        ori_id = strtol(ori_id_str, &err, 10);
        OP_REQUIRES(ctx, *err == 0, errors::InvalidArgument("Invalid feature id ", ori_id_str, ". Set hash_feature_id = True?"))
      }
      ori_id = labs(ori_id % vocab_size);
      p += offset;
      if (*p == ':') {
        OP_REQUIRES(ctx, sscanf(p, ":%f%n", &fv, &offset) == 1, errors::InvalidArgument("Invalid feature value: ", ori_id_str))
        p += offset;
      } else {
        fv = 1;
      }
      auto iter = ori_id_map.find(ori_id);
      if (iter == ori_id_map.end()) {
        fid = ori_id_map.size();
        ori_id_map[ori_id] = fid;
      } else {
        fid = iter->second;
      }
      feature_ids.push_back(fid);
      feature_vals.push_back(fv);
    }
    feature_poses.push_back(feature_ids.size());
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
