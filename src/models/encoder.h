#pragma once

#include <torch/nn.h>
#include <string>
#include "rnn.h"
#include "cli_options.h"

using namespace torch::nn;

// Marian/Nematus-style BiDeep encoder
class BiDeepEncoderImpl : public Module {
 public:
  explicit BiDeepEncoderImpl(const ModelOptions &model_options);
  Tensor forward(const MaskedData &input);

 private:
  Embedding emb_{nullptr};
  StackedRNN rnn_fw_{nullptr};
  StackedRNN rnn_bw_{nullptr};
  StackedRNN rnn_uni_{nullptr};
  EncoderType type_;
  bool skip_;
};
TORCH_MODULE(BiDeepEncoder);
