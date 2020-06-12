#pragma once

#include <torch/nn.h>
#include "encoder.h"
#include "decoder.h"

using namespace torch::nn;

// Forward declaration
struct ModelOptions;

template <typename DecoderType=BiDeepDecoder>
class EncoderDecoderImpl : public Module {
 public:
  explicit EncoderDecoderImpl(ModelOptions &model_options) {
    encoder_ = register_module("encoder", BiDeepEncoder(model_options));
    decoder_ = register_module("decoder", DecoderType(model_options));
  }

  torch::Tensor forward(MaskedData &src_batch, MaskedData &trg_batch) {
    auto encoder_states = encoder_->forward(src_batch);
    return decoder_->forward(encoder_states, src_batch.lengths, src_batch.mask, trg_batch.data);
  }

  void print_params() {
    for (const auto& pair : named_parameters()) {
      std::cout << pair.key() << ": " << pair.value().sizes() << std::endl;
    }
  }

 private:
  BiDeepEncoder encoder_{nullptr};
  DecoderType decoder_{nullptr};
};

template <typename DecoderType=BiDeepDecoder>
TORCH_MODULE_IMPL(EncoderDecoder, EncoderDecoderImpl<DecoderType>);
