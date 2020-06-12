#pragma once

#include <torch/nn.h>

using namespace torch::nn;
using torch::Tensor;

class GlobalAttentionImpl : public Module {
 public:
  GlobalAttentionImpl(const size_t enc_state_dim, const size_t dec_state_dim) {
    att_context_ = register_module("context_map",
                           Linear(LinearOptions(enc_state_dim,
                                                enc_state_dim)
                                                .bias(false)));
    att_dec_state_ = register_module("dec_state_map",
                                     Linear(LinearOptions(dec_state_dim,
                                                          enc_state_dim)
                                                          .bias(false)));
    att_bias_ = register_parameter("bias",
                                   torch::zeros(enc_state_dim));
    att_score_ = register_module("score",
                                 Linear(LinearOptions(enc_state_dim, 1)));
    
  }

  std::pair<Tensor, Tensor> forward(const Tensor &dec_state) {
    Tensor weights = functional::softmax(
                        att_score_->forward(
                          torch::tanh(att_dec_state_->forward(dec_state) + mapped_context_ + att_bias_))
                        + batch_mask_,
                        /*dim=*/0);
    Tensor att_context = torch::sum(encoder_states_ * weights, /*dim=*/0);
    return std::pair<Tensor, Tensor>(att_context, weights);
  }

  void set_context(const Tensor &encoder_states, const Tensor &src_mask) {
    // Set encoder context once per batch instead of repeating at every time-step
    encoder_states_ = encoder_states;
    batch_mask_ = src_mask.unsqueeze(-1).to(torch::kFloat).log();
    mapped_context_ = att_context_->forward(encoder_states);
  }

 private:
  Linear att_context_{nullptr};
  Linear att_dec_state_{nullptr};
  Linear att_score_{nullptr};
  Tensor att_bias_;

  Tensor encoder_states_;
  Tensor batch_mask_;
  Tensor mapped_context_;
};
TORCH_MODULE(GlobalAttention);
