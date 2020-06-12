#pragma once

#include <torch/nn.h>
#include <string>
#include "types.h"
#include "attention.h"

using namespace torch::nn;
using torch::Tensor;

// Deep Transition GRU Cell
// v_{k,1} = GRU_{k,1}(in_k, state_k)
// v_{k,t} = GRU_{k,t}(0, v_{k, t−1}) for 1 < k ≤ L_s
//
// DTGRU_k(in_k, state_k) = v_{k,L_s}
class DTGRUCellImpl : public Module {
 public:
  explicit DTGRUCellImpl(size_t input_dim,
                         size_t hidden_dim,
                         size_t transition_depth=1);
  Tensor step(const Tensor &input, const Tensor &state);
  Tensor forward(const Tensor &input);

 protected:
  ModuleList dt_cell_;
  size_t rnn_dim_;
};
TORCH_MODULE(DTGRUCell);

class CondDTGRUCellImpl : public Module {
 public:
  explicit CondDTGRUCellImpl(size_t input_dim,
                             size_t hidden_dim,
                             size_t transition_depth=1);
  std::tuple<Tensor, Tensor> step(const Tensor &input, const Tensor &state);
  void set_attention_context(const Tensor &encoder_states, const Tensor &src_mask);

 private:
  ModuleList dt_cell_;
  size_t rnn_dim_;
  GlobalAttention att_{nullptr};
};
TORCH_MODULE(CondDTGRUCell);

// Stack of DTGRUCells
class StackedRNNImpl : public Module {
 public:
  explicit StackedRNNImpl(size_t input_dim,
                          size_t hidden_dim,
                          size_t depth=1,
                          size_t transition_depth=1,
                          StackedRNNDir dir=StackedRNNDir::forward,
                          bool skip=false);

  Tensor forward(const Tensor &input, const Tensor &lengths);
  void insert_conditional_cell(size_t input_dim,
                               size_t hidden_dim,
                               size_t cell_depth);
  int64_t num_layers() { return stack_->size(); }

  // For decoder
  std::tuple<Tensor, Tensor> step(const Tensor &input, const Tensor &state);
  void set_attention_context(const Tensor &encoder_states, const Tensor &src_mask);

 private:
  ModuleList stack_;
  StackedRNNDir dir_;
  size_t rnn_dim_;
  bool skip_;
};
TORCH_MODULE(StackedRNN);
