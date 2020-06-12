#include "rnn.h"
#include "rnn_utils.h"

using torch::indexing::Ellipsis;

DTGRUCellImpl::DTGRUCellImpl(size_t input_dim,
                             size_t hidden_dim,
                             size_t transition_depth)
    : rnn_dim_(hidden_dim) {
  for(size_t i = 1; i <= transition_depth; ++i) {
    dt_cell_->push_back(
      register_module("cell" + std::to_string(i),
                      GRUCell(input_dim, rnn_dim_)));
    input_dim = rnn_dim_; // For layers > 1
  }
}

CondDTGRUCellImpl::CondDTGRUCellImpl(size_t input_dim,
                                     size_t hidden_dim,
                                     size_t transition_depth)
    : rnn_dim_(hidden_dim) {
  for(size_t i = 1; i <= transition_depth; ++i) {
    dt_cell_->push_back(
      register_module("cell" + std::to_string(i),
                      GRUCell(input_dim, rnn_dim_)));
    if(i == 1) {
      input_dim = 2 * rnn_dim_; // For second layer
    }
    else {
      input_dim = rnn_dim_; // For layers > 1
    }
  }
  att_ = register_module("attention",
                         GlobalAttention(2 * rnn_dim_,
                                         rnn_dim_));
}

// One time-step of deep transition cell
// Input input: {batch_size, input_dim}
// Input state: {batch_size, rnn_dim}
// Returns: {batch_size, rnn_dim}
Tensor DTGRUCellImpl::step(const Tensor &input, const Tensor &state) {
  Tensor curr_state = state;
  Tensor curr_input = input;
  for(size_t l = 0; l < dt_cell_->size(); ++l) {
    curr_state = dt_cell_[l]->as<GRUCell>()->forward(curr_input, curr_state);
    curr_input = torch::zeros_like(curr_state, torch::kFloat); // Zero input for higher layers
  }
  return curr_state;
}

// One time-step of deep transition cell with attention
// Input input: {batch_size, input_dim}
// Input state: {batch_size, rnn_dim}
// Returns: {batch_size, rnn_dim}
std::tuple<Tensor, Tensor> CondDTGRUCellImpl::step(const Tensor &input, const Tensor &state) {
  Tensor curr_state = state;
  Tensor curr_input = input;
  Tensor att_context;
  for(size_t l = 0; l < dt_cell_->size(); ++l) {
    curr_state = dt_cell_[l]->as<GRUCell>()->forward(curr_input, curr_state);
    if(l == 0) {
      curr_input = std::get<0>(att_->forward(curr_state));
      att_context = curr_input.clone();
    }
    else {
      curr_input = torch::zeros_like(curr_state, torch::kFloat); // Zero input for higher layers
    }
  }
  return std::tuple<Tensor, Tensor>(curr_state, att_context);
}

// Transduce entire sequence with deep transition cell
// Input: {seq_len, batch_size, input_dim}
// Returns: {seq_len, batch_size, rnn_dim}
Tensor DTGRUCellImpl::forward(const Tensor &input) {
  Tensor out = torch::empty({input.size(-3), input.size(-2), static_cast<int64_t>(rnn_dim_)},
                            input.device()); // {seq_len, batch_size, rnn_dim}
  Tensor state = torch::zeros({input.size(-2), static_cast<int64_t>(rnn_dim_)},
                              input.device()); // {batch_size, rnn_dim}
  for(int64_t t = 0; t < input.size(0); ++t) {
    state = step(input.index({t, Ellipsis}), state);
    out.index_put_({t, Ellipsis}, state);
  }
  return out;
}

void CondDTGRUCellImpl::set_attention_context(const Tensor &encoder_states, const Tensor &src_mask) {
  att_->set_context(encoder_states, src_mask);
}

void StackedRNNImpl::set_attention_context(const Tensor &encoder_states, const Tensor &src_mask) {
  stack_[0]->as<CondDTGRUCell>()->set_attention_context(encoder_states, src_mask);
}

StackedRNNImpl::StackedRNNImpl(size_t input_dim,
                               size_t hidden_dim,
                               size_t depth,
                               size_t transition_depth,
                               StackedRNNDir dir,
                               bool skip)
    : dir_(dir), rnn_dim_(hidden_dim), skip_(skip) {
  for(size_t l = 1; l <= depth; ++l) {
    stack_->push_back(
      register_module("layer" + std::to_string(l),
                      DTGRUCell(input_dim, hidden_dim, transition_depth)));
      input_dim = hidden_dim; // For layers > 1
  }
}

// Insert a CondDTGRUCell at the bottom of the stack (used in BiDeepDecoder)
void StackedRNNImpl::insert_conditional_cell(size_t input_dim,
                                             size_t hidden_dim,
                                             size_t cell_depth) {
  stack_->insert(0, register_module("base",
                                    CondDTGRUCell(input_dim,
                                                  hidden_dim,
                                                  cell_depth)));
}

// One time step of StackedRNN where first cell is CondDTGRU.
// Used in BiDeepDecoder
// Input input: {batch_size, input_dim}
// Input state: {layers, batch_size, rnn_dim}
// Returns: {layers, batch_size, rnn_dim}
std::tuple<Tensor, Tensor> StackedRNNImpl::step(const Tensor &input, const Tensor &state) {
  Tensor layer_input = input, layer_out;
  Tensor out_state = torch::empty_like(state);
  Tensor att_context;
  std::tie(layer_out, att_context) = stack_[0]->as<CondDTGRUCell>()->step(layer_input, state.index({0, Ellipsis}));
  out_state.index_put_({0, Ellipsis}, layer_out);
  layer_input = layer_out;
  for(int64_t l = 1; l < static_cast<int64_t>(stack_->size()); ++l) {
    layer_out = stack_[l]->as<DTGRUCell>()->step(layer_input, state.index({l, Ellipsis}));
    out_state.index_put_({l, Ellipsis}, layer_out);
    layer_input = layer_out;
  }
  return std::tuple<Tensor, Tensor>(out_state, att_context);
}

// Transduces entire sequence with StackedRNN and returns final layer outputs
// Used in BiDeepEncoder
// Input input: {seq_len, batch_size, input_dim}
// Input lengths: {batch_size}
// Returns: {seq_len, batch_size, rnn_dim}
Tensor StackedRNNImpl::forward(const Tensor &input, const Tensor &lengths) {
  Tensor layer_input = input, layer_out;
  for(size_t l = 0; l < stack_->size(); ++l) {
    bool backward_layer =
        (dir_ == StackedRNNDir::backward
        || (dir_ == StackedRNNDir::alternating_backward && l % 2 == 0)  // Even layers (0, 2, ...)
        || (dir_ == StackedRNNDir::alternating_forward && l % 2 != 0)); // Odd layers (1, 3, ...)
    if(backward_layer) {
      // Reverse layer inputs
      layer_input = reverse_padded_sequence(layer_input, lengths);
    }
    layer_out = stack_[l]->as<DTGRUCell>()->forward(layer_input);
    if(backward_layer) {
      // Reverse layer outputs
      layer_input = reverse_padded_sequence(layer_out, lengths);
    }
    if(skip_ && l > 0) {
      // Skip connections in higher layers
      layer_out += layer_input;
    }
    layer_input = layer_out;
  }
  return layer_input;
}
