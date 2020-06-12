#include <torch/nn.h>
#include "decoder.h"

using namespace torch::nn;
using namespace torch::indexing;

BiDeepDecoderImpl::BiDeepDecoderImpl(const ModelOptions &model_options) : rnn_dim_(model_options.rnn_dim) {
  map_to_decoder_ = register_module("map_to_dec_state",
                                    Linear(2 * model_options.rnn_dim,
                                           model_options.rnn_dim));
  emb_ = register_module("emb",
                         Embedding(EmbeddingOptions(model_options.trg_vocab_size,
                                                    model_options.emb_dim)
                                                    .padding_idx(0)));
  rnn_ = register_module("rnn",
                         StackedRNN(model_options.rnn_dim,
                                    model_options.rnn_dim,
                                    model_options.dec_depth - 1,
                                    model_options.dec_high_cell_depth,
                                    StackedRNNDir::forward,
                                    model_options.skip));
  // Insert CondDTGRUCell at position 0
  rnn_->insert_conditional_cell(model_options.emb_dim,
                                model_options.rnn_dim,
                                model_options.dec_base_cell_depth);
  output_ = register_module("output",
                            DeepOutput(model_options.emb_dim,
                                       model_options.rnn_dim,
                                       model_options.trg_vocab_size));

  if(model_options.tied_embeddings) {
    output_->set_weight_matrix(emb_->weight);
  }
}

// Masked average over time steps of encoder states
// Returns {batch_size, rnn_dim}
Tensor BiDeepDecoderImpl::start_state(const Tensor &encoder_output, const Tensor &src_lengths, const Tensor &src_mask) {
  return torch::tanh(
    map_to_decoder_->forward(
      (encoder_output * src_mask.unsqueeze(-1))
        .sum(/*dim=*/0, /*keepdim=*/false)
        / src_lengths.unsqueeze(-1)));
}

// One time step of decoder
std::tuple<Tensor, Tensor> BiDeepDecoderImpl::step(const Tensor &input, Tensor state) {
  return rnn_->step(input, state);
}

// Returns {seq_len, batch_size, vocab_size}
Tensor BiDeepDecoderImpl::forward(const Tensor &encoder_output, const Tensor &src_lengths, const Tensor &src_mask, const Tensor &trg_input) {
  // Embed target inputs
  Tensor trg_embedded = emb_->forward(trg_input);

  // Placeholders. Stacking as we go would cause repeated reallocation
  Tensor batch_states = torch::empty({trg_embedded.size(0), trg_embedded.size(1), static_cast<int64_t>(rnn_dim_)},
                                      torch::TensorOptions().device(trg_embedded.device()));
  Tensor batch_contexts = torch::empty({trg_embedded.size(0), trg_embedded.size(1), 2 * static_cast<int64_t>(rnn_dim_)},
                                        torch::TensorOptions().device(trg_embedded.device()));

  Tensor state = start_state(encoder_output, src_lengths, src_mask);
  // Expand start state for each layer
  // {batch_size, rnn_dim} -> {layers, batch_size, rnn_dim}
  state = state.unsqueeze(-3).expand({rnn_->num_layers(), -1, -1}).clone();

  rnn_->set_attention_context(encoder_output, src_mask);
  Tensor att_context;

  int64_t seq_len = trg_embedded.size(0);
  Tensor first_input = torch::zeros_like(trg_embedded.index({0, Ellipsis}));

  // Loop over time steps
  for(int64_t i = 0; i < seq_len; i++) {
    if(i == 0) {
      // Zero input for first step
      std::tie(state, att_context) = step(first_input, state);
    }
    else {
      // Input previous target word
      std::tie(state, att_context) = step(trg_embedded.index({i - 1, Ellipsis}), state);
    }
    batch_contexts.index_put_({i, Ellipsis}, att_context);
    batch_states.index_put_({i, Ellipsis}, state.index({-1, Ellipsis}));
  }
  return output_->forward(trg_embedded, batch_states, batch_contexts);
}

BiDeepDecoderImpl::DeepOutputImpl::DeepOutputImpl(size_t emb_dim, size_t rnn_dim, size_t vocab_size) {
  out_emb_ = register_module("out_emb",
                             Linear(emb_dim, emb_dim));
  out_dec_ = register_module("out_dec",
                             Linear(rnn_dim, emb_dim));
  out_context_ = register_module("out_context",
                                 Linear(2 * rnn_dim, emb_dim));
  output_ = register_module("out_final",
                            Linear(emb_dim, vocab_size));
}

// Return {seq_len, batch_size, vocab_size}
Tensor BiDeepDecoderImpl::DeepOutputImpl::forward(const Tensor &prev_embedding, const Tensor &dec_state, const Tensor &context) {
  return output_->forward(
    torch::tanh(
      out_emb_->forward(prev_embedding) + out_dec_(dec_state) + out_context_(context)));
}

void BiDeepDecoderImpl::DeepOutputImpl::set_weight_matrix(const Tensor &weight) {
  output_->weight = weight;
}
