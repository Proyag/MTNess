#include <torch/nn.h>
#include <spdlog/spdlog.h>
#include "decoder.h"

using namespace torch::nn;
using namespace torch::indexing;

SutskeverDecoderImpl::SutskeverDecoderImpl(const ModelOptions &model_options) {
  emb_ = register_module("emb",
                         Embedding(EmbeddingOptions(model_options.trg_vocab_size,
                                                    model_options.emb_dim)
                                                    .padding_idx(0)));
  rnn_ = register_module("rnn",
                         GRU(GRUOptions(2 * model_options.rnn_dim,
                                        model_options.rnn_dim)
                                        .num_layers(model_options.dec_depth)));
  output_ = register_module(
      "output", Linear(model_options.rnn_dim, model_options.trg_vocab_size));

  if (model_options.emb_dim != model_options.rnn_dim) {
    spdlog::error("Embedding dimension must match RNN dimension for Sutskever model");
  }

  if (model_options.tied_embeddings) {
    output_->weight = emb_->weight;
  }
}

// For each sequence in the batch, set the start state as the last, i.e. (length-1)-th state
// Returns {1, batch_size, 2*rnn_dim}
Tensor SutskeverDecoderImpl::start_state(const Tensor &encoder_output, const Tensor &src_lengths, const Tensor &src_mask) {
  auto start_state = encoder_output.gather(/*dim=*/0,
                        (src_lengths - 1).unsqueeze(-1).expand({src_lengths.size(0), encoder_output.size(-1)}).unsqueeze(0));
  return start_state;
}

std::tuple<Tensor, Tensor> SutskeverDecoderImpl::step(const Tensor &input, Tensor state) {
  return rnn_->forward(input, state);
}

// Returns {seq_len, batch_size, vocab_size}
Tensor SutskeverDecoderImpl::forward(const Tensor &encoder_output, const Tensor &lengths, const Tensor &src_mask, const Tensor &trg_input) {
  Tensor trg_embedded = emb_->forward(trg_input);
  auto state = start_state(encoder_output, lengths, src_mask);
  Tensor decoder_output = torch::empty(
      trg_embedded.sizes(), torch::TensorOptions().device(trg_embedded.device()));
  Tensor step_output = torch::empty_like(state);

  for (int i = 0; i < trg_embedded.size(0); i++) {
    std::tie(step_output, state) = step(
        trg_embedded.index({Slice(i, i + 1), Ellipsis}), state);
    decoder_output.index_put_({i, Ellipsis}, step_output);
  }
  return output_->forward(decoder_output);
}
