#include "encoder.h"
#include "types.h"

BiDeepEncoderImpl::BiDeepEncoderImpl(const ModelOptions &model_options)
    : type_(model_options.enc_type), skip_(model_options.skip) {
  // Embedding
  emb_ = register_module("emb",
                         Embedding(EmbeddingOptions(model_options.src_vocab_size,
                                                    model_options.emb_dim)
                                                    .padding_idx(0)));

  size_t bi_layers, uni_layers, depth = model_options.enc_depth;
  StackedRNNDir forward_dir = type_ == EncoderType::alternating ? StackedRNNDir::alternating_forward
                                                                : StackedRNNDir::forward;
  StackedRNNDir backward_dir = type_ == EncoderType::alternating ? StackedRNNDir::alternating_backward
                                                                 : StackedRNNDir::backward;

  if(type_ == EncoderType::bidirectional || type_ == EncoderType::alternating) {
    // <depth> forward and backward layers
    bi_layers = depth;
    uni_layers = 0;
  }
  else {  // type_ == StackedRNNType::bi_unidirectional
    // 1 bidirectional layer, <depth>-1 forward layers
    bi_layers = 1;
    uni_layers = depth - 1;
  }

  // Forward stack
  rnn_fw_ = register_module("rnn_forward",
                            StackedRNN(model_options.emb_dim,
                                       model_options.rnn_dim,
                                       bi_layers,
                                       model_options.enc_cell_depth,
                                       forward_dir));

  // Backward stack
  rnn_bw_ = register_module("rnn_backward",
                            StackedRNN(model_options.emb_dim,
                                       model_options.rnn_dim,
                                       bi_layers,
                                       model_options.enc_cell_depth,
                                       backward_dir));

  // Optional unidirectional stack
  if(uni_layers > 0) {
    rnn_uni_ = register_module("rnn_uni",
                               StackedRNN(2 * model_options.rnn_dim,
                                          2 * model_options.rnn_dim,
                                          uni_layers,
                                          model_options.enc_cell_depth));
  }
}

// Returns {seq_len, batch_size, 2*rnn_dim}
Tensor BiDeepEncoderImpl::forward(const MaskedData &input) {
  Tensor emb_input = emb_->forward(input.data);
  Tensor out = torch::cat({rnn_fw_->forward(emb_input, input.lengths),
                           rnn_bw_->forward(emb_input, input.lengths)},
                           /*dim=*/-1);
  if(type_ == EncoderType::bi_unidirectional) {
    out = rnn_uni_->forward(out, input.lengths);
  }
  return out;
}
