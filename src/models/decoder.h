#pragma once

#include <torch/nn.h>
#include "cli_options.h"
#include "attention.h"
#include "rnn.h"

using namespace torch::nn;
using namespace torch::indexing;
using torch::Tensor;

class GenericDecoderImpl : public Module {
 public:
  virtual Tensor start_state(const Tensor &encoder_output, const Tensor &lengths, const Tensor &src_mask) = 0;
  virtual std::tuple<Tensor, Tensor> step(const Tensor &input, Tensor state) = 0;
  virtual Tensor forward(const Tensor &encoder_output, const Tensor &lengths, const Tensor &src_mask, const Tensor &trg_input) = 0;
};


// Basic Sutskever-style decoder without attention
class SutskeverDecoderImpl : public GenericDecoderImpl {
 public:
  explicit SutskeverDecoderImpl(const ModelOptions &model_options);
  virtual std::tuple<Tensor, Tensor> step(const Tensor &input, Tensor state) override;
  virtual Tensor forward(const Tensor &encoder_output, const Tensor &lengths, const Tensor &src_mask, const Tensor &trg_input) override;

 private:
  Embedding emb_{nullptr};
  GRU rnn_{nullptr};
  Linear output_{nullptr};

  virtual Tensor start_state(const Tensor &encoder_output, const Tensor &lengths, const Tensor &src_mask) override;
};
TORCH_MODULE(SutskeverDecoder);


// Based on Nematus/Marian: https://www.aclweb.org/anthology/E17-3017.pdf
class BiDeepDecoderImpl : public GenericDecoderImpl {
 public:
  explicit BiDeepDecoderImpl(const ModelOptions &model_options);
  virtual std::tuple<Tensor, Tensor> step(const Tensor &input, Tensor state) override;
  virtual Tensor forward(const Tensor &encoder_output, const Tensor &lengths, const Tensor &src_mask, const Tensor &trg_input) override;

 private:
  // Deep output layer
  class DeepOutputImpl : public Module {
   public:
    DeepOutputImpl(size_t emb_dim, size_t rnn_dim, size_t vocab_size);
    Tensor forward(const Tensor &prev_embedding, const Tensor &dec_state, const Tensor &context);
    void set_weight_matrix(const Tensor &weight);

   private:
    Linear output_{nullptr};
    Linear out_emb_{nullptr};
    Linear out_dec_{nullptr};
    Linear out_context_{nullptr};
    
  };
  TORCH_MODULE(DeepOutput);

  Linear map_to_decoder_{nullptr};
  Embedding emb_{nullptr};
  StackedRNN rnn_{nullptr};
  DeepOutput output_{nullptr};

  size_t rnn_dim_;

  virtual Tensor start_state(const Tensor &encoder_output, const Tensor &lengths, const Tensor &mask) override;
};
TORCH_MODULE(BiDeepDecoder);
