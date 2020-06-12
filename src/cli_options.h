#pragma once

#include <torch/torch.h>
#include <CLI11/CLI11.hpp>
#include <memory>
#include <string>
#include <vector>
#include "types.h"

using std::string;
using std::vector;

struct GeneralOptions {};

struct ModelOptions {
  EncoderType enc_type = EncoderType::bidirectional;
  size_t emb_dim = 512;
  size_t rnn_dim = 1024;
  size_t enc_depth = 1;
  size_t enc_cell_depth = 1;
  size_t dec_depth = 1;
  size_t dec_base_cell_depth = 2;
  size_t dec_high_cell_depth = 1;
  size_t src_vocab_size;
  size_t trg_vocab_size;
  bool tied_embeddings = false;
  bool skip = false;
};

struct TrainingOptions {
  vector<string> training_data;
  vector<string> spm_models;
  bool reverse_src = false;
  size_t batch_size = 32;
  size_t epochs = 1;
  size_t max_length = 100;
  bool cpu = false;
  torch::DeviceType device = torch::kCUDA;
  double learning_rate = 1e-3;
  size_t disp_freq = 100;
};

struct ValidationOptions {};

struct TranslationOptions {};

struct Options {
  GeneralOptions general_options;
  ModelOptions model_options;
  TrainingOptions training_options;
  ValidationOptions validation_options;
  TranslationOptions translation_options;
};

std::shared_ptr<Options> configure_cli(CLI::App &app);
