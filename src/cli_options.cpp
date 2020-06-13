#include <map>
#include <string>
#include "cli_options.h"

std::shared_ptr<Options> configure_cli(CLI::App &app) {
  app.require_subcommand();
  auto options = std::make_shared<Options>();

  // Sub-commands
  auto train = app.add_subcommand("train", "MTNess model training");
  // auto translate = app.add_subcommand("translate", "MTNess translation");
  
  app.add_option("--emb-dim",
                 options->model_options.emb_dim,
                 "Dimension of embeddings",
                 true)
      ->check(CLI::PositiveNumber);
  app.add_option("--rnn-dim",
                 options->model_options.rnn_dim,
                 "Hidden dimension of RNN",
                 true)
      ->check(CLI::PositiveNumber);
  app.add_option("--vocab-size",
                 options->model_options.vocab_size,
                 "Maximum size of vocab",
                 true)
      ->check(CLI::PositiveNumber);
  app.add_option("--enc-type",
                 options->model_options.enc_type,
                 "Type of encoder")
      ->transform(CLI::CheckedTransformer(encoder_type_map, CLI::ignore_case));
  app.add_option("--enc-depth",
                 options->model_options.enc_depth,
                 "Number of stacked layers in encoder RNN",
                 true)
      ->check(CLI::PositiveNumber);
  app.add_option("--enc-cell-depth",
                 options->model_options.enc_cell_depth,
                 "Number of deep transition cells in encoder RNN",
                 true)
      ->check(CLI::PositiveNumber);
  app.add_option("--dec-depth",
                 options->model_options.dec_depth,
                 "Number of stacked layers in the decoder RNN",
                 true)
      ->check(CLI::PositiveNumber);
  app.add_option("--dec-base-cell-depth",
                 options->model_options.dec_base_cell_depth,
                 "Number of deep transition cells in first layer of decoder RNN",
                 true)
      ->check(CLI::PositiveNumber);
  app.add_option("--dec-high-cell-depth",
                 options->model_options.dec_high_cell_depth,
                 "Number of deep transition cells in higher layers of decoder RNN",
                 true)
      ->check(CLI::NonNegativeNumber);
  app.add_flag("--tied-embeddings",
               options->model_options.tied_embeddings,
               "Tie target embeddings and output layer weights");
  app.add_flag("--skip",
               options->model_options.skip,
               "Skip (residual) connections in RNN stacks");
  
  train->add_option("--training-data",
                    options->training_options.training_data,
                    "Paths to training datasets")
      ->required()
      ->expected(2)
      ->check(CLI::ExistingFile);
  train->add_option("--spm-model",
                    options->training_options.spm_models,
                    "Path to SPM models. Created if non-existent")
      ->required()
      ->expected(2);
  train->add_flag("--reverse-src",
                  options->training_options.reverse_src,
                  "Reverse source sentences");
  train->add_option("--max-length",
                    options->training_options.max_length,
                    "Max length of sentences, longer ones are cropped",
                    true)
      ->check(CLI::PositiveNumber);
  train->add_option("--batch-size",
                    options->training_options.batch_size,
                    "Training minibatch size",
                    true)
      ->check(CLI::PositiveNumber);
  train->add_option("--learning-rate,--lr",
                    options->training_options.learning_rate,
                    "Initial learning rate for Adam",
                    true);
  train->add_option("--epochs",
                    options->training_options.epochs,
                    "Number of epochs to train",
                    true)
      ->check(CLI::NonNegativeNumber);
  train->add_flag("--cpu,!--gpu",
                  options->training_options.cpu,
                  "No GPU, use CPU only");
  train->add_option("--disp-freq",
                    options->training_options.disp_freq,
                    "Display status after every n updates",
                    true)
      ->check(CLI::PositiveNumber);

  app.callback([options]() {
    if(!torch::cuda::is_available() || options->training_options.cpu) {
      std::cerr << "GPU disabled or not found. Using CPU only" << std::endl;
      options->training_options.cpu = true;
      options->training_options.device = torch::kCPU;
    }
  });

  return options;
}
