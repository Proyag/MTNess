#include <torch/data.h>
#include <CLI11/CLI11.hpp>
#include <sentencepiece_processor.h>
#include <spdlog/spdlog.h>
#include <iostream>
#include <memory>
#include <chrono>
#include "cli_options.h"
#include "dataset.h"
#include "batch_transform.h"
#include "models/encdec.h"
#include "models/rnn.h"

// // NASTY: Would be nice to reduce this to a "reduction", but
// // https://pytorch.org/cppdocs/api/structtorch_1_1nn_1_1_cross_entropy_loss_options.html
// // shows that the reduction is restricted to only kNone, kSum, or kMean
// // TODO: Figure out how to make a custom reduction,
// // then use this instead of ignore_index 0 (as eos should be considered)
// Tensor calc_loss(const Tensor &output, const MaskedData &target, torch::nn::CrossEntropyLoss &loss) {
//   return ((loss->forward(output.permute({0,2,1}), target.data) * target.mask).sum(0) / target.lengths).mean();
// }

int main(int argc, char **argv) {
  // Parse CLI arguments
  CLI::App cli{"MTNess"};
  std::shared_ptr<Options> options = configure_cli(cli);
  CLI11_PARSE(cli, argc, argv);

  // Load or create SPM models
  auto src_spm_processor = load_or_create_vocab(options->training_options.spm_models[0],
                                                options->training_options.training_data[0],
                                                options->model_options.vocab_size);
  auto trg_spm_processor = load_or_create_vocab(options->training_options.spm_models[1],
                                                options->training_options.training_data[1],
                                                options->model_options.vocab_size);
  options->model_options.src_vocab_size = src_spm_processor->GetPieceSize();
  options->model_options.trg_vocab_size = trg_spm_processor->GetPieceSize();

  // Initialise dataset and dataloader
  auto dataset = torch::data::datasets::make_shared_dataset<TranslationDataset>(
                     options->training_options.training_data[0],
                     options->training_options.training_data[1],
                     std::move(src_spm_processor),
                     std::move(trg_spm_processor))
                     .map(PadAndStack<>());
  auto dataloader = torch::data::make_data_loader(
      std::move(dataset),
      std::move(DataLoaderOptions()
                  .batch_size(options->training_options.batch_size)
                  .workers(1)
                  .enforce_ordering(true)));

  // Build model
  EncoderDecoder<BiDeepDecoder> model(options->model_options);
  model->to(options->training_options.device, /*non_blocking=*/true);
  model->print_params();

  auto loss_fn = torch::nn::CrossEntropyLoss(torch::nn::CrossEntropyLossOptions().ignore_index(0));

  auto optimizer = torch::optim::Adam(model->parameters(), torch::optim::AdamOptions(options->training_options.learning_rate));

  size_t total_sentences = 0;
  size_t sentences_since_last = 0;
  size_t updates = 0;
  auto last_time = std::chrono::high_resolution_clock::now();

  // Training loop
  for(size_t epoch = 1; epoch <= options->training_options.epochs; ++epoch) {
    for(auto& batch : *dataloader) {
      // Move data to GPU if enabled
      batch.data.to(options->training_options.device);
      batch.target.to(options->training_options.device);

      // Forward pass
      auto decoder_output = model->forward(batch.data, batch.target);
      auto loss = loss_fn->forward(decoder_output.permute({0,2,1}), batch.target.data);

      // Backprop
      loss.backward();
      optimizer.step();

      // For display purposes
      ++updates;
      total_sentences += batch.data.data.size(-1);
      sentences_since_last += batch.data.data.size(-1);
      if(updates % options->training_options.disp_freq == 0) {
        auto curr_time = std::chrono::high_resolution_clock::now();
        auto time_passed = std::chrono::duration_cast<std::chrono::duration<double>>(curr_time - last_time);
        spdlog::info("Epoch: {} ||| Updates: {} ||| Sentences: {} ||| Sentences/second: {:.2f} ||| Loss: {:.5f}", epoch, updates, total_sentences, sentences_since_last / time_passed.count(), loss.item<double>());
        last_time = curr_time;
        sentences_since_last = 0;
      }
    }
  }

  return 0;
}
