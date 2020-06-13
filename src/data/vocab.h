#include <sentencepiece_processor.h>
#include <sentencepiece_trainer.h>
#include <spdlog/spdlog.h>
#include <filesystem>
#include <string>
#include <sstream>

using sentencepiece::SentencePieceProcessor;
using std::string;

std::unique_ptr<SentencePieceProcessor> load_vocab(const string &spm_path);
void create_vocab(const string &spm_path, const string &text_file, const size_t vocab_size);
std::unique_ptr<SentencePieceProcessor> load_or_create_vocab(const string &spm_path, const string &text_file, const size_t &vocab_size);

// If SPM model exists, loads it.
// If SPM model doesn't exist, creates it.
std::unique_ptr<SentencePieceProcessor> load_or_create_vocab(const string &spm_path,
                                                             const string &text_file,
                                                             const size_t &vocab_size) {
  if(!std::filesystem::exists(spm_path + ".model")) {
    // Create new vocab if file doesn't exist
    spdlog::info("SPM model with prefix {} not found - Creating new model", spm_path);
    create_vocab(spm_path, text_file, vocab_size);
  }
  auto spm_processor = load_vocab(spm_path);
  return spm_processor;
}

// Loads an existing SPM model
std::unique_ptr<SentencePieceProcessor> load_vocab(const string &spm_path) {
  std::unique_ptr<SentencePieceProcessor> spm_processor = std::make_unique<SentencePieceProcessor>();
  string model_name = spm_path + ".model";
  spdlog::info("Loading SentencePiece model from {}", model_name);
  const auto spm_load_status = spm_processor->Load(model_name);
  // Check spm load status
  if (!spm_load_status.ok()) {
    spdlog::error("SentencePiece loading error: {}", spm_load_status.ToString());
  }
  spm_processor->SetEncodeExtraOptions("eos");
  return spm_processor;
}

// Creates a new SPM model
void create_vocab(const string &spm_path, const string &text_file, const size_t vocab_size) {
  // TODO: Common vocab for multiple input files
  std::stringstream train_cmd;
  train_cmd << " --bos_id=-1 --eos_id=0 --unk_id=1"; // Non-negotiable
  train_cmd << " --hard_vocab_limit=false"; // Is this necessary?
  train_cmd << " --vocab_size=" << vocab_size;
  train_cmd << " --model_prefix=" << spm_path;
  train_cmd << " --input=" + text_file;
  const auto train_status = sentencepiece::SentencePieceTrainer::Train(train_cmd.str());
  if(!train_status.ok()) {
    spdlog::error("SentencePiece training error: {}", train_status.ToString());
  }
  std::filesystem::remove(spm_path + ".vocab"); // .vocab files are not used
}
