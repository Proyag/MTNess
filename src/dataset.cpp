#include <sentencepiece_trainer.h>
#include <spdlog/spdlog.h>
#include <filesystem>
#include "dataset.h"

void MaskedData::to(const torch::DeviceType &device, const bool non_blocking) {
  data = data.to(device, non_blocking);
  mask = mask.to(device, non_blocking);
  lengths = lengths.to(device, non_blocking);
}

TranslationDataset::TranslationDataset(const string& src_path,
                                       const string& trg_path,
                                       std::unique_ptr<SentencePieceProcessor> src_spm_processor,
                                       std::unique_ptr<SentencePieceProcessor> trg_spm_processor,
                                       bool reverse_src)
    : src_file_(std::ifstream(src_path)),
      trg_file_(std::ifstream(trg_path)),
      src_spm_processor_(std::move(src_spm_processor)),
      trg_spm_processor_(std::move(trg_spm_processor)) {
  if(reverse_src) {
    // Reverse source sentence (e.g. for Sutskever-style models)
    src_spm_processor_->SetEncodeExtraOptions("reverse:eos");
  }
}

torch::optional<std::vector<Example<MaskedData, MaskedData>>> TranslationDataset::get_batch(size_t batch_size) {
  std::vector<Example<MaskedData, MaskedData>> batch;
  std::vector<int> src_ids, trg_ids;
  string src_line, trg_line;
  batch.reserve(batch_size);
  while(batch.size() < batch_size && !src_file_.eof() && !trg_file_.eof()) {
    std::getline(src_file_, src_line);
    std::getline(trg_file_, trg_line);
    src_spm_processor_->Encode(src_line, &src_ids);
    trg_spm_processor_->Encode(trg_line, &trg_ids);
    batch.emplace_back(MaskedData(torch::tensor(src_ids),
                                  torch::ones(src_ids.size(), torch::dtype(torch::kBool)),
                                  src_ids.size()),
                       MaskedData(torch::tensor(trg_ids),
                                  torch::ones(trg_ids.size(), torch::dtype(torch::kBool)),
                                  trg_ids.size()));
  }
  if(batch.size() == 0 && src_file_.eof() && trg_file_.eof()) {
    return torch::optional<std::vector<Example<MaskedData, MaskedData>>>();
  }
  return batch;
}

torch::optional<size_t> TranslationDataset::size() const {
  return torch::optional<size_t>();
}

void TranslationDataset::reset() {
  // TODO: There should be shuffling logic here later
  src_file_.clear();
  trg_file_.clear();
  src_file_.seekg(0, std::ios::beg);
  trg_file_.seekg(0, std::ios::beg);
}

void TranslationDataset::save(torch::serialize::OutputArchive &archive) const {
  // TODO
  return;
}

void TranslationDataset::load(torch::serialize::InputArchive &archive) {
  // TODO
  return;
}

// If SPM model exists, loads it.
// If SPM model doesn't exists, creates it.
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
