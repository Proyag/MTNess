#pragma once

#include <torch/torch.h>
#include <vector>
#include <string>
#include <fstream>
#include <memory>
#include <sentencepiece_processor.h>
#include "types.h"

using namespace torch::data;
using torch::Tensor;
using std::string;
using std::vector;
using sentencepiece::SentencePieceProcessor;

class TranslationDataset : public datasets::StatefulDataset<TranslationDataset, vector<Example<MaskedData, MaskedData>>> {
 public:
  explicit TranslationDataset(const string& src_path,
                              const string& trg_path,
                              std::unique_ptr<SentencePieceProcessor> src_spm_processor,
                              std::unique_ptr<SentencePieceProcessor> trg_spm_processor,
                              bool reverse_src=false);

  torch::optional<std::vector<Example<MaskedData, MaskedData>>> get_batch(size_t batch_size) override;
  void reset() override;
  void save(torch::serialize::OutputArchive &archive) const override;
  void load(torch::serialize::InputArchive &archive) override;
  torch::optional<size_t> size() const override;

 private:
  std::ifstream src_file_;
  std::ifstream trg_file_;
  std::unique_ptr<SentencePieceProcessor> src_spm_processor_;
  std::unique_ptr<SentencePieceProcessor> trg_spm_processor_;
};

std::unique_ptr<SentencePieceProcessor> load_vocab(const string& spm_path);
