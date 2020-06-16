#include <algorithm>
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
                                       const size_t& maxibatch_size,
                                       const MaxiBatchSortKey sort,
                                       bool reverse_src)
    : src_file_(std::ifstream(src_path)),
      trg_file_(std::ifstream(trg_path)),
      src_spm_processor_(std::move(src_spm_processor)),
      trg_spm_processor_(std::move(trg_spm_processor)),
      maxi_batch_(maxibatch_size, sort) {
  if(reverse_src) {
    // Reverse source sentence (e.g. for Sutskever-style models)
    src_spm_processor_->SetEncodeExtraOptions("reverse:eos");
  }
}

torch::optional<std::vector<Example<MaskedData, MaskedData>>> TranslationDataset::get_batch(const size_t batch_size) {
  std::vector<Example<MaskedData, MaskedData>> batch;
  if(maxi_batch_.empty()) {
    maxi_batch_.fill(src_file_, trg_file_, src_spm_processor_, trg_spm_processor_, batch_size);
  }
  batch.reserve(batch_size);
  while(batch.size() < batch_size && !maxi_batch_.empty()) {
    const auto [src_ids, trg_ids] = maxi_batch_.pop();
    batch.emplace_back(MaskedData(torch::tensor(src_ids),
                                  torch::ones(src_ids.size(), torch::dtype(torch::kBool)),
                                  src_ids.size()),
                       MaskedData(torch::tensor(trg_ids),
                                  torch::ones(trg_ids.size(), torch::dtype(torch::kBool)),
                                  trg_ids.size()));
  }
  if(batch.size() == 0 && src_file_.eof() && trg_file_.eof()) {
    // End of epoch
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

MaxiBatch::MaxiBatch(const size_t& maxibatch_size, const MaxiBatchSortKey sort)
    : maxi_batch_({}), maxibatch_size_(maxibatch_size), sort_(sort) {}

array<vector<int>, 2> MaxiBatch::pop() {
  auto el = std::move(maxi_batch_.front());
  maxi_batch_.pop_front();
  return el;
}

// Sorts the loaded batches according to sort_.
// Order unchanged if sort_ is MaxiBatchSortKey::none
void MaxiBatch::sort() {
  if(sort_ == MaxiBatchSortKey::none) {
    // No sorting
    return;
  }
  // Sort maxi_batch
  int sort_key = sort_ == MaxiBatchSortKey::source ? 0 : 1;
  std::sort(maxi_batch_.begin(),
            maxi_batch_.end(),
            [&sort_key](const auto& lhs, const auto& rhs) {
                return lhs[sort_key].size() < rhs[sort_key].size();
            });
}

void MaxiBatch::fill(std::ifstream &src_file,
                     std::ifstream &trg_file,
                     const std::unique_ptr<SentencePieceProcessor> &src_spm_processor,
                     const std::unique_ptr<SentencePieceProcessor> &trg_spm_processor,
                     const size_t &minibatch_size) {
  string src_line, trg_line;
  vector<int> src_ids, trg_ids;
  while(maxi_batch_.size() < maxibatch_size_ * minibatch_size && !src_file.eof() && !trg_file.eof()) {
    std::getline(src_file, src_line);
    std::getline(trg_file, trg_line);
    src_spm_processor->Encode(src_line, &src_ids);
    trg_spm_processor->Encode(trg_line, &trg_ids);
    maxi_batch_.emplace_back(array<vector<int>, 2>({src_ids, trg_ids}));
  }
  sort();
}
