#pragma once

#include <torch/torch.h>
#include <vector>

using namespace torch::data;
using torch::indexing::Slice;
using torch::indexing::None;
using torch::nn::utils::rnn::pad_sequence;

// Transform used to pad and stack batches yielded by dataset
// Returns batches of shape {seq_len, batch_size}
template <typename T = Example<MaskedData, MaskedData>>
class PadAndStack;

template <>
class PadAndStack<Example<MaskedData, MaskedData>>
    : public transforms::Collation<Example<MaskedData, MaskedData>> {
 public:
  PadAndStack() {}
  Example<MaskedData, MaskedData> apply_batch(vector<Example<MaskedData, MaskedData>> examples) override {
    vector<Tensor> data, targets, data_mask, targets_mask;
    vector<int64_t> data_lengths, target_lengths;

    // Reserve memory to avoid repeated reallocation
    data.reserve(examples.size());
    targets.reserve(examples.size());
    data_mask.reserve(examples.size());
    targets_mask.reserve(examples.size());
    data_lengths.reserve(examples.size());
    target_lengths.reserve(examples.size());

    for (Example<MaskedData, MaskedData>& example : examples) {
      data.emplace_back(std::move(example.data.data));
      targets.emplace_back(std::move(example.target.data));
      data_mask.emplace_back(std::move(example.data.mask));
      targets_mask.emplace_back(std::move(example.target.mask));
      data_lengths.emplace_back(std::move(example.data.lengths.item<int64_t>()));
      target_lengths.emplace_back(std::move(example.target.lengths.item<int64_t>()));
    }

    // Pad batch and masks to the length of the longest sequence
    Tensor padded_data = pad_sequence(std::move(data));
    Tensor padded_targets = pad_sequence(std::move(targets));
    Tensor padded_data_mask = pad_sequence(std::move(data_mask));
    Tensor padded_targets_mask = pad_sequence(std::move(targets_mask));

    return {MaskedData(padded_data,
                       padded_data_mask,
                       torch::tensor(data_lengths)),
            MaskedData(padded_targets,
                       padded_targets_mask,
                       torch::tensor(target_lengths))};
  }
};

template <>
class PadAndStack<Example<MaskedData, example::NoTarget>>
    : public transforms::Collation<Example<MaskedData, example::NoTarget>> {
 public:
  PadAndStack() {}
  Example<MaskedData, example::NoTarget> apply_batch(std::vector<Example<MaskedData, example::NoTarget>> examples) override {
    vector<torch::Tensor> data, data_mask;
    vector<int64_t> data_lengths;
    data.reserve(examples.size());
    data_mask.reserve(examples.size());
    data_lengths.reserve(examples.size());
    for (auto& example : examples) {
      data.emplace_back(std::move(example.data.data));
      data_mask.emplace_back(std::move(example.data.mask));
      data_lengths.emplace_back(std::move(example.data.lengths[0].item<int64_t>()));
    }

    // Pad batch to the length of the longest sequence
    Tensor padded_data = pad_sequence(std::move(data));
    Tensor padded_data_mask = pad_sequence(std::move(data_mask));
    return MaskedData(padded_data, padded_data_mask, torch::tensor(data_lengths));
  }
};
