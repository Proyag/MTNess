#pragma once

#include <torch/torch.h>
#include <unordered_map>

using torch::Tensor;

// Holds data batches with corresponding mask and sequence lengths
// for use with torch::data::Example<>
struct MaskedData {
 public:
  // MaskedData batch constructor
  MaskedData(const Tensor &data, const Tensor &mask, const Tensor &lengths)
    : data(data), mask(mask), lengths(lengths) {}
  // Constructor for MaskedData single examples (length is a number, not a tensor)
  MaskedData(const Tensor &data, const Tensor &mask, const int64_t &length)
    : MaskedData(data, mask, torch::tensor(length)) {}
  // Method to move both batch and mask to device
  void to(const torch::DeviceType &device, const bool non_blocking=true);

  Tensor data, mask, lengths;
};

enum class StackedRNNDir {
  // https://www.aclweb.org/anthology/W17-4710.pdf
  forward,
  backward,
  alternating_forward,
  alternating_backward
};

enum class EncoderType {
  bidirectional,
  alternating,
  bi_unidirectional
};

// For option parsing
static std::unordered_map<std::string, EncoderType> encoder_type_map{
    {"bidirectional", EncoderType::bidirectional},
    {"alternating", EncoderType::alternating},
    {"bi-unidirectional", EncoderType::bi_unidirectional}};
