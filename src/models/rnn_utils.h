#pragma once

#include <torch/torch.h>

using torch::Tensor;

// Reverse the non-padding part of sequences, keeping the padding constant
// Input seq: if batch_first is false: {seq_len, batch_size, dim}
//            if batch_first is true:  {batch_size, seq_len, dim}
// Input lengths: {batch_size}
// Input batch_first: True if first dim of seq is batch
// Returns: Same shape as seq
Tensor reverse_padded_sequence(Tensor &seq, const Tensor &lengths, bool batch_first=false) {
  // https://github.com/pytorch/pytorch/pull/2053#issuecomment-319922541
  using torch::indexing::Slice;
  using torch::indexing::Ellipsis;
  if(batch_first) {
    seq = seq.transpose(0, 1);
  }
  Tensor reversed_indices = torch::arange(0, seq.size(0),
                                          torch::TensorOptions()
                                            .dtype(torch::kLong)
                            ).unsqueeze(-1).unsqueeze(-1).expand_as(seq).clone();
  for(int i = 0; i < lengths.numel(); ++i) {
    long length = lengths[i].item<long>();
    reversed_indices.index_put_({Slice(0, length), i, Ellipsis},
      reversed_indices.index({Slice(0, length), i, Ellipsis}).flip(0));
  }
  reversed_indices = reversed_indices.to(seq.device());
  Tensor reversed_seq = seq.gather(0, reversed_indices);
  if(batch_first) {
    reversed_seq = reversed_seq.transpose(0, 1);
  }
  return reversed_seq;
}
