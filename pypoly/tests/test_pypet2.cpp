#include "gtest/gtest.h"
#include "pypoly/core/pypet/parser.h"

namespace pypoly {
namespace pypet {
// paste the implementation of forward in
// pypet/python/examples/stacked_rnn_example.py
const auto test_source = R"PYPET(
def forward(self, input: ReadTensorArray, batch_size: int,
            seq_lens: List[int], depth: int, output: ReadWriteTensorArray):
    for i in range(batch_size):
        seq_len = seq_lens[i]
        for j in range(seq_len):  # data-dependent loop bound.
            output[i][j] = input[i][j]
)PYPET";

TEST(ParserTest, Test2) {
  TorchParser p(test_source);
  ScopParser scop_parser(p.Parse());
  scop_parser.Parse();

  ASSERT_TRUE(1);
}
}  // namespace pypet
}  // namespace pypoly
