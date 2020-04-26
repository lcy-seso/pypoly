#include "gtest/gtest.h"
#include "pypet/core/parser.h"

namespace pypet {
const auto test_source = R"PYPET(
def forward(self, x: Tensor, batch_size: int, seq_len: int) -> Tensor:
    for i in range(batch_size):
        for j in range(seq_len):
            y = torch.nn.Linear(x)
    return y
)PYPET";

TEST(ParserTest, Test1) {
  TorchParser p(test_source);
  ScopParser scop_parser(p.Parse());
  scop_parser.DumpAST();

  ASSERT_TRUE(1);
}
}  // namespace pypet
