#include "gtest/gtest.h"
#include "pypoly/core/pypet/tree.h"

#include <isl/arg.h>
#include <isl/ctx.h>
#include <isl/options.h>
#include <torch/csrc/jit/frontend/source_range.h>

#include <iostream>
namespace pypoly {
namespace pypet {

PypetTree* GenTestExample() {
  struct isl_options* options = isl_options_new_with_defaults();
  isl_ctx* ctx = isl_ctx_alloc_with_options(&isl_options_args, options);

  torch::jit::SourceRange range = torch::jit::SourceRange();
  PypetTree* tree =
      CreatePypetTreeBlock(ctx, range, 1, 1 /*1 statement in the block*/);

  PypetTreeFree(tree);
  isl_ctx_free(ctx);
  return tree;
}

TEST(TestTree2Scop, Test1) {
  PypetTree* tree = GenTestExample();
  std::cout << tree << std::endl;
  ASSERT_TRUE(1);
}
}  // namespace pypet
}  // namespace pypoly
