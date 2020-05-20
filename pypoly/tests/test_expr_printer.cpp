#include "gtest/gtest.h"
#include "pypoly/core/pypet/expr.h"

#include <isl/arg.h>
#include <isl/ctx.h>
#include <isl/options.h>

#include <iostream>
namespace pypoly {
namespace pypet {

TEST(TestExprPrinter, Test1) {
  struct isl_options* options = isl_options_new_with_defaults();
  isl_ctx* ctx = isl_ctx_alloc_with_options(&isl_options_args, options);

  PypetExpr* expr1 = PypetExprAlloc(ctx, PYPET_EXPR_ERROR);
  std::cout << expr1;
  PypetExprFree(expr1);

  PypetExpr* expr2 = PypetExprAlloc(ctx, PYPET_EXPR_ACCESS);
  std::cout << expr2;
  PypetExprFree(expr2);

  PypetExpr* expr3 = PypetExprFromIntVal(ctx, 5);
  std::cout << expr3;
  PypetExprFree(expr3);

  PypetExpr* expr4 = PypetExprCreateCall(ctx, "tanh", 0);
  std::cout << expr4;
  PypetExprFree(expr4);

  PypetExpr* expr5 = PypetExprAlloc(ctx, PYPET_EXPR_OP);
  expr5->op = PYPET_ADD;
  std::cout << expr5;
  PypetExprFree(expr5);

  isl_ctx_free(ctx);
  ASSERT_TRUE(1);
}
}  // namespace pypet
}  // namespace pypoly
