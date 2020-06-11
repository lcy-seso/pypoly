#include "pypoly/core/pypet/expr_arg.h"

#include "pypoly/core/pypet/context.h"
#include "pypoly/core/pypet/isl_printer.h"

namespace pypoly {
namespace pypet {

namespace {

PypetExpr* SpliceSum(PypetExpr* expr, int dim, int pos) {
  CHECK(expr);
  PypetExpr* arg = PypetExprCopy(expr->args[pos]);
  expr = PypetExprInsertArg(expr, pos + 1, arg->args[1]);
  expr = PypetExprSetArg(expr, pos, arg->args[0]);
  isl_space* space = isl_multi_pw_aff_get_space(expr->acc.index);
  space = isl_space_map_from_set(isl_space_domain(space));
  isl_multi_aff* multi_aff = isl_multi_aff_identity(space);
  isl_aff* lhs = isl_multi_aff_get_aff(multi_aff, dim + pos);
  isl_aff* rhs = isl_multi_aff_get_aff(multi_aff, dim + pos + 1);
  lhs = isl_aff_add(lhs, rhs);
  multi_aff = isl_multi_aff_set_aff(multi_aff, dim + pos, lhs);
  return expr->AccessPullbackMultiAff(multi_aff);
}

PypetExpr* PlugInArgs(PypetExpr* expr, void* user) {
  PypetContext* context = static_cast<PypetContext*>(user);
  return PypetExprAccessPlugInArgs(expr, context);
}

}  // namespace

PypetExpr* PypetExprAccessPlugInArgs(PypetExpr* expr, PypetContext* context) {
  CHECK(expr->type == PypetExprType::PYPET_EXPR_ACCESS);
  expr = expr->RemoveDuplicateArgs();
  if (expr->arg_num == 0) {
    return expr;
  }
  for (int i = expr->arg_num - 1; i >= 0; --i) {
    PypetExpr* arg = expr->args[i];
    isl_pw_aff* pw_aff = PypetExprExtractAffine(arg, context);
    if (!isl_pw_aff_involves_nan(pw_aff)) {
      expr = expr->PlugIn(i, pw_aff);
      continue;
    }
    isl_pw_aff_free(pw_aff);
    if (arg->type == PYPET_EXPR_OP && arg->op == PYPET_ADD) {
      expr = SpliceSum(expr, PypetContextDim(context), i);
      i += 2;
    }
  }
  return expr;
}

PypetExpr* PypetExprPlugInArgs(PypetExpr* expr, PypetContext* context) {
  return PypetExprMapExprOfType(expr, PypetExprType::PYPET_EXPR_ACCESS,
                                PlugInArgs, context);
}

}  // namespace pypet
}  // namespace pypoly
