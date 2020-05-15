#include "pypoly/core/pypet/expr.h"

namespace pypoly {
namespace pypet {

namespace {
static void PypetFuncArgFree(struct PypetFuncSummaryArg* arg) {
  if (arg->type == PYPET_ARG_INT) isl_id_free(arg->id);
  if (arg->type != PYPET_ARG_ARRAY) return;
  for (size_t type = PYPET_EXPR_ACCESS_BEGIN; type < PYPET_EXPR_ACCESS_END;
       ++type)
    arg->access[type] = isl_union_map_free(arg->access[type]);
}

__isl_null PypetFuncSummary* PypetFuncSummaryFree(
    __isl_take PypetFuncSummary* summary) {
  int i;

  if (!summary) return nullptr;
  if (--summary->ref > 0) return nullptr;

  for (i = 0; i < summary->n; ++i) PypetFuncArgFree(&summary->arg[i]);

  isl_ctx_deref(summary->ctx);
  free(summary);
  return nullptr;
}
}  // namespace

__isl_null PypetExpr* PypetExprFree(__isl_take PypetExpr* expr) {
  if (!expr) return nullptr;
  if (--expr->ref > 0) return nullptr;

  for (unsigned int i = 0; i < expr->arg_num; ++i) PypetExprFree(expr->args[i]);
  free(expr->args);

  switch (expr->type) {
    case PYPET_EXPR_ACCESS:
      isl_id_free(expr->acc.ref_id);
      for (int type = PYPET_EXPR_ACCESS_BEGIN; type < PYPET_EXPR_ACCESS_END;
           ++type)
        isl_union_map_free(expr->acc.access[type]);
      isl_multi_pw_aff_free(expr->acc.index);
      break;
    case PYPET_EXPR_CALL:
      free(expr->call.name);
      PypetFuncSummaryFree(expr->call.summary);
      break;
    case PYPET_EXPR_OP:
    case PYPET_EXPR_ERROR:
      break;
      return nullptr;
  }
}

}  // namespace pypet
}  // namespace pypoly
