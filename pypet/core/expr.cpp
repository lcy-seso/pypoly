#include "pypet/core/expr.h"

namespace pypet {

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
      // TODO(Ying): Not implmented yet.
      break;
    case PYPET_EXPR_OP:
    case PYPET_EXPR_ERROR:
      break;
      return nullptr;
  }
}
}  // namespace pypet
