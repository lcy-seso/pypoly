#include "pypoly/core/pypet/nest.h"

namespace pypoly {
namespace pypet {

bool PypetNestedInId(isl_id* id) {
  CHECK(id);
  if (isl_id_get_user(id) == nullptr) {
    return false;
  }
  const char* name = isl_id_get_name(id);
  return !strcmp(name, "__PypetExpr");
}

bool PypetNestedInSpace(isl_space* space, int pos) {
  isl_id* id = isl_space_get_dim_id(space, isl_dim_param, pos);
  bool nested = PypetNestedInId(id);
  isl_id_free(id);
  return nested;
}

bool PypetNestedInSet(isl_set* set, int pos) {
  isl_id* id = isl_set_get_dim_id(set, isl_dim_param, pos);
  bool nested = PypetNestedInId(id);
  isl_id_free(id);
  return nested;
}

isl_space* PypetNestedRemoveFromSpace(isl_space* space) {
  int param_num = isl_space_dim(space, isl_dim_param);
  for (int i = param_num - 1; i >= 0; --i) {
    if (PypetNestedInSpace(space, i)) {
      space = isl_space_drop_dims(space, isl_dim_param, i, 1);
    }
  }
  return space;
}

isl_set* PypetNestedRemoveFromSet(isl_set* set) {
  int param_num = isl_set_dim(set, isl_dim_param);
  for (int i = param_num - 1; i >= 0; --i) {
    if (PypetNestedInSet(set, i)) {
      set = isl_set_project_out(set, isl_dim_param, i, 1);
    }
  }
  return set;
}

void PypetExprFreeWrap(void* user) { PypetExprFree((PypetExpr*)user); }

isl_id* PypetNestedPypetExpr(PypetExpr* expr) {
  isl_id* id = isl_id_alloc(PypetExprGetCtx(expr), "__PypetExpr", expr);
  id = isl_id_set_free_user(id, &PypetExprFreeWrap);
  return id;
}

bool PypetNestedAnyInSpace(isl_space* space) {
  int param_num = isl_space_dim(space, isl_dim_param);
  for (int i = 0; i < param_num; ++i) {
    if (PypetNestedInSpace(space, i)) {
      return true;
    }
  }
  return false;
}

PypetExpr* PypetNestedExtractExpr(isl_id* id) {
  return PypetExprCopy(static_cast<PypetExpr*>(isl_id_get_user(id)));
}

bool PypetNestedAnyInPwAff(isl_pw_aff* pa) {
  isl_space* space = isl_pw_aff_get_space(pa);
  bool nested = PypetNestedAnyInSpace(space);
  isl_space_free(space);
  return nested;
}

PypetStmt* PypetStmtResolveNested(PypetStmt* stmt) {
  // TODO
  return nullptr;
}

PypetScop* PypetScopResolveNested(PypetScop* scop) {
  for (int i = 0; i < scop->stmt_num; ++i) {
    scop->stmts[i] = PypetStmtResolveNested(scop->stmts[i]);
  }
  return scop;
}

}  // namespace pypet
}  // namespace pypoly
