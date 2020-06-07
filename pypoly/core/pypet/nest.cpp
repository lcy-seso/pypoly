#include "pypoly/core/pypet/nest.h"

#include "pypoly/core/pypet/aff.h"
#include "pypoly/core/pypet/tree.h"

namespace pypoly {
namespace pypet {

namespace {

PypetExpr* PypetTreeResolveNestedFunc(PypetExpr* expr, void* user) {
  isl_space* space = static_cast<isl_space*>(user);
  return PypetExprResolveNested(expr, space);
}

void PypetExprFreeWrap(void* user) { PypetExprFree((PypetExpr*)user); }

int SetDim(PypetExpr* expr, void* user) {
  int* dim = static_cast<int*>(user);
  isl_space* space = PypetExprAccessGetDomainSpace(expr);
  *dim = isl_space_dim(space, isl_dim_set);
  isl_space_free(space);
  return -1;
}

bool ExprIsNan(PypetExpr* expr) {
  if (expr->type != PypetExprType::PYPET_EXPR_INT) {
    return false;
  }
  isl_val* v = isl_val_copy(expr->i);
  int is_nan = isl_val_is_nan(v);
  isl_val_free(v);
  return is_nan;
}

}  // namespace

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

int PypetNestedNInSpace(isl_space* space) {
  int param_num = isl_space_dim(space, isl_dim_param);
  int n = 0;
  for (int i = 0; i < param_num; ++i) {
    if (PypetNestedInSpace(space, i)) {
      ++n;
    }
  }
  return n;
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

int PypetExprDomainDim(PypetExpr* expr) {
  int dim = -1;
  if (PypetExprForeachAccessExpr(expr, SetDim, &dim) >= 0) {
    return 0;
  }
  return dim;
}

PypetExpr* Embed(PypetExpr* expr, isl_space* space) {
  int n = PypetExprDomainDim(expr);
  CHECK_GE(n, 0);

  space = isl_space_copy(space);
  isl_multi_pw_aff* multi_pw_aff =
      isl_multi_pw_aff_from_multi_aff(PypetPrefixProjection(space, n));
  return PypetExprUpdateDomain(expr, multi_pw_aff);
}

int PypetExtractNestedFromSpace(isl_space* space, int arg_num, PypetExpr** args,
                                int* param2pos) {
  isl_space* domain = isl_space_copy(space);
  domain = PypetNestedRemoveFromSpace(domain);
  int param_num = isl_space_dim(space, isl_dim_param);
  for (int i = 0; i < param_num; ++i) {
    isl_id* id = isl_space_get_dim_id(space, isl_dim_param, i);
    if (!PypetNestedInId(id)) {
      isl_id_free(id);
      continue;
    }

    args[arg_num] = Embed(PypetNestedExtractExpr(id), domain);
    isl_id_free(id);
    CHECK(args[arg_num]);

    int j = 0;
    for (; j < arg_num; ++j) {
      if (args[j]->IsEqual(args[arg_num])) {
        break;
      }
    }

    if (j < arg_num) {
      PypetExprFree(args[arg_num]);
      args[arg_num] = nullptr;
      param2pos[i] = j;
    } else {
      param2pos[i] = arg_num++;
    }
  }
  isl_space_free(domain);
  return arg_num;
}

PypetExpr* PypetExprExtractNested(PypetExpr* expr, int n, int* param2pos) {
  isl_ctx* ctx = PypetExprGetCtx(expr);
  PypetExpr** args = isl_calloc_array(ctx, PypetExpr*, n);
  CHECK(args);

  int arg_num = expr->arg_num;
  isl_space* space = PypetExprAccessGetDomainSpace(expr);
  n = PypetExtractNestedFromSpace(space, 0, args, param2pos);
  isl_space_free(space);
  CHECK_GE(n, 0);

  expr = PypetExprSetNArgs(expr, arg_num + n);
  for (int i = 0; i < n; ++i) {
    expr = PypetExprSetArg(expr, arg_num + i, args[i]);
  }
  free(args);
  return expr;
}

PypetExpr* PypetExprResolveNested(PypetExpr* expr, isl_space* domain) {
  CHECK(expr);
  CHECK(domain);

  int arg_num = expr->arg_num;
  for (int i = 0; i < arg_num; ++i) {
    PypetExpr* arg = PypetExprGetArg(expr, i);
    arg = PypetExprResolveNested(arg, domain);
    expr = PypetExprSetArg(expr, i, arg);
  }

  if (expr->type != PypetExprType::PYPET_EXPR_ACCESS) {
    return expr;
  }

  int dim = isl_space_dim(domain, isl_dim_set);
  int n_in = dim + arg_num;

  isl_space* space = PypetExprAccessGetParameterSpace(expr);
  int n = PypetNestedNInSpace(space);
  isl_space_free(space);
  if (n == 0) {
    return expr;
  }

  expr = PypetExprAccessAlignParams(expr);

  space = PypetExprAccessGetParameterSpace(expr);
  int param_num = isl_space_dim(space, isl_dim_param);
  isl_space_free(space);

  isl_ctx* ctx = PypetExprGetCtx(expr);

  int* param2pos = isl_alloc_array(ctx, int, param_num);
  int* t2pos = isl_alloc_array(ctx, int, n);
  CHECK(param2pos);
  CHECK(t2pos);

  expr = PypetExprExtractNested(expr, n, param2pos);
  expr = MarkSelfDependences(expr, arg_num);

  n = 0;
  space = PypetExprAccessGetParameterSpace(expr);
  param_num = isl_space_dim(space, isl_dim_param);
  for (int i = param_num - 1; i >= 0; --i) {
    isl_id* id = isl_space_get_dim_id(space, isl_dim_param, i);
    if (!PypetNestedInId(id)) {
      isl_id_free(id);
      continue;
    }
    expr = PypetExprAccessMoveDims(expr, isl_dim_in, n_in + n, isl_dim_param, i,
                                   1);
    t2pos[n] = n_in + param2pos[i];
    ++n;
    isl_id_free(id);
  }
  isl_space_free(space);

  space = isl_space_copy(domain);
  space = isl_space_from_domain(space);
  space = isl_space_add_dims(space, isl_dim_out, expr->arg_num);
  space = isl_space_wrap(space);
  isl_local_space* ls = isl_local_space_from_space(isl_space_copy(space));
  space = isl_space_from_domain(space);
  space = isl_space_add_dims(space, isl_dim_out, n_in + n);
  isl_multi_aff* ma = isl_multi_aff_zero(space);

  for (int i = 0; i < n_in; ++i) {
    isl_aff* aff =
        isl_aff_var_on_domain(isl_local_space_copy(ls), isl_dim_set, i);
    ma = isl_multi_aff_set_aff(ma, i, aff);
  }
  for (int i = 0; i < n; ++i) {
    isl_aff* aff =
        isl_aff_var_on_domain(isl_local_space_copy(ls), isl_dim_set, t2pos[i]);
    ma = isl_multi_aff_set_aff(ma, n_in + i, aff);
  }
  isl_local_space_free(ls);

  expr = PypetExprAccessPullbackMultiAff(expr, ma);

  expr = RemoveMarkedSelfDependences(expr, dim, arg_num);

  free(t2pos);
  free(param2pos);
  return expr;
}

PypetTree* PypetTreeResolveNested(PypetTree* tree, isl_space* space) {
  return PypetTreeMapExpr(tree, PypetTreeResolveNestedFunc, space);
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

PypetExpr* MarkSelfDependences(PypetExpr* expr, int first) {
  if (expr->acc.write == 1) {
    return expr;
  }
  int n = expr->arg_num;
  for (int i = first; i < n; ++i) {
    PypetExpr* arg = PypetExprGetArg(expr, i);
    bool mark = PypetExprIsSubAccess(expr, arg, first);
    PypetExprFree(arg);
    if (!mark) {
      continue;
    }
    arg = PypetExprFromIslVal(isl_val_nan(PypetExprGetCtx(expr)));
    expr = PypetExprSetArg(expr, i, arg);
  }
  return expr;
}

PypetExpr* RemoveMarkedSelfDependences(PypetExpr* expr, int dim, int first) {
  for (int i = expr->arg_num - 1; i >= first; --i) {
    PypetExpr* arg = PypetExprGetArg(expr, i);
    bool is_nan = ExprIsNan(arg);
    PypetExprFree(arg);
    if (!is_nan) {
      continue;
    }
    expr = expr->AccessProjectOutArg(dim, i);
  }
  return expr;
}

}  // namespace pypet
}  // namespace pypoly
