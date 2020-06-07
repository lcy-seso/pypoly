#include "pypoly/core/pypet/pypet.h"

#include "pypoly/core/pypet/aff.h"
#include "pypoly/core/pypet/expr.h"
#include "pypoly/core/pypet/nest.h"
#include "pypoly/core/pypet/tree.h"

namespace pypoly {
namespace pypet {

namespace {

isl_set* DropArguments(isl_set* domain) {
  if (isl_set_is_wrapping(domain)) {
    domain = isl_map_domain(isl_set_unwrap(domain));
  }
  return domain;
}

isl_set* AccessExtractContext(PypetExpr* expr, isl_set* context) {
  isl_multi_pw_aff* mpa = isl_multi_pw_aff_copy(expr->acc.index);
  isl_set* domain = DropArguments(isl_multi_pw_aff_domain(mpa));
  domain = isl_set_reset_tuple_id(domain);
  context = isl_set_intersect(context, domain);
  return context;
}

isl_set* ExprExtractContext(PypetExpr* expr, isl_set* context) {
  if (expr->type == PypetExprType::PYPET_EXPR_OP &&
      expr->op == PypetOpType::PYPET_COND) {
    bool is_aff = PypetExprIsAffine(expr->args[0]);
    context = ExprExtractContext(expr->args[0], context);
    isl_set* lhs_context =
        ExprExtractContext(expr->args[1], isl_set_copy(context));
    isl_set* rhs_context = ExprExtractContext(expr->args[2], context);

    if (is_aff) {
      isl_multi_pw_aff* mpa = isl_multi_pw_aff_copy(expr->args[0]->acc.index);
      isl_pw_aff* pa = isl_multi_pw_aff_get_pw_aff(mpa, 0);
      isl_multi_pw_aff_free(mpa);
      isl_set* zero_set = DropArguments(isl_pw_aff_zero_set(pa));
      zero_set = isl_set_reset_tuple_id(zero_set);
      lhs_context = isl_set_subtract(lhs_context, isl_set_copy(zero_set));
      rhs_context = isl_set_intersect(rhs_context, zero_set);
    }
    context = isl_set_union(lhs_context, rhs_context);
    context = isl_set_coalesce(context);
    return context;
  }

  for (int i = 0; i < expr->arg_num; ++i) {
    context = ExprExtractContext(expr->args[i], context);
  }

  if (expr->type == PypetExprType::PYPET_EXPR_ACCESS) {
    context = AccessExtractContext(expr, context);
  }
  return context;
}

}  // namespace

PypetStmt* PypetStmt::Create(isl_set* domain, int id, PypetTree* tree) {
  isl_ctx* ctx = tree->ctx;
  PypetStmt* stmt = isl_calloc_type(ctx, PypetStmt);

  isl_id* label = nullptr;
  if (tree->label) {
    label = isl_id_copy(tree->label);
  } else {
    char name[50];
    snprintf(name, sizeof(name), "S_%d", id);
    label = isl_id_alloc(ctx, name, nullptr);
  }

  domain = isl_set_set_tuple_id(domain, label);
  isl_space* space = isl_set_get_space(domain);
  space = PypetNestedRemoveFromSpace(space);
  isl_multi_aff* multi_aff =
      PypetPrefixProjection(space, isl_space_dim(space, isl_dim_set));

  isl_multi_pw_aff* add_name = isl_multi_pw_aff_from_multi_aff(multi_aff);
  tree = PypetTreeUpdateDomain(tree, add_name);

  stmt->range = *tree->range;
  stmt->domain = domain;
  stmt->body = tree;
  return stmt;
}

isl_set* StmtExtractContext(PypetStmt* stmt, isl_set* context) {
  // TODO(yizhu1): affine assume
  for (int i = 0; i < stmt->arg_num; ++i) {
    context = ExprExtractContext(stmt->args[i], context);
  }
  if (stmt->body->type != PypetTreeType::PYPET_TREE_EXPR) {
    return context;
  }
  PypetExpr* body = PypetExprCopy(stmt->body->ast.Expr.expr);
  context = ExprExtractContext(body, context);
  PypetExprFree(body);
  return context;
}

PypetScop* PypetScop::Create(isl_space* space) {
  isl_schedule* schedule = isl_schedule_empty(isl_space_copy(space));
  return Create(space, 0, schedule);
}

PypetScop* PypetScop::Create(isl_space* space, PypetStmt* stmt) {
  isl_set* set = PypetNestedRemoveFromSet(isl_set_copy(stmt->domain));
  isl_union_set* domain = isl_union_set_from_set(set);
  isl_schedule* schedule = isl_schedule_from_domain(domain);

  PypetScop* scop = PypetScop::Create(space, 1, schedule);
  scop->context = StmtExtractContext(stmt, scop->context);
  scop->stmts[0] = stmt;
  scop->range = stmt->range;
  return scop;
}

PypetScop* PypetScop::Create(isl_space* space, int n, isl_schedule* schedule) {
  CHECK(space);
  CHECK_GE(n, 0);
  CHECK(schedule);

  isl_ctx* ctx = isl_space_get_ctx(space);
  PypetScop* scop = isl_calloc_type(ctx, PypetScop);

  scop->context = isl_set_universe(isl_space_copy(space));
  scop->context_value = isl_set_universe(isl_space_params(space));
  scop->stmts = isl_calloc_array(ctx, PypetStmt*, n);
  scop->schedule = schedule;
  scop->stmt_num = n;
  scop->array_num = 0;
  scop->arrays = nullptr;

  return scop;
}

PypetScop* PypetScopRestrict(PypetScop* scop, isl_set* cond) {
  // TODO(yizhu1): pet_scop_restrict_skip

  scop->context = isl_set_intersect(scop->context, isl_set_copy(cond));
  scop->context =
      isl_set_union(scop->context, isl_set_complement(isl_set_copy(cond)));
  scop->context = isl_set_coalesce(scop->context);
  scop->context = PypetNestedRemoveFromSet(scop->context);

  isl_set_free(cond);
  return scop;
}

PypetScop* PypetScopRestrictContext(PypetScop* scop, isl_set* context) {
  context = PypetNestedRemoveFromSet(context);
  scop->context = isl_set_intersect(scop->context, context);
  return scop;
}

PypetScop* PypetScopAdd(isl_ctx* ctx, isl_schedule* schedule, PypetScop* lhs,
                        PypetScop* rhs) {
  if (lhs->stmt_num == 0) {
    return rhs;
  }

  if (rhs->stmt_num == 0) {
    return lhs;
  }

  isl_space* space = isl_set_get_space(lhs->context);
  PypetScop* ret = PypetScop::Create(space, lhs->stmt_num + rhs->stmt_num,
                                     isl_schedule_copy(schedule));
  ret->arrays =
      isl_calloc_array(ctx, PypetArray*, lhs->array_num + rhs->array_num);
  ret->array_num = lhs->array_num + rhs->array_num;

  for (int i = 0; i < lhs->stmt_num; ++i) {
    ret->stmts[i] = lhs->stmts[i];
    lhs->stmts[i] = nullptr;
  }

  for (int i = 0; i < rhs->stmt_num; ++i) {
    ret->stmts[i + lhs->stmt_num] = rhs->stmts[i];
    rhs->stmts[i] = nullptr;
  }

  for (int i = 0; i < lhs->array_num; ++i) {
    ret->arrays[i] = lhs->arrays[i];
    lhs->arrays[i] = nullptr;
  }

  for (int i = 0; i < rhs->array_num; ++i) {
    ret->arrays[i + lhs->array_num] = rhs->arrays[i];
    rhs->arrays[i] = nullptr;
  }

  ret = ScopCollectImplications(ctx, ret, lhs, rhs);
  ret = PypetScopRestrictContext(ret, isl_set_copy(lhs->context));
  ret = PypetScopRestrictContext(ret, isl_set_copy(rhs->context));
  ret = PypetScopCombineSkips(ret, lhs, rhs);
  ret = PypetScopCombineStartEnd(ret, lhs, rhs);
  ret = PypetScopCollectIndependence(ctx, ret, lhs, rhs);

  return ret;
}

PypetScop* PypetScopAddSeq(isl_ctx* ctx, PypetScop* lhs, PypetScop* rhs) {
  // TODO(yizhu1): break and continue
  isl_schedule* schedule = isl_schedule_sequence(
      isl_schedule_copy(lhs->schedule), isl_schedule_copy(rhs->schedule));
  return PypetScopAdd(ctx, schedule, lhs, rhs);
}

PypetScop* PypetScopAddPar(isl_ctx* ctx, PypetScop* lhs, PypetScop* rhs) {
  isl_schedule* schedule = isl_schedule_set(isl_schedule_copy(lhs->schedule),
                                            isl_schedule_copy(rhs->schedule));
  return PypetScopAdd(ctx, schedule, lhs, rhs);
}

}  // namespace pypet
}  // namespace pypoly
