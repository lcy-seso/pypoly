#include "pypoly/core/pypet/tree2scop.h"

#include "pypoly/core/pypet/isl_printer.h"
#include "pypoly/core/pypet/pypet.h"

namespace pypoly {
namespace pypet {

namespace {

isl_pw_aff* PypetExprExtractAffine(PypetExpr* expr, PypetContext* context);

int PypetContextDim(PypetContext* pc) {
  CHECK(pc);
  return isl_set_dim(pc->domain, isl_dim_set);
}

PypetContext* ContextAlloc(isl_set* domain, bool allow_nested) {
  PypetContext* context =
      isl_calloc_type(isl_set_get_ctx(domain), struct PypetContext);
  CHECK(context);

  context->ref = 1;
  context->domain = domain;
  context->allow_nested = allow_nested;
  return context;
}

PypetContext* PypetContextDup(PypetContext* context) {
  CHECK(context);
  PypetContext* dup =
      ContextAlloc(isl_set_copy(context->domain), context->allow_nested);
  dup->assignments = context->assignments;
  dup->extracted_affine = context->extracted_affine;
  return dup;
}

PypetContext* PypetContextCow(PypetContext* context) {
  CHECK(context);
  if (context->ref == 1) {
    // TODO(yizhu1): check pointers
    context->extracted_affine.clear();
    return context;
  }
  --context->ref;
  return PypetContextDup(context);
}

PypetContext* ExtendDomain(PypetContext* context, isl_id* id) {
  CHECK(context);
  context = PypetContextCow(context);
  CHECK(id);
  int pos = PypetContextDim(context);
  context->domain = isl_set_add_dims(context->domain, isl_dim_set, 1);
  context->domain = isl_set_set_dim_id(context->domain, isl_dim_set, pos, id);
  CHECK(context->domain);
  return context;
}

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

isl_space* PypetNestedRemoveFromSpace(isl_space* space) {
  int param_num = isl_space_dim(space, isl_dim_param);
  for (int i = param_num - 1; i >= 0; --i) {
    if (PypetNestedInSpace(space, i)) {
      space = isl_space_drop_dims(space, isl_dim_param, i, 1);
    }
  }
  return space;
}

isl_space* PypetContextGetSpace(PypetContext* context) {
  CHECK(context);
  isl_space* space = isl_set_get_space(context->domain);
  space = PypetNestedRemoveFromSpace(space);
  return space;
}

PypetContext* PypetContextSetValue(PypetContext* context, isl_id* id,
                                   isl_pw_aff* pw_aff) {
  context = PypetContextCow(context);
  CHECK(context);
  context->assignments[id] = pw_aff;
  return context;
}

int PypetContextGetDim(PypetContext* context) {
  CHECK(context);
  return isl_set_dim(context->domain, isl_dim_set);
}

PypetContext* PypetContextAddInnerIterator(PypetContext* context, isl_id* id) {
  CHECK(context);
  CHECK(id);
  int pos = PypetContextGetDim(context);
  context = ExtendDomain(context, isl_id_copy(id));
  CHECK(context);
  isl_space* space = PypetContextGetSpace(context);
  isl_local_space* local_space = isl_local_space_from_space(space);
  isl_aff* aff = isl_aff_var_on_domain(local_space, isl_dim_set, pos);
  isl_pw_aff* pw_aff = isl_pw_aff_from_aff(aff);
  context = PypetContextSetValue(context, id, pw_aff);
  return context;
}

PypetContext* PypetContextCopy(PypetContext* context) {
  CHECK(context);
  ++context->ref;
  return context;
}

PypetContext* PypetContextClearValue(PypetContext* context, isl_id* id) {
  CHECK(context);
  context = PypetContextCow(context);
  CHECK(context);
  context->assignments.erase(id);
  CHECK(context);
  return context;
}

int ClearWrite(PypetExpr* expr, void* user) {
  PypetContext** context = static_cast<PypetContext**>(user);
  if (expr->acc.write == 0) {
    return 0;
  }
  if (PypetExprIsScalarAccess(expr) == 0) {
    return 0;
  }

  isl_id* id = PypetExprAccessGetId(expr);
  if (isl_id_get_user(id)) {
    *context = PypetContextClearValue(*context, id);
  } else {
    isl_id_free(id);
  }
  return 0;
}

PypetContext* PypetContextClearWritesInTree(PypetContext* context,
                                            PypetTree* tree) {
  CHECK_GE(PypetTreeForeachAccessExpr(tree, &ClearWrite, &context), 0);
  return context;
}

bool PypetContextIsAssigned(PypetContext* context, isl_id* id) {
  return context->assignments.find(id) != context->assignments.end();
}

isl_multi_aff* PypetPrefixProjection(isl_space* space, int n) {
  int dim = isl_space_dim(space, isl_dim_set);
  return isl_multi_aff_project_out_map(space, isl_dim_set, n, dim - n);
}

isl_pw_aff* PypetContextGetValue(PypetContext* context, isl_id* id) {
  CHECK(context);
  CHECK(id);
  isl_pw_aff* pa = context->assignments[id];
  int dim = isl_pw_aff_dim(pa, isl_dim_in);
  if (dim == isl_set_dim(context->domain, isl_dim_set)) {
    return pa;
  }
  isl_multi_aff* ma = PypetPrefixProjection(PypetContextGetSpace(context), dim);
  pa = isl_pw_aff_pullback_multi_aff(pa, ma);
  return pa;
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

isl_pw_aff* NestedAccess(PypetExpr* expr, PypetContext* context) {
  CHECK(expr);
  CHECK(context);
  CHECK(context->allow_nested);
  CHECK(expr->type == PypetExprType::PYPET_EXPR_ACCESS);
  if (expr->arg_num > 0) {
    return NonAffine(PypetContextGetSpace(context));
  }
  isl_space* space = PypetExprAccessGetParameterSpace(expr);
  bool nested = PypetNestedAnyInSpace(space);
  isl_space_free(space);
  if (nested) {
    return NonAffine(PypetContextGetSpace(context));
  }
  isl_id* id = PypetNestedPypetExpr(PypetExprCopy(expr));
  space = PypetContextGetSpace(context);
  space = isl_space_insert_dims(space, isl_dim_param, 0, 1);

  space = isl_space_set_dim_id(space, isl_dim_param, 0, id);
  isl_local_space* ls = isl_local_space_from_space(space);
  isl_aff* aff = isl_aff_var_on_domain(ls, isl_dim_param, 0);
  return isl_pw_aff_from_aff(aff);
}

isl_pw_aff* ExtractAffineFromAccess(PypetExpr* expr, PypetContext* context) {
  if (PypetExprIsAffine(expr)) {
    return PypetExprGetAffine(expr);
  }
  CHECK_NE(expr->type_size, 0) << std::endl << expr;
  if (!PypetExprIsScalarAccess(expr)) {
    return NestedAccess(expr, context);
  }
  isl_id* id = PypetExprAccessGetId(expr);
  if (PypetContextIsAssigned(context, id)) {
    return PypetContextGetValue(context, id);
  }
  isl_id_free(id);
  return NestedAccess(expr, context);
}

isl_pw_aff* ExtractAffineFromInt(PypetExpr* expr, PypetContext* context) {
  CHECK(expr);
  isl_local_space* local_space =
      isl_local_space_from_space(PypetContextGetSpace(context));
  isl_aff* aff = isl_aff_val_on_domain(local_space, isl_val_copy(expr->i));
  return isl_pw_aff_from_aff(aff);
}

isl_pw_aff* ExtractAffineAddSub(PypetExpr* expr, PypetContext* context) {
  CHECK(expr);
  CHECK_EQ(expr->arg_num, 2);
  isl_pw_aff* lhs = PypetExprExtractAffine(expr->args[0], context);
  isl_pw_aff* rhs = PypetExprExtractAffine(expr->args[1], context);
  switch (expr->op) {
    case PypetOpType::PYPET_ADD:
      return isl_pw_aff_add(lhs, rhs);
    case PypetOpType::PYPET_SUB:
      return isl_pw_aff_sub(lhs, rhs);
    default:
      UNIMPLEMENTED();
      break;
  }
  return nullptr;
}

isl_pw_aff* ExtractAffineDivMod(PypetExpr* expr, PypetContext* context) {
  UNIMPLEMENTED();
  return nullptr;
}

isl_pw_aff* ExtractAffineMul(PypetExpr* expr, PypetContext* context) {
  UNIMPLEMENTED();
  return nullptr;
}

isl_pw_aff* ExtractAffineNeg(PypetExpr* expr, PypetContext* context) {
  UNIMPLEMENTED();
  return nullptr;
}

isl_pw_aff* ExtractAffineCond(PypetExpr* expr, PypetContext* context) {
  UNIMPLEMENTED();
  return nullptr;
}

isl_pw_aff* IndicatorFunction(isl_set* set, isl_set* dom) {
  isl_pw_aff* pw_aff = isl_set_indicator_function(set);
  pw_aff = isl_pw_aff_intersect_domain(pw_aff, isl_set_coalesce(dom));
  return pw_aff;
}

isl_pw_aff* PypetAnd(isl_pw_aff* lhs, isl_pw_aff* rhs) {
  isl_set* dom = isl_set_intersect(isl_pw_aff_domain(isl_pw_aff_copy(lhs)),
                                   isl_pw_aff_domain(isl_pw_aff_copy(rhs)));
  isl_set* cond = isl_set_intersect(isl_pw_aff_non_zero_set(lhs),
                                    isl_pw_aff_non_zero_set(rhs));
  return IndicatorFunction(cond, dom);
}

isl_pw_aff* PypetComparison(PypetOpType type, isl_pw_aff* lhs,
                            isl_pw_aff* rhs) {
  CHECK(lhs);
  CHECK(rhs);
  if (isl_pw_aff_involves_nan(lhs) || isl_pw_aff_involves_nan(rhs)) {
    LOG(FATAL) << "unexpected input";
    return nullptr;
  }
  isl_set* dom = isl_set_intersect(isl_pw_aff_domain(isl_pw_aff_copy(lhs)),
                                   isl_pw_aff_domain(isl_pw_aff_copy(rhs)));
  isl_set* cond = nullptr;
  switch (type) {
    case PypetOpType::PYPET_LT:
      cond = isl_pw_aff_lt_set(lhs, rhs);
      break;
    case PypetOpType::PYPET_LE:
      cond = isl_pw_aff_le_set(lhs, rhs);
      break;
    case PypetOpType::PYPET_GT:
      cond = isl_pw_aff_gt_set(lhs, rhs);
      break;
    case PypetOpType::PYPET_GE:
      cond = isl_pw_aff_ge_set(lhs, rhs);
      break;
    case PypetOpType::PYPET_EQ:
      cond = isl_pw_aff_eq_set(lhs, rhs);
      break;
    case PypetOpType::PYPET_NE:
      cond = isl_pw_aff_ne_set(lhs, rhs);
      break;
    default:
      break;
  }
  cond = isl_set_coalesce(cond);
  return IndicatorFunction(cond, dom);
}

isl_pw_aff* PypetExprExtractComparison(PypetOpType type, PypetExpr* lhs,
                                       PypetExpr* rhs, PypetContext* context) {
  if (type == PypetOpType::PYPET_GT) {
    return PypetExprExtractComparison(PypetOpType::PYPET_LT, rhs, lhs, context);
  }
  if (type == PypetOpType::PYPET_GE) {
    return PypetExprExtractComparison(PypetOpType::PYPET_LE, rhs, lhs, context);
  }
  if (type == PypetOpType::PYPET_LT || type == PypetOpType::PYPET_LT) {
    if (rhs->IsMin()) {
      return PypetAnd(
          PypetExprExtractComparison(type, lhs, rhs->args[1], context),
          PypetExprExtractComparison(type, lhs, rhs->args[2], context));
    }
    if (lhs->IsMax()) {
      return PypetAnd(
          PypetExprExtractComparison(type, lhs->args[1], rhs, context),
          PypetExprExtractComparison(type, lhs->args[2], rhs, context));
    }
  }
  return PypetComparison(type, PypetExprExtractAffine(lhs, context),
                         PypetExprExtractAffine(rhs, context));
}

isl_pw_aff* ExtractComparison(PypetExpr* expr, PypetContext* context) {
  CHECK(expr);
  CHECK_EQ(expr->arg_num, 2);
  return PypetExprExtractComparison(expr->op, expr->args[0], expr->args[1],
                                    context);
}

isl_pw_aff* ExtractBoolean(PypetExpr* expr, PypetContext* context) {
  UNIMPLEMENTED();
  return nullptr;
}

isl_pw_aff* ExtractImplicitCondition(PypetExpr* expr, PypetContext* context) {
  UNIMPLEMENTED();
  return nullptr;
}

isl_pw_aff* PypetExprExtractAffineCondition(PypetExpr* expr,
                                            PypetContext* context) {
  CHECK(expr);
  if (expr->IsComparison()) {
    return ExtractComparison(expr, context);
  } else if (expr->IsBoolean()) {
    return ExtractBoolean(expr, context);
  } else {
    return ExtractImplicitCondition(expr, context);
  }
}

isl_val* WrapMod(isl_ctx* ctx, unsigned width) {
  return isl_val_2exp(isl_val_int_from_ui(ctx, width));
}

isl_pw_aff* PypetWrapPwAff(isl_pw_aff* pw_aff, unsigned width) {
  isl_val* mod = WrapMod(isl_pw_aff_get_ctx(pw_aff), width);
  return isl_pw_aff_mod_val(pw_aff, mod);
}

isl_pw_aff* AvoidOverflow(isl_pw_aff* pw_aff, unsigned width) {
  isl_space* space = isl_pw_aff_get_domain_space(pw_aff);
  isl_local_space* local_space = isl_local_space_from_space(space);

  isl_ctx* ctx = isl_pw_aff_get_ctx(pw_aff);
  isl_val* val = isl_val_int_from_ui(ctx, width - 1);
  val = isl_val_2exp(val);

  isl_aff* bound = isl_aff_zero_on_domain(local_space);
  bound = isl_aff_add_constant_val(bound, val);
  isl_pw_aff* b = isl_pw_aff_from_aff(bound);

  isl_set* dom = isl_pw_aff_lt_set(isl_pw_aff_copy(pw_aff), isl_pw_aff_copy(b));
  pw_aff = isl_pw_aff_intersect_domain(pw_aff, dom);

  b = isl_pw_aff_neg(b);
  dom = isl_pw_aff_ge_set(isl_pw_aff_copy(pw_aff), b);
  pw_aff = isl_pw_aff_intersect_domain(pw_aff, dom);
  return pw_aff;
}

isl_pw_aff* SignedOverflow(isl_pw_aff* pw_aff, unsigned width) {
  CHECK(pw_aff);
  return AvoidOverflow(pw_aff, width);
}

isl_pw_aff* ExtractAffineFromOp(PypetExpr* expr, PypetContext* context) {
  isl_pw_aff* ret = nullptr;
  switch (expr->op) {
    case PypetOpType::PYPET_ADD:
    case PypetOpType::PYPET_SUB:
      ret = ExtractAffineAddSub(expr, context);
      break;
    case PypetOpType::PYPET_DIV:
    case PypetOpType::PYPET_MOD:
      ret = ExtractAffineDivMod(expr, context);
      break;
    case PypetOpType::PYPET_MUL:
      ret = ExtractAffineMul(expr, context);
      break;
    case PypetOpType::PYPET_MINUS:
      return ExtractAffineNeg(expr, context);
    case PypetOpType::PYPET_COND:
      return ExtractAffineCond(expr, context);
    case PypetOpType::PYPET_EQ:
    case PypetOpType::PYPET_NE:
    case PypetOpType::PYPET_LE:
    case PypetOpType::PYPET_GE:
    case PypetOpType::PYPET_LT:
    case PypetOpType::PYPET_GT:
      return PypetExprExtractAffineCondition(expr, context);
    default:
      UNIMPLEMENTED();
      break;
  }
  CHECK(ret);
  CHECK(isl_pw_aff_involves_nan(ret) == 0);
  if (expr->type_size > 0) {
    ret = PypetWrapPwAff(ret, expr->type_size);
  } else {
    ret = SignedOverflow(ret, -expr->type_size);
  }
  return ret;
}

isl_pw_aff* PypetExprExtractAffine(PypetExpr* expr, PypetContext* context) {
  CHECK(expr);
  auto iter = context->extracted_affine.find(expr);
  if (iter != context->extracted_affine.end()) {
    return iter->second;
  }

  isl_pw_aff* pw_aff = nullptr;
  switch (expr->type) {
    case PypetExprType::PYPET_EXPR_ACCESS:
      pw_aff = ExtractAffineFromAccess(expr, context);
      break;
    case PypetExprType::PYPET_EXPR_INT:
      pw_aff = ExtractAffineFromInt(expr, context);
      break;
    case PypetExprType::PYPET_EXPR_OP:
      pw_aff = ExtractAffineFromOp(expr, context);
      break;
    case PypetExprType::PYPET_EXPR_CALL:
    default:
      UNIMPLEMENTED();
      break;
  }

  context->extracted_affine.insert(std::make_pair(expr, pw_aff));
  return pw_aff;
}

isl_stat ExtractCst(isl_set* set, isl_aff* aff, void* user) {
  isl_val** inc = static_cast<isl_val**>(user);
  if (isl_aff_is_cst(aff)) {
    isl_val_free(*inc);
    *inc = isl_aff_get_constant_val(aff);
  }
  isl_set_free(set);
  isl_aff_free(aff);
  return isl_stat_ok;
}

isl_val* PypetExtractCst(isl_pw_aff* pw_aff) {
  CHECK(pw_aff);
  isl_val* val = isl_val_nan(isl_pw_aff_get_ctx(pw_aff));
  if (isl_pw_aff_n_piece(pw_aff) != 1) {
    return val;
  }
  CHECK_GE(isl_pw_aff_foreach_piece(pw_aff, &ExtractCst, &val), 0);
  return val;
}

isl_set* PypetContextGetDomain(PypetContext* context) {
  CHECK(context);
  return isl_set_copy(context->domain);
}

PypetExpr* AccessPlugInAffineRead(PypetExpr* expr, void* user) {
  CHECK(expr->type == PypetExprType::PYPET_EXPR_ACCESS);
  PypetContext* context = static_cast<PypetContext*>(user);
  if (expr->acc.write) {
    return expr;
  }
  if (!PypetExprIsScalarAccess(expr)) {
    return expr;
  }

  isl_pw_aff* pw_aff = PypetExprExtractAffine(expr, context);
  PypetExprFree(expr);
  return PypetExprFromIndex(isl_multi_pw_aff_from_pw_aff(pw_aff));
}

PypetExpr* PlugInAffineRead(PypetExpr* expr, PypetContext* context) {
  return PypetExprMapExprOfType(expr, PypetExprType::PYPET_EXPR_ACCESS,
                                AccessPlugInAffineRead, context);
}

PypetExpr* SpliceSum(PypetExpr* expr, int dim, int pos) {
  CHECK(expr);
  PypetExpr* arg = expr->args[pos];
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

PypetExpr* PlugInArgs(PypetExpr* expr, void* user) {
  PypetContext* context = static_cast<PypetContext*>(user);
  return PypetExprAccessPlugInArgs(expr, context);
}

PypetExpr* PypetExprPlugInArgs(PypetExpr* expr, PypetContext* context) {
  return PypetExprMapExprOfType(expr, PypetExprType::PYPET_EXPR_ACCESS,
                                PlugInArgs, context);
}

int CheckOnlyAffine(PypetExpr* expr, void* user) {
  int* only_affine = static_cast<int*>(user);
  if (!PypetExprIsAffine(expr)) {
    *only_affine = 0;
    return -1;
  } else {
    *only_affine = 1;
    return 0;
  }
}

bool HasOnlyAffineAccessSubExpr(PypetExpr* expr) {
  int only_affine = -1;
  PypetExprForeachAccessExpr(expr, &CheckOnlyAffine, &only_affine);
  return only_affine > 0;
}

PypetExpr* ExprPlugInAffine(PypetExpr* expr, void* user) {
  CHECK(expr);
  PypetExprType type = expr->type;
  PypetContext* context = static_cast<PypetContext*>(user);
  if (type != PypetExprType::PYPET_EXPR_CALL &&
      type != PypetExprType::PYPET_EXPR_OP) {
    return expr;
  }
  bool contains_access = HasOnlyAffineAccessSubExpr(expr);
  if (!contains_access) {
    return expr;
  }

  isl_pw_aff* pw_aff = PypetExprExtractAffine(expr, context);
  CHECK_EQ(isl_pw_aff_involves_nan(pw_aff), 0);

  PypetExprFree(expr);
  expr = PypetExprFromIndex(isl_multi_pw_aff_from_pw_aff(pw_aff));
  return expr;
}

PypetExpr* PlugInAffine(PypetExpr* expr, PypetContext* context) {
  return PypetExprMapTopDown(expr, ExprPlugInAffine, context);
}

PypetExpr* MergeConditionalAccesses(PypetExpr* expr) {
  // TODO(yizhu1): PYPET_COND
  return expr;
}

PypetExpr* PlugInSummaries(PypetExpr* expr, PypetContext* context) {
  // TODO(yizhu1): function summaries
  return expr;
}

PypetExpr* PypetContextEvaluateExpr(PypetContext* context, PypetExpr* expr) {
  LOG(INFO) << expr;
  expr = PypetExprInsertDomain(expr, PypetContextGetSpace(context));
  LOG(INFO) << expr;
  expr = PlugInAffineRead(expr, context);
  LOG(INFO) << expr;
  expr = PypetExprPlugInArgs(expr, context);
  LOG(INFO) << expr;
  expr = PlugInAffine(expr, context);
  LOG(INFO) << expr;
  expr = MergeConditionalAccesses(expr);
  expr = PlugInSummaries(expr, context);
  return expr;
}

PypetContext* PypetContextSetAllowNested(PypetContext* context, bool val) {
  CHECK(context);
  if (context->allow_nested == val) {
    return context;
  }
  context = PypetContextCow(context);
  CHECK(context);
  context->allow_nested = val;
  return context;
}

int IsAssigned(PypetExpr* expr, PypetTree* tree) {
  isl_id* id = PypetExprAccessGetId(expr);
  int assigned = PypetTreeWrites(tree, id);
  isl_id_free(id);
  return assigned;
}

bool PypetNestedAnyInPwAff(isl_pw_aff* pa) {
  isl_space* space = isl_pw_aff_get_space(pa);
  bool nested = PypetNestedAnyInSpace(space);
  isl_space_free(space);
  return nested;
}

PypetExpr* PypetNestedExtractExpr(isl_id* id) {
  return PypetExprCopy(static_cast<PypetExpr*>(isl_id_get_user(id)));
}

bool IsNestedAllowed(isl_pw_aff* pa, PypetTree* tree) {
  CHECK(tree);
  if (!PypetNestedAnyInPwAff(pa)) {
    return true;
  }
  if (PypetTreeHasContinueOrBreak(tree)) {
    return false;
  }
  int param_num = isl_pw_aff_dim(pa, isl_dim_param);
  for (int i = 0; i < param_num; ++i) {
    isl_id* id = isl_pw_aff_get_dim_id(pa, isl_dim_param, i);
    if (!PypetNestedExtractExpr(id)) {
      isl_id_free(id);
      continue;
    }
    PypetExpr* expr = PypetNestedExtractExpr(id);
    bool allowed = expr->type == PypetExprType::PYPET_EXPR_ACCESS &&
                   !IsAssigned(expr, tree);
    PypetExprFree(expr);
    isl_id_free(id);
    if (!allowed) {
      return false;
    }
  }
  return true;
}

bool CanWrap(isl_set* cond, PypetExpr* iv, isl_val* inc) {
  isl_set* test = isl_set_copy(cond);
  isl_ctx* ctx = isl_set_get_ctx(test);
  isl_val* limit = nullptr;
  if (isl_val_is_neg(inc)) {
    limit = isl_val_zero(ctx);
  } else {
    limit = isl_val_int_from_ui(ctx, iv->type_size);
    limit = isl_val_2exp(limit);
    limit = isl_val_sub_ui(limit, 1);
  }
  test = isl_set_fix_val(cond, isl_dim_set, 0, limit);
  bool ret = !isl_set_is_empty(test);
  isl_set_free(test);
  return ret;
}

isl_set* EnforceSubset(isl_set* lhs, isl_set* rhs) {
  int pos = isl_set_dim(lhs, isl_dim_set) - 1;
  lhs = isl_set_subtract(lhs, rhs);
  lhs = isl_set_eliminate(lhs, isl_dim_set, pos, 1);
  return isl_set_complement(lhs);
}

isl_set* StridedDomain(isl_pw_aff* init, isl_val* inc) {
  int dim = isl_pw_aff_dim(init, isl_dim_in);

  init = isl_pw_aff_add_dims(init, isl_dim_in, 1);
  isl_space* space = isl_pw_aff_get_domain_space(init);
  isl_local_space* ls = isl_local_space_from_space(space);
  isl_aff* aff = isl_aff_zero_on_domain(isl_local_space_copy(ls));
  aff = isl_aff_add_coefficient_val(aff, isl_dim_in, dim, inc);
  init = isl_pw_aff_add(init, isl_pw_aff_from_aff(aff));

  aff = isl_aff_var_on_domain(ls, isl_dim_set, dim - 1);
  isl_set* set = isl_pw_aff_eq_set(isl_pw_aff_from_aff(aff), init);

  set = isl_set_lower_bound_si(set, isl_dim_set, dim, 0);
  set = isl_set_project_out(set, isl_dim_set, dim, 1);
  return set;
}

isl_multi_aff* MapToLast(PypetContext* pc, int loop_nr, isl_id* id) {
  char name[50];
  isl_space* space = PypetContextGetSpace(pc);
  int pos = isl_space_dim(space, isl_dim_set) - 1;
  isl_local_space* ls = isl_local_space_from_space(space);
  isl_aff* aff = isl_aff_var_on_domain(ls, isl_dim_set, pos);
  isl_multi_aff* multi_aff = isl_multi_aff_from_aff(aff);

  isl_id* label = nullptr;
  if (id) {
    label = isl_id_copy(id);
  } else {
    snprintf(name, sizeof(name), "L_%d", loop_nr);
    label = isl_id_alloc(isl_set_get_ctx(pc->domain), name, nullptr);
  }
  multi_aff = isl_multi_aff_set_tuple_id(multi_aff, isl_dim_out, label);
  return multi_aff;
}

isl_set* ValidOnNext(isl_set* cond, isl_set* dom, isl_val* inc) {
  int pos = isl_set_dim(dom, isl_dim_set) - 1;
  isl_space* space = isl_set_get_space(dom);
  space = isl_space_map_from_set(space);
  isl_multi_aff* multi_aff = isl_multi_aff_identity(space);
  isl_aff* aff = isl_multi_aff_get_aff(multi_aff, pos);
  aff = isl_aff_add_constant_val(aff, inc);
  multi_aff = isl_multi_aff_set_aff(multi_aff, pos, aff);
  cond = isl_set_preimage_multi_aff(cond, multi_aff);
  return EnforceSubset(dom, cond);
}

PypetContext* PypetContextIntersectDomain(PypetContext* pc, isl_set* domain) {
  CHECK(pc);
  pc = PypetContextCow(pc);
  pc->domain = isl_set_intersect(pc->domain, domain);
  CHECK(pc->domain);
  return pc;
}

}  // namespace

/*
 * create PypetContext from a given domain.
 */
__isl_give PypetContext* CreatePypetContext(__isl_take isl_set* domain) {
  if (!domain) return nullptr;

  PypetContext* pc =
      isl_calloc_type(isl_set_get_ctx(domain), struct PypetContext);
  if (!pc) {
    isl_set_free(domain);
    return nullptr;
  }

  pc->ref = 1;
  pc->domain = domain;
  pc->assignments.clear();
  pc->allow_nested = true;
  pc->extracted_affine.clear();
  return pc;
}

/*
 * Free a reference to "pc" and return nullptr.
 */
__isl_null PypetContext* FreePypetContext(__isl_take PypetContext* pc) {
  if (!pc) return nullptr;
  if (--pc->ref > 0) return nullptr;

  isl_set_free(pc->domain);
  free(pc);
  return nullptr;
}

int AddParameter(PypetExpr* expr, void* user) {
  PypetContext* context = static_cast<PypetContext*>(user);
  if (!PypetExprIsScalarAccess(expr)) {
    // TODO(yizhu): get_array_size
    return 0;
  }
  if (!expr->acc.read) {
    return 0;
  }
  if (expr->type_size == 0) {
    return 0;
  }

  isl_id* id = PypetExprAccessGetId(expr);
  if (PypetContextIsAssigned(context, id)) {
    isl_id_free(id);
    return 0;
  }

  isl_space* space = PypetContextGetSpace(context);
  int pos = isl_space_find_dim_by_id(space, isl_dim_param, id);
  if (pos < 0) {
    pos = isl_space_dim(space, isl_dim_param);
    space = isl_space_add_dims(space, isl_dim_param, 1);
    space = isl_space_set_dim_id(space, isl_dim_param, pos, isl_id_copy(id));
  }

  isl_local_space* ls = isl_local_space_from_space(space);
  isl_aff* aff = isl_aff_var_on_domain(ls, isl_dim_param, pos);
  isl_pw_aff* pa = isl_pw_aff_from_aff(aff);
  context = PypetContextSetValue(context, id, pa);
  return 0;
}

/*
 * Add an assignment to "pc" for each parameter in "tree".
 */
__isl_give PypetContext* PypetContextAddParameter(__isl_keep PypetTree* tree,
                                                  __isl_keep PypetContext* pc) {
  CHECK_GE(PypetTreeForeachAccessExpr(tree, AddParameter, pc), 0);
  pc = PypetContextClearWritesInTree(pc, tree);
  return pc;
}

__isl_keep PypetScop* TreeToScop::ScopFromBlock(__isl_keep PypetTree* tree,
                                                __isl_keep PypetContext* pc,
                                                __isl_take PypetState* state) {
  return nullptr;
}

__isl_keep PypetScop* TreeToScop::ScopFromBreak(__isl_keep PypetTree* tree,
                                                __isl_keep PypetContext* pc,
                                                __isl_take PypetState* state) {
  return nullptr;
}

__isl_keep PypetScop* TreeToScop::ScopFromContinue(
    __isl_keep PypetTree* tree, __isl_keep PypetContext* pc,
    __isl_take PypetState* state) {
  return nullptr;
}

__isl_keep PypetScop* TreeToScop::ScopFromDecl(__isl_keep PypetTree* tree,
                                               __isl_keep PypetContext* pc,
                                               __isl_take PypetState* state) {
  return nullptr;
}

__isl_keep PypetScop* TreeToScop::ScopFromTreeExpr(
    __isl_keep PypetTree* tree, __isl_keep PypetContext* pc,
    __isl_take PypetState* state) {
  return nullptr;
}

__isl_keep PypetScop* TreeToScop::ScopFromReturn(__isl_keep PypetTree* tree,
                                                 __isl_keep PypetContext* pc,
                                                 __isl_take PypetState* state) {
  return nullptr;
}

__isl_keep PypetScop* TreeToScop::ScopFromIf(__isl_keep PypetTree* tree,
                                             __isl_keep PypetContext* pc,
                                             __isl_take PypetState* state) {
  return nullptr;
}

__isl_keep PypetScop* TreeToScop::ScopFromAffineFor(
    __isl_keep PypetTree* tree, __isl_take isl_pw_aff* init_val,
    __isl_take isl_pw_aff* pa_inc, __isl_take isl_val* inc,
    __isl_take PypetContext* pc, __isl_take PypetState* state) {
  int pos = PypetContextDim(pc) - 1;
  isl_set* domain = PypetContextGetDomain(pc);
  PypetExpr* cond_expr = PypetExprCopy(tree->ast.Loop.cond);
  cond_expr = PypetContextEvaluateExpr(pc, cond_expr);
  PypetContext* pc_nested = PypetContextCopy(pc);
  pc_nested = PypetContextSetAllowNested(pc_nested, true);

  isl_pw_aff* pa = PypetExprExtractAffineCondition(cond_expr, pc_nested);
  FreePypetContext(pc_nested);
  PypetExprFree(cond_expr);

  isl_set* valid_inc = isl_pw_aff_domain(pa_inc);

  bool is_unsigned = tree->ast.Loop.iv->type_size > 0;
  CHECK(!is_unsigned);
  bool is_non_affine =
      isl_pw_aff_involves_nan(pa) || !IsNestedAllowed(pa, tree->ast.Loop.body);
  CHECK(!is_non_affine);

  isl_set* valid_cond = isl_pw_aff_domain(isl_pw_aff_copy(pa));
  isl_set* cond = isl_pw_aff_non_zero_set(pa);
  valid_cond = isl_set_coalesce(valid_cond);
  bool is_one = isl_val_is_one(inc) || isl_val_is_negone(inc);
  bool is_virtual =
      is_unsigned && (!is_one || CanWrap(cond, tree->ast.Loop.iv, inc));
  CHECK(!is_virtual);

  isl_map* init_val_map = isl_map_from_pw_aff(isl_pw_aff_copy(init_val));
  init_val_map = isl_map_equate(init_val_map, isl_dim_in, pos, isl_dim_out, 0);
  isl_set* valid_cond_init =
      EnforceSubset(isl_map_domain(init_val_map), isl_set_copy(valid_cond));

  isl_set* valid_init = nullptr;
  if (is_one) {
    isl_pw_aff_free(init_val);
    pa = PypetExprExtractComparison(isl_val_is_pos(inc) ? PYPET_GE : PYPET_LE,
                                    tree->ast.Loop.iv, tree->ast.Loop.init, pc);
    valid_init = isl_pw_aff_domain(isl_pw_aff_copy(pa));
    valid_init = isl_set_eliminate(valid_init, isl_dim_set,
                                   isl_set_dim(domain, isl_dim_set) - 1, 1);
    cond = isl_pw_aff_non_zero_set(pa);
    domain = isl_set_intersect(domain, cond);
  } else {
    valid_init = isl_pw_aff_domain(isl_pw_aff_copy(init_val));
    isl_set* strided = StridedDomain(init_val, isl_val_copy(inc));
    domain = isl_set_intersect(domain, strided);
  }

  cond = isl_set_align_params(cond, isl_set_get_space(domain));
  domain = isl_set_intersect(domain, cond);

  isl_multi_aff* sched = MapToLast(pc, state->loop_num++, tree->label);
  if (isl_val_is_neg(inc)) {
    sched = isl_multi_aff_neg(sched);
  }

  isl_set* valid_cond_next =
      ValidOnNext(valid_cond, isl_set_copy(domain), isl_val_copy(inc));
  valid_inc = EnforceSubset(isl_set_copy(domain), valid_inc);

  pc = PypetContextIntersectDomain(pc, isl_set_copy(domain));

  PypetScop* scop = ToScop(tree->ast.Loop.body, pc, state);
  scop->ResetSkips();
  scop->ResolveNested();
  scop->SetIndependence(tree, domain, isl_val_sgn(inc), pc, state);

  valid_inc = isl_set_intersect(valid_inc, valid_cond_next);
  valid_inc = isl_set_intersect(valid_inc, valid_cond_init);
  valid_inc = isl_set_project_out(valid_inc, isl_dim_set, pos, 1);
  scop->RestrictContext(valid_inc);

  isl_val_free(inc);
  valid_init = isl_set_project_out(valid_init, isl_dim_set, pos, 1);
  scop->RestrictContext(valid_init);
  FreePypetContext(pc);
  return scop;
}

__isl_keep PypetScop* TreeToScop::ScopFromAffineForInit(
    __isl_keep PypetTree* tree, __isl_take isl_pw_aff* init_val,
    __isl_take isl_pw_aff* pa_inc, __isl_take isl_val* inc,
    __isl_keep PypetContext* pc_init, __isl_take PypetContext* pc,
    __isl_take PypetState* state) {
  CHECK_EQ(tree->ast.Loop.declared, 1);
  return ScopFromAffineFor(tree, init_val, pa_inc, inc, pc, state);
}

__isl_keep PypetScop* TreeToScop::ScopFromFor(__isl_keep PypetTree* tree,
                                              __isl_keep PypetContext* init_pc,
                                              __isl_take PypetState* state) {
  CHECK(tree);
  isl_id* iv = PypetExprAccessGetId(tree->ast.Loop.iv);
  PypetContext* pc = PypetContextCopy(init_pc);
  pc = PypetContextAddInnerIterator(pc, iv);
  pc = PypetContextClearWritesInTree(pc, tree->ast.Loop.body);

  PypetContext* pc_init_val = PypetContextCopy(pc);
  pc_init_val = PypetContextClearValue(pc_init_val, isl_id_copy(iv));
  isl_pw_aff* init_val =
      PypetExprExtractAffine(tree->ast.Loop.init, pc_init_val);
  CHECK(init_val);
  FreePypetContext(pc_init_val);
  isl_pw_aff* pa_inc = PypetExprExtractAffine(tree->ast.Loop.inc, pc);
  CHECK(pa_inc);
  isl_val* inc = PypetExtractCst(pa_inc);
  CHECK(inc);
  if (!isl_pw_aff_involves_nan(pa_inc) && !isl_pw_aff_involves_nan(init_val) &&
      !isl_val_is_nan(inc)) {
    return ScopFromAffineForInit(tree, init_val, pa_inc, inc, init_pc, pc,
                                 state);
  } else {
    UNIMPLEMENTED();
    return nullptr;
  }
}

__isl_keep PypetScop* TreeToScop::ToScop(__isl_take PypetTree* tree,
                                         __isl_take PypetContext* pc,
                                         __isl_take PypetState* state) {
  struct PypetScop* scop = nullptr;

  if (!tree) return nullptr;

  switch (tree->type) {
    case PYPET_TREE_ERROR:
      return nullptr;
    case PYPET_TREE_BLOCK:
      return ScopFromBlock(tree, pc, state);
    case PYPET_TREE_BREAK:
      return ScopFromBreak(tree, pc, state);
    case PYPET_TREE_CONTINUE:
      return ScopFromContinue(tree, pc, state);
    case PYPET_TREE_DECL:
    case PYPET_TREE_DECL_INIT:
      return ScopFromDecl(tree, pc, state);
    case PYPET_TREE_EXPR:
      return ScopFromTreeExpr(tree, pc, state);
    case PYPET_TREE_RETURN:
      return ScopFromReturn(tree, pc, state);
    case PYPET_TREE_IF:
    case PYPET_TREE_IF_ELSE:
      scop = ScopFromIf(tree, pc, state);
      break;
    case PYPET_TREE_FOR:
      scop = ScopFromFor(tree, pc, state);
      break;
  }
  if (!scop) return nullptr;
  return scop;
}

__isl_give PypetScop* TreeToScop::ScopFromTree(__isl_keep PypetTree* tree) {
  // create a universe set as the initial domain.
  isl_set* domain = isl_set_universe(isl_space_set_alloc(ctx, 0, 0));
  // create context with the given domain.
  PypetContext* pc = CreatePypetContext(domain);
  pc = PypetContextAddParameter(tree, pc);

  struct PypetState state = {0};
  state.ctx = tree->ctx;
  state.int_size = 8;  // TODO(yizhu1): update interface
  state.user = nullptr;
  struct PypetScop* scop = ToScop(tree, pc, &state);
  PypetTreeFree(tree);

  if (scop) {
    // Compute the parameter domain of the given set.
    scop->context = isl_set_params(scop->context);
  }

  FreePypetContext(pc);
  return scop;
}

}  // namespace pypet
}  // namespace pypoly
