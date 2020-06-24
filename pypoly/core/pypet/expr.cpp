#include "pypoly/core/pypet/expr.h"

#include "pypoly/core/pypet/aff.h"
#include "pypoly/core/pypet/context.h"
#include "pypoly/core/pypet/isl_printer.h"
#include "pypoly/core/pypet/nest.h"

namespace pypoly {
namespace pypet {

namespace {

static void PypetFuncArgFree(struct PypetFuncSummaryArg* arg) {
  if (!arg) return;

  if (arg->type == PYPET_ARG_INT) {
    isl_id_free(arg->id);
  }
  if (arg->type != PYPET_ARG_ARRAY) return;
  for (size_t type = PYPET_EXPR_ACCESS_BEGIN; type < PYPET_EXPR_ACCESS_END;
       ++type) {
    arg->access[type] = isl_union_map_free(arg->access[type]);
  }
}

__isl_null PypetFuncSummary* PypetFuncSummaryFree(
    __isl_take PypetFuncSummary* summary) {
  if (!summary) return nullptr;
  if (--summary->ref > 0) return nullptr;

  for (size_t i = 0; i < summary->n; ++i) {
    PypetFuncArgFree(&summary->arg[i]);
  }

  isl_ctx_deref(summary->ctx);
  if (summary) {
    free(summary);
  }
  return nullptr;
}

int MultiPwAffIsEqual(isl_multi_pw_aff* lhs, isl_multi_pw_aff* rhs) {
  int equal = isl_multi_pw_aff_plain_is_equal(lhs, rhs);
  if (equal < 0 || equal) {
    return equal;
  }
  rhs = isl_multi_pw_aff_copy(rhs);
  rhs = isl_multi_pw_aff_align_params(rhs, isl_multi_pw_aff_get_space(lhs));
  equal = isl_multi_pw_aff_plain_is_equal(lhs, rhs);
  isl_multi_pw_aff_free(rhs);
  return equal;
}

isl_pw_aff* NestedAccess(PypetExpr* expr, PypetContext* context) {
  CHECK(expr);
  CHECK(context);
  if (!context->allow_nested) {
    return NonAffine(PypetContextGetSpace(context));
  }
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
  CHECK(expr);
  CHECK_EQ(expr->arg_num, 2);
  isl_pw_aff* lhs = PypetExprExtractAffine(expr->args[0], context);
  isl_pw_aff* rhs = PypetExprExtractAffine(expr->args[1], context);
  int lhs_cst = isl_pw_aff_is_cst(lhs);
  int rhs_cst = isl_pw_aff_is_cst(rhs);
  CHECK_GE(lhs_cst, 0);
  CHECK_GE(rhs_cst, 0);
  if (lhs_cst || rhs_cst) {
    return isl_pw_aff_mul(lhs, rhs);
  }
  isl_pw_aff_free(lhs);
  isl_pw_aff_free(rhs);
  return NonAffine(PypetContextGetSpace(context));
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
    LOG(FATAL) << "unexpected input, lhs: " << lhs << ", rhs: " << rhs;
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

isl_pw_aff* PypetToBool(isl_pw_aff* pw_aff) {
  CHECK(pw_aff);
  if (isl_pw_aff_involves_nan(pw_aff)) {
    isl_space* space = isl_pw_aff_get_domain_space(pw_aff);
    isl_local_space* local_space = isl_local_space_from_space(space);
    isl_pw_aff_free(pw_aff);
    return isl_pw_aff_nan_on_domain(local_space);
  }
  isl_set* dom = isl_pw_aff_domain(isl_pw_aff_copy(pw_aff));
  isl_set* cond = isl_pw_aff_non_zero_set(pw_aff);
  pw_aff = IndicatorFunction(cond, dom);
  return pw_aff;
}

isl_pw_aff* ExtractImplicitCondition(PypetExpr* expr, PypetContext* context) {
  isl_pw_aff* ret = PypetExprExtractAffine(expr, context);
  return PypetToBool(ret);
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
    case PypetOpType::PYPET_APPLY:
    case PypetOpType::PYPET_LIST_LITERAL:
      return NonAffine(PypetContextGetSpace(context));
    default:
      LOG(FATAL) << expr;
      break;
  }
  CHECK(ret);
  if (isl_pw_aff_involves_nan(ret)) {
    isl_space* space = isl_pw_aff_get_domain_space(ret);
    isl_pw_aff_free(ret);
    return NonAffine(space);
  }
  if (expr->type_size > 0) {
    ret = PypetWrapPwAff(ret, expr->type_size);
  } else if (expr->type_size < 0) {
    ret = SignedOverflow(ret, -expr->type_size);
  }
  return ret;
}

struct PypetExprWritesData {
  isl_id* id;
  int found;
};

int Writes(PypetExpr* expr, void* user) {
  PypetExprWritesData* data = static_cast<PypetExprWritesData*>(user);

  if (!expr->acc.write) {
    return 0;
  }
  if (PypetExprIsAffine(expr)) {
    return 0;
  }

  isl_id* write_id = PypetExprAccessGetId(expr);
  isl_id_free(write_id);
  if (!write_id) {
    return -1;
  }
  if (write_id != data->id) {
    return 0;
  }
  data->found = 1;
  return -1;
}

isl_set* AddArguments(isl_set* domain, int n) {
  if (n == 0) {
    return domain;
  }
  isl_map* map = isl_map_from_domain(domain);
  map = isl_map_add_dims(map, isl_dim_out, n);
  return isl_map_wrap(map);
}

PypetExpr* PypetExprUpdateDomainWrapperFunc(PypetExpr* expr, void* user) {
  isl_multi_pw_aff* update = static_cast<isl_multi_pw_aff*>(user);
  return PypetExprAccessUpdateDomain(expr, update);
}

}  // namespace

__isl_give PypetExpr* PypetExprAlloc(isl_ctx* ctx, PypetExprType expr_type) {
  PypetExpr* expr = isl_alloc_type(ctx, struct PypetExpr);
  CHECK(expr);
  expr->ctx = ctx;
  isl_ctx_ref(ctx);
  expr->type = expr_type;
  expr->ref = 1;

  expr->type_size = 0;
  expr->arg_num = 0;
  expr->args = nullptr;

  switch (expr_type) {
    case PYPET_EXPR_ACCESS:
      expr->acc.ref_id = nullptr;
      expr->acc.index = nullptr;
      expr->acc.depth = 0;
      expr->acc.write = 0;
      expr->acc.kill = 0;
      for (int i = 0; i < PYPET_EXPR_ACCESS_END; ++i)
        expr->acc.access[i] = nullptr;
      break;
    default:
      break;
  }
  return expr;
}

__isl_null PypetExpr* PypetExprFree(__isl_take PypetExpr* expr) {
  if (!expr) return nullptr;
  if (--expr->ref > 0) return nullptr;

  for (size_t i = 0; i < expr->arg_num; ++i) {
    PypetExprFree(expr->args[i]);
  }
  if (expr->args) {
    free(expr->args);
  }

  switch (expr->type) {
    case PYPET_EXPR_ACCESS:
      isl_id_free(expr->acc.ref_id);
      for (int type = PYPET_EXPR_ACCESS_BEGIN; type < PYPET_EXPR_ACCESS_END;
           ++type)
        isl_union_map_free(expr->acc.access[type]);
      isl_multi_pw_aff_free(expr->acc.index);
      break;
    case PYPET_EXPR_CALL:
      if (expr->call.name) {
        free(expr->call.name);
      }
      PypetFuncSummaryFree(expr->call.summary);
      break;
    case PYPET_EXPR_INT:
      isl_val_free(expr->i);
      break;
    case PYPET_EXPR_OP:
    case PYPET_EXPR_ERROR:
      break;
  }

  isl_ctx_deref(expr->ctx);
  free(expr);
  return nullptr;
}

__isl_give PypetExpr* PypetExprCreateCall(isl_ctx* ctx, const char* name,
                                          size_t arg_num) {
  PypetExpr* expr = PypetExprAlloc(ctx, PYPET_EXPR_CALL);
  CHECK(expr);
  expr->arg_num = arg_num;
  expr->call.name = strdup(name);
  if (!expr->call.name) {
    return PypetExprFree(expr);
  }

  expr->call.summary = nullptr;
  return expr;
}

PypetExpr* PypetExprDup(PypetExpr* expr) {
  CHECK(expr);
  PypetExpr* dup = PypetExprAlloc(expr->ctx, expr->type);
  dup = PypetExprSetTypeSize(dup, expr->type_size);
  dup = PypetExprSetNArgs(dup, expr->arg_num);
  for (int i = 0; i < expr->arg_num; ++i) {
    dup = PypetExprSetArg(dup, i, PypetExprCopy(expr->args[i]));
  }

  switch (expr->type) {
    case PypetExprType::PYPET_EXPR_ACCESS:
      if (expr->acc.ref_id) {
        dup->acc.ref_id = isl_id_copy(expr->acc.ref_id);
      }
      dup =
          PypetExprAccessSetIndex(dup, isl_multi_pw_aff_copy(expr->acc.index));
      dup->acc.depth = expr->acc.depth;
      for (int type = PypetExprAccessType::PYPET_EXPR_ACCESS_BEGIN;
           type < PypetExprAccessType::PYPET_EXPR_ACCESS_END; ++type) {
        if (expr->acc.access[type] == nullptr) {
          continue;
        }
        dup->acc.access[type] = isl_union_map_copy(expr->acc.access[type]);
      }
      dup->acc.read = expr->acc.read;
      dup->acc.write = expr->acc.write;
      dup->acc.kill = expr->acc.kill;
      break;
    case PypetExprType::PYPET_EXPR_CALL:
      UNIMPLEMENTED();
      break;
    case PypetExprType::PYPET_EXPR_INT:
      dup->i = isl_val_copy(expr->i);
      break;
    case PypetExprType::PYPET_EXPR_OP:
      dup->op = expr->op;
      break;
    default:
      UNIMPLEMENTED();
      break;
  }
  return dup;
}

PypetExpr* PypetExprCow(PypetExpr* expr) {
  CHECK(expr);
  if (expr->ref == 1) {
    expr->hash = 0;
    return expr;
  } else {
    expr->ref--;
    return PypetExprDup(expr);
  }
}

PypetExpr* PypetExprNewBinary(int type_size, PypetOpType type, PypetExpr* lhs,
                              PypetExpr* rhs) {
  PypetExpr* expr = PypetExprAlloc(lhs->ctx, PypetExprType::PYPET_EXPR_OP);
  expr = PypetExprSetNArgs(expr, 2);
  expr->op = type;
  expr->type_size = type_size;
  expr->args[0] = lhs;
  expr->args[1] = rhs;
  return expr;
}

PypetExpr* PypetExprNewTernary(PypetExpr* p, PypetExpr* q, PypetExpr* r) {
  PypetExpr* expr = PypetExprAlloc(p->ctx, PypetExprType::PYPET_EXPR_OP);
  expr = PypetExprSetNArgs(expr, 3);
  expr->op = PypetOpType::PYPET_COND;
  expr->args[0] = p;
  expr->args[1] = q;
  expr->args[2] = r;
  return expr;
}

PypetExpr* PypetExprSetTypeSize(PypetExpr* expr, int type_size) {
  expr = PypetExprCow(expr);
  CHECK(expr);
  expr->type_size = type_size;
  return expr;
}

PypetExpr* PypetExprFromIslVal(isl_val* val) {
  isl_ctx* ctx = isl_val_get_ctx(val);
  PypetExpr* expr = PypetExprAlloc(ctx, PypetExprType::PYPET_EXPR_INT);
  expr->i = isl_val_copy(val);
  return expr;
}

PypetExpr* PypetExprFromIntVal(isl_ctx* ctx, long val) {
  isl_val* expr_val = isl_val_int_from_si(ctx, val);
  return PypetExprFromIslVal(expr_val);
}

PypetExpr* PypetExprAccessSetIndex(PypetExpr* expr, isl_multi_pw_aff* index) {
  expr = PypetExprCow(expr);
  CHECK(expr);
  CHECK(index);
  CHECK(expr->type == PYPET_EXPR_ACCESS);
  if (expr->acc.index != nullptr) {
    isl_multi_pw_aff_free(expr->acc.index);
  }
  expr->acc.index = index;
  expr->acc.depth = isl_multi_pw_aff_dim(index, isl_dim_out);
  return expr;
}

PypetExpr* PypetExprFromIndex(isl_multi_pw_aff* index) {
  CHECK(index);
  isl_ctx* ctx = isl_multi_pw_aff_get_ctx(index);
  PypetExpr* expr = PypetExprAlloc(ctx, PYPET_EXPR_ACCESS);
  CHECK(expr);
  expr->acc.read = 1;
  expr->acc.write = 0;
  expr->acc.index = nullptr;
  return PypetExprAccessSetIndex(expr, index);
}

PypetExpr* PypetExprSetNArgs(PypetExpr* expr, int n) {
  CHECK(expr);
  if (expr->arg_num == n) {
    return expr;
  }
  expr = PypetExprCow(expr);
  CHECK(expr);
  if (n < expr->arg_num) {
    for (int i = n; i < expr->arg_num; ++i) {
      PypetExprFree(expr->args[i]);
    }
    expr->arg_num = n;
    return expr;
  }
  PypetExpr** args = isl_realloc_array(expr->ctx, expr->args, PypetExpr*, n);
  CHECK(args);
  expr->args = args;
  for (int i = expr->arg_num; i < n; ++i) {
    expr->args[i] = nullptr;
  }
  expr->arg_num = n;
  return expr;
}

PypetExpr* PypetExprCopy(PypetExpr* expr) {
  CHECK(expr);
  ++expr->ref;
  return expr;
}

PypetExpr* PypetExprGetArg(PypetExpr* expr, int pos) {
  CHECK(expr);
  CHECK_GE(pos, 0);
  CHECK_LT(pos, expr->arg_num);
  return PypetExprCopy(expr->args[pos]);
}

PypetExpr* PypetExprSetArg(PypetExpr* expr, int pos, PypetExpr* arg) {
  CHECK(expr);
  CHECK(arg);
  CHECK_GE(pos, 0);
  CHECK_LT(pos, expr->arg_num);

  if (expr->args[pos] == arg) {
    PypetExprFree(arg);
    return expr;
  }

  expr = PypetExprCow(expr);
  CHECK(expr);
  if (expr->args[pos] != nullptr) {
    PypetExprFree(expr->args[pos]);
  }
  expr->args[pos] = arg;
  return expr;
}

isl_union_map* PypetExprAccessGetDependentAccess(PypetExpr* expr,
                                                 PypetExprAccessType type) {
  CHECK(expr);
  CHECK_EQ(expr->type, PypetExprType::PYPET_EXPR_ACCESS);

  if (expr->acc.access[type]) {
    return isl_union_map_copy(expr->acc.access[type]);
  }

  bool empty = false;
  if (type == PypetExprAccessType::PYPET_EXPR_ACCESS_MAY_READ) {
    empty = !expr->acc.read;
  } else {
    empty = !expr->acc.write;
  }

  if (!empty) {
    expr = PypetExprCopy(expr);
    expr = IntroduceAccessRelations(expr);
    isl_union_map* access = isl_union_map_copy(expr->acc.access[type]);
    PypetExprFree(expr);
    return access;
  }

  return isl_union_map_empty(PypetExprAccessGetParameterSpace(expr));
}

isl_union_map* PypetExprAccessGetAccess(PypetExpr* expr,
                                        PypetExprAccessType type) {
  CHECK(expr);
  CHECK_EQ(expr->type, PypetExprType::PYPET_EXPR_ACCESS);

  if (expr->arg_num != 0 &&
      type == PypetExprAccessType::PYPET_EXPR_ACCESS_MUST_WRITE) {
    return isl_union_map_empty(PypetExprAccessGetParameterSpace(expr));
  }

  isl_union_map* access = PypetExprAccessGetDependentAccess(expr, type);
  if (expr->arg_num == 0) {
    return access;
  }

  isl_space* space = isl_multi_pw_aff_get_space(expr->acc.index);
  space = isl_space_domain(space);
  isl_map* map = isl_map_universe(isl_space_unwrap(space));
  map = isl_map_domain_map(map);
  access = isl_union_map_apply_domain(access, isl_union_map_from_map(map));
  return access;
}

isl_space* PypetExprAccessGetAugmentedDomainSpace(PypetExpr* expr) {
  CHECK(expr);
  CHECK(expr->type == PypetExprType::PYPET_EXPR_ACCESS);
  isl_space* space = isl_multi_pw_aff_get_space(expr->acc.index);
  space = isl_space_domain(space);
  return space;
}

isl_space* PypetExprAccessGetDomainSpace(PypetExpr* expr) {
  isl_space* space = PypetExprAccessGetAugmentedDomainSpace(expr);
  if (isl_space_is_wrapping(space) == true) {
    space = isl_space_domain(isl_space_unwrap(space));
  }
  return space;
}

PypetExpr* PypetExprInsertArg(PypetExpr* expr, int pos, PypetExpr* arg) {
  CHECK(expr);
  CHECK(arg);
  CHECK(expr->type == PypetExprType::PYPET_EXPR_ACCESS);
  int n = expr->arg_num;
  CHECK_GE(pos, 0);
  CHECK_LE(pos, n);
  expr = PypetExprSetNArgs(expr, n + 1);
  for (int i = n; i > pos; --i) {
    PypetExprSetArg(expr, i, PypetExprGetArg(expr, i - 1));
  }
  expr = PypetExprSetArg(expr, pos, arg);

  isl_space* space = PypetExprAccessGetDomainSpace(expr);
  space = isl_space_from_domain(space);
  space = isl_space_add_dims(space, isl_dim_out, n + 1);

  isl_multi_aff* multi_aff = nullptr;
  if (n == 0) {
    multi_aff = isl_multi_aff_domain_map(space);
  } else {
    multi_aff = isl_multi_aff_domain_map(isl_space_copy(space));
    isl_multi_aff* new_multi_aff = isl_multi_aff_range_map(space);
    space = isl_space_range(isl_multi_aff_get_space(new_multi_aff));
    isl_multi_aff* proj =
        isl_multi_aff_project_out_map(space, isl_dim_set, pos, 1);
    new_multi_aff = isl_multi_aff_pullback_multi_aff(proj, new_multi_aff);
    multi_aff = isl_multi_aff_range_product(multi_aff, new_multi_aff);
  }
  return PypetExprAccessPullbackMultiAff(expr, multi_aff);
}

PypetExpr* PypetExprAccessPullbackMultiAff(PypetExpr* expr,
                                           isl_multi_aff* multi_aff) {
  expr = PypetExprCow(expr);
  CHECK(expr);
  CHECK(multi_aff);
  CHECK(expr->type == PypetExprType::PYPET_EXPR_ACCESS);
  for (int type = PypetExprAccessType::PYPET_EXPR_ACCESS_BEGIN;
       type < PypetExprAccessType::PYPET_EXPR_ACCESS_END; ++type) {
    if (expr->acc.access[type] == nullptr) {
      continue;
    }
    expr->acc.access[type] = isl_union_map_preimage_domain_multi_aff(
        expr->acc.access[type], isl_multi_aff_copy(multi_aff));
    CHECK(expr->acc.access[type]);
  }
  expr->acc.index =
      isl_multi_pw_aff_pullback_multi_aff(expr->acc.index, multi_aff);
  CHECK(expr->acc.index);
  return expr;
}

PypetExpr* PypetExprAccessMoveDims(PypetExpr* expr, enum isl_dim_type dst_type,
                                   unsigned dst_pos, enum isl_dim_type src_type,
                                   unsigned src_pos, unsigned n) {
  expr = PypetExprCow(expr);
  CHECK(expr->type == PypetExprType::PYPET_EXPR_ACCESS);

  for (int type = PypetExprAccessType::PYPET_EXPR_ACCESS_BEGIN;
       type < PypetExprAccessType::PYPET_EXPR_ACCESS_END; ++type) {
    if (!expr->acc.access[type]) {
      continue;
    }
    expr->acc.access[type] = PypetUnionMapMoveDims(
        expr->acc.access[type], dst_type, dst_pos, src_type, src_pos, n);
  }
  expr->acc.index = isl_multi_pw_aff_move_dims(expr->acc.index, dst_type,
                                               dst_pos, src_type, src_pos, n);
  return expr;
}

PypetExpr* PypetExprAccessAlignParams(PypetExpr* expr) {
  expr = PypetExprCow(expr);
  CHECK(expr->type == PypetExprType::PYPET_EXPR_ACCESS);

  if (!PypetExprAccessHasAnyAccessRelation(expr)) {
    return expr;
  }

  isl_space* space = isl_multi_pw_aff_get_space(expr->acc.index);
  for (int type = PypetExprAccessType::PYPET_EXPR_ACCESS_BEGIN;
       type < PypetExprAccessType::PYPET_EXPR_ACCESS_END; ++type) {
    if (!expr->acc.access[type]) {
      continue;
    }
    space = isl_space_align_params(
        space, isl_union_map_get_space(expr->acc.access[type]));
  }
  expr->acc.index =
      isl_multi_pw_aff_align_params(expr->acc.index, isl_space_copy(space));
  for (int type = PypetExprAccessType::PYPET_EXPR_ACCESS_BEGIN;
       type < PypetExprAccessType::PYPET_EXPR_ACCESS_END; ++type) {
    if (!expr->acc.access[type]) {
      continue;
    }
    expr->acc.access[type] = isl_union_map_align_params(expr->acc.access[type],
                                                        isl_space_copy(space));
    CHECK(expr->acc.access[type]);
  }
  isl_space_free(space);
  return expr;
}

bool PypetExprAccessHasAnyAccessRelation(PypetExpr* expr) {
  CHECK(expr);
  CHECK(expr->type == PypetExprType::PYPET_EXPR_ACCESS);
  for (int type = PypetExprAccessType::PYPET_EXPR_ACCESS_BEGIN;
       type < PypetExprAccessType::PYPET_EXPR_ACCESS_END; ++type) {
    if (expr->acc.access[type]) {
      return true;
    }
  }
  return false;
}

bool PypetExprIsSubAccess(PypetExpr* lhs, PypetExpr* rhs, int arg_num) {
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->type != PypetExprType::PYPET_EXPR_ACCESS) {
    return false;
  }
  if (rhs->type != PypetExprType::PYPET_EXPR_ACCESS) {
    return false;
  }
  if (PypetExprIsAffine(lhs) || PypetExprIsAffine(rhs)) {
    return false;
  }
  int lhs_arg_num = lhs->arg_num;
  int rhs_arg_num = rhs->arg_num;
  if (lhs_arg_num != rhs_arg_num) {
    return false;
  }
  if (lhs_arg_num > arg_num) {
    arg_num = lhs_arg_num;
  }
  for (int i = 0; i < lhs_arg_num; ++i) {
    if (lhs->args[i]->IsEqual(rhs->args[i]) == false) {
      return false;
    }
  }
  isl_id* lhs_id = PypetExprAccessGetId(lhs);
  isl_id* rhs_id = PypetExprAccessGetId(rhs);
  isl_id_free(lhs_id);
  isl_id_free(rhs_id);
  if (!lhs_id || !rhs_id) {
    return false;
  }
  if (lhs_id != rhs_id) {
    return false;
  }

  lhs = PypetExprCopy(lhs);
  rhs = PypetExprCopy(rhs);
  lhs = IntroduceAccessRelations(lhs);
  rhs = IntroduceAccessRelations(rhs);

  int is_subset = isl_union_map_is_subset(
      lhs->acc.access[PypetExprAccessType::PYPET_EXPR_ACCESS_MAY_READ],
      rhs->acc.access[PypetExprAccessType::PYPET_EXPR_ACCESS_MAY_READ]);
  CHECK_GE(is_subset, 0);
  PypetExprFree(lhs);
  PypetExprFree(rhs);
  return is_subset;
}

isl_union_map* ConstructAccessRelation(PypetExpr* expr) {
  CHECK(expr);
  isl_map* access =
      isl_map_from_multi_pw_aff(isl_multi_pw_aff_copy(expr->acc.index));
  CHECK(access);
  int dim = isl_map_dim(access, isl_dim_out);
  CHECK_LE(dim, expr->acc.depth);
  if (dim < expr->acc.depth) {
    access = ExtendRange(access, expr->acc.depth - dim);
  }
  return isl_union_map_from_map(access);
}

isl_map* ExtendRange(isl_map* access, int n) {
  isl_id* id = isl_map_get_tuple_id(access, isl_dim_out);
  if (!isl_map_range_is_wrapping(access)) {
    access = isl_map_add_dims(access, isl_dim_out, n);
  } else {
    isl_map* domain = isl_map_copy(access);
    domain = isl_map_range_factor_domain(domain);
    access = isl_map_range_factor_range(access);
    access = ExtendRange(access, n);
    access = isl_map_range_product(domain, access);
  }
  access = isl_map_set_tuple_id(access, isl_dim_out, id);
  return access;
}

PypetExpr* IntroduceAccessRelations(PypetExpr* expr) {
  CHECK(expr);
  if (expr->HasRelevantAccessRelation()) {
    return expr;
  }

  isl_union_map* access = ConstructAccessRelation(expr);

  expr->hash = 0;
  if (expr->acc.kill &&
      !expr->acc.access[PypetExprAccessType::PYPET_EXPR_ACCESS_FAKE_KILL]) {
    expr->acc.access[PypetExprAccessType::PYPET_EXPR_ACCESS_FAKE_KILL] =
        isl_union_map_copy(access);
  }
  if (expr->acc.read &&
      !expr->acc.access[PypetExprAccessType::PYPET_EXPR_ACCESS_MAY_READ]) {
    expr->acc.access[PypetExprAccessType::PYPET_EXPR_ACCESS_MAY_READ] =
        isl_union_map_copy(access);
  }
  if (expr->acc.write &&
      !expr->acc.access[PypetExprAccessType::PYPET_EXPR_ACCESS_MAY_WRITE]) {
    expr->acc.access[PypetExprAccessType::PYPET_EXPR_ACCESS_MAY_WRITE] =
        isl_union_map_copy(access);
  }
  if (expr->acc.write &&
      !expr->acc.access[PypetExprAccessType::PYPET_EXPR_ACCESS_MUST_WRITE]) {
    expr->acc.access[PypetExprAccessType::PYPET_EXPR_ACCESS_MUST_WRITE] =
        isl_union_map_copy(access);
  }
  isl_union_map_free(access);
  CHECK(expr->HasRelevantAccessRelation());
  return expr;
}

isl_multi_pw_aff* PypetArraySubscript(isl_multi_pw_aff* base,
                                      isl_pw_aff* index) {
  int member_access = isl_multi_pw_aff_range_is_wrapping(base);
  CHECK_GE(member_access, 0);

  if (member_access > 0) {
    isl_id* id = isl_multi_pw_aff_get_tuple_id(base, isl_dim_out);
    isl_multi_pw_aff* domain = isl_multi_pw_aff_copy(base);
    domain = isl_multi_pw_aff_range_factor_domain(domain);
    isl_multi_pw_aff* range = isl_multi_pw_aff_range_factor_range(base);
    range = PypetArraySubscript(range, index);
    isl_multi_pw_aff* access = isl_multi_pw_aff_range_product(domain, range);
    access = isl_multi_pw_aff_set_tuple_id(access, isl_dim_out, id);
    return access;
  } else {
    isl_id* id = isl_multi_pw_aff_get_tuple_id(base, isl_dim_set);
    isl_set* domain = isl_pw_aff_nonneg_set(isl_pw_aff_copy(index));
    index = isl_pw_aff_intersect_domain(index, domain);
    isl_multi_pw_aff* access = isl_multi_pw_aff_from_pw_aff(index);
    access = isl_multi_pw_aff_flat_range_product(base, access);
    access = isl_multi_pw_aff_set_tuple_id(access, isl_dim_set, id);
    return access;
  }
}

PypetExpr* PypetExprAccessSubscript(PypetExpr* expr, PypetExpr* index) {
  expr = PypetExprCow(expr);
  CHECK(expr);
  CHECK(index);
  CHECK(expr->type == PypetExprType::PYPET_EXPR_ACCESS);
  int n = expr->arg_num;
  expr = PypetExprInsertArg(expr, n, index);
  isl_space* space = isl_multi_pw_aff_get_domain_space(expr->acc.index);
  isl_local_space* local_space = isl_local_space_from_space(space);
  isl_pw_aff* pw_aff =
      isl_pw_aff_from_aff(isl_aff_var_on_domain(local_space, isl_dim_set, n));
  expr->acc.index = PypetArraySubscript(expr->acc.index, pw_aff);
  CHECK(expr->acc.index);
  expr->acc.depth = isl_multi_pw_aff_dim(expr->acc.index, isl_dim_out);
  return expr;
}

PypetExpr* PypetExprAccessMember(PypetExpr* expr, isl_id* id) {
  expr = PypetExprCow(expr);
  CHECK(expr);
  CHECK(id);
  CHECK(expr->type == PypetExprType::PYPET_EXPR_ACCESS);
  isl_space* space = isl_multi_pw_aff_get_domain_space(expr->acc.index);
  space = isl_space_from_domain(space);
  space = isl_space_set_tuple_id(space, isl_dim_out, id);
  isl_multi_pw_aff* field_access = isl_multi_pw_aff_zero(space);
  expr->acc.index = PypetArrayMember(expr->acc.index, field_access);
  CHECK(expr->acc.index);
  return expr;
}

PypetExpr* BuildPypetBinaryOpExpr(isl_ctx* ctx, PypetOpType op_type,
                                  PypetExpr* lhs, PypetExpr* rhs) {
  PypetExpr* expr = PypetExprAlloc(ctx, PypetExprType::PYPET_EXPR_OP);
  expr->arg_num = 2;
  expr->args = isl_alloc_array(ctx, PypetExpr*, 2);
  expr->args[0] = lhs;
  expr->args[1] = rhs;
  expr->op = op_type;
  return expr;
}

char* PypetArrayMemberAccessName(isl_ctx* ctx, const char* base,
                                 const char* field) {
  int len = strlen(base) + 1 + strlen(field);
  char* name = isl_alloc_array(ctx, char, len + 1);
  CHECK(name);
  snprintf(name, len + 1, "%s_%s", base, field);
  return name;
}

isl_multi_pw_aff* PypetArrayMember(isl_multi_pw_aff* base,
                                   isl_multi_pw_aff* field) {
  isl_ctx* ctx = isl_multi_pw_aff_get_ctx(base);
  const char* base_name = isl_multi_pw_aff_get_tuple_name(base, isl_dim_out);
  const char* field_name = isl_multi_pw_aff_get_tuple_name(field, isl_dim_out);
  char* name = PypetArrayMemberAccessName(ctx, base_name, field_name);
  isl_multi_pw_aff* access = isl_multi_pw_aff_range_product(base, field);
  access = isl_multi_pw_aff_set_tuple_name(access, isl_dim_out, name);
  free(name);
  return access;
}

int PypetExprForeachExprOfType(PypetExpr* expr, PypetExprType type,
                               const std::function<int(PypetExpr*, void*)>& fn,
                               void* user) {
  CHECK(expr);
  for (int i = 0; i < expr->arg_num; ++i) {
    if (PypetExprForeachExprOfType(expr->args[i], type, fn, user) < 0) {
      return -1;
    }
  }
  if (expr->type == type) {
    return fn(expr, user);
  } else {
    return 0;
  }
}

int PypetExprForeachAccessExpr(PypetExpr* expr,
                               const std::function<int(PypetExpr*, void*)>& fn,
                               void* user) {
  return PypetExprForeachExprOfType(expr, PypetExprType::PYPET_EXPR_ACCESS, fn,
                                    user);
}

int PypetExprIsScalarAccess(PypetExpr* expr) {
  CHECK(expr);
  if (expr->type != PypetExprType::PYPET_EXPR_ACCESS) {
    return 0;
  }
  if (isl_multi_pw_aff_range_is_wrapping(expr->acc.index)) {
    return 0;
  }
  return expr->acc.depth == 0;
}

PypetExpr* PypetExprMapExprOfType(
    PypetExpr* expr, PypetExprType type,
    const std::function<PypetExpr*(PypetExpr*, void*)>& fn, void* user) {
  int start = 0;
  if (expr->type == PypetExprType::PYPET_EXPR_OP &&
      expr->op == PypetOpType::PYPET_ATTRIBUTE) {
    start = 1;
  }
  for (int i = start; i < expr->arg_num; ++i) {
    PypetExpr* arg = PypetExprGetArg(expr, i);
    arg = PypetExprMapExprOfType(arg, type, fn, user);
    expr = PypetExprSetArg(expr, i, arg);
  }
  if (expr->type == type) {
    expr = fn(expr, user);
  }
  return expr;
}

PypetExpr* PypetExprMapAccess(
    PypetExpr* expr, const std::function<PypetExpr*(PypetExpr*, void*)>& fn,
    void* user) {
  return PypetExprMapExprOfType(expr, PypetExprType::PYPET_EXPR_ACCESS, fn,
                                user);
}

PypetExpr* PypetExprMapTopDown(
    PypetExpr* expr, const std::function<PypetExpr*(PypetExpr*, void*)>& fn,
    void* user) {
  CHECK(expr);
  expr = fn(expr, user);
  for (int i = 0; i < expr->arg_num; ++i) {
    PypetExpr* arg = PypetExprGetArg(expr, i);
    arg = PypetExprMapTopDown(arg, fn, user);
    expr = PypetExprSetArg(expr, i, arg);
  }
  return expr;
}

isl_id* PypetExprAccessGetId(PypetExpr* expr) {
  CHECK(expr);
  CHECK(expr->type == PypetExprType::PYPET_EXPR_ACCESS);
  if (isl_multi_pw_aff_range_is_wrapping(expr->acc.index)) {
    isl_space* space = isl_multi_pw_aff_get_space(expr->acc.index);
    space = isl_space_range(space);
    while (space != nullptr && isl_space_is_wrapping(space)) {
      space = isl_space_domain(isl_space_unwrap(space));
    }
    isl_id* id = isl_space_get_tuple_id(space, isl_dim_set);
    isl_space_free(space);
    return id;
  } else {
    return isl_multi_pw_aff_get_tuple_id(expr->acc.index, isl_dim_out);
  }
}

int PypetExprWrites(PypetExpr* expr, isl_id* id) {
  PypetExprWritesData data = {id, 0};
  if (PypetExprForeachAccessExpr(expr, Writes, &data) < 0 && !data.found) {
    return -1;
  }
  return data.found;
}

isl_pw_aff* NonAffine(isl_space* space) {
  return isl_pw_aff_nan_on_domain(isl_local_space_from_space(space));
}

isl_space* PypetExprAccessGetParameterSpace(PypetExpr* expr) {
  CHECK(expr);
  CHECK(expr->type == PypetExprType::PYPET_EXPR_ACCESS);
  isl_space* space = isl_multi_pw_aff_get_space(expr->acc.index);
  return isl_space_params(space);
}

isl_ctx* PypetExprGetCtx(PypetExpr* expr) { return expr->ctx; }

bool PypetExprIsAffine(PypetExpr* expr) {
  CHECK(expr);
  CHECK(expr->type == PypetExprType::PYPET_EXPR_ACCESS);
  int has_id = isl_multi_pw_aff_has_tuple_id(expr->acc.index, isl_dim_out);
  CHECK_GE(has_id, 0);
  return !has_id;
}

isl_pw_aff* PypetExprExtractComparison(PypetOpType type, PypetExpr* lhs,
                                       PypetExpr* rhs, PypetContext* context) {
  if (type == PypetOpType::PYPET_GT) {
    return PypetExprExtractComparison(PypetOpType::PYPET_LT, rhs, lhs, context);
  }
  if (type == PypetOpType::PYPET_GE) {
    return PypetExprExtractComparison(PypetOpType::PYPET_LE, rhs, lhs, context);
  }
  if (type == PypetOpType::PYPET_LT || type == PypetOpType::PYPET_LE) {
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
  isl_pw_aff* lhs_pw_aff = PypetExprExtractAffine(lhs, context);
  isl_pw_aff* rhs_pw_aff = PypetExprExtractAffine(rhs, context);
  return PypetComparison(type, lhs_pw_aff, rhs_pw_aff);
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

isl_pw_aff* PypetExprExtractAffine(PypetExpr* expr, PypetContext* context) {
  CHECK(expr);
  for (auto iter = context->extracted_affine.begin();
       iter != context->extracted_affine.end(); ++iter) {
    if (iter->first->IsEqual(expr)) {
      return isl_pw_aff_copy(iter->second);
    }
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

  context->extracted_affine.insert(
      {PypetExprCopy(expr), isl_pw_aff_copy(pw_aff)});
  return pw_aff;
}

isl_pw_aff* PypetExprGetAffine(PypetExpr* expr) {
  CHECK(PypetExprIsAffine(expr));
  isl_multi_pw_aff* multi_pw_aff = isl_multi_pw_aff_copy(expr->acc.index);
  isl_pw_aff* pw_aff = isl_multi_pw_aff_get_pw_aff(multi_pw_aff, 0);
  isl_multi_pw_aff_free(multi_pw_aff);
  return pw_aff;
}

bool PypetExpr::IsComparison() {
  if (type != PypetExprType::PYPET_EXPR_OP) {
    return false;
  }
  switch (op) {
    case PypetOpType::PYPET_EQ:
    case PypetOpType::PYPET_NE:
    case PypetOpType::PYPET_LE:
    case PypetOpType::PYPET_GE:
    case PypetOpType::PYPET_LT:
    case PypetOpType::PYPET_GT:
      return true;
    default:
      return false;
  }
}

bool PypetExpr::IsBoolean() {
  // TODO(yizhu1): add support for land, lor and lnot
  return false;
}

bool PypetExpr::IsMin() {
  if (type != PypetExprType::PYPET_EXPR_OP) {
    return false;
  }
  if (op != PypetOpType::PYPET_APPLY) {
    return false;
  }
  if (arg_num != 3) {
    return false;
  }
  isl_space* space = isl_multi_pw_aff_get_space(acc.index);
  isl_id* id = isl_space_get_tuple_id(space, isl_dim_out);

  if (strcmp(isl_id_get_name(id), "min") != 0) {
    return false;
  }
  return true;
}

bool PypetExpr::IsMax() {
  if (type != PypetExprType::PYPET_EXPR_OP) {
    return false;
  }
  if (op != PypetOpType::PYPET_APPLY) {
    return false;
  }
  if (arg_num != 3) {
    return false;
  }
  isl_space* space = isl_multi_pw_aff_get_space(acc.index);
  isl_id* id = isl_space_get_tuple_id(space, isl_dim_out);

  if (strcmp(isl_id_get_name(id), "max") != 0) {
    return false;
  }
  return true;
}

PypetExpr* PypetExprInsertDomain(PypetExpr* expr, isl_space* space) {
  space = isl_space_from_domain(space);
  isl_multi_pw_aff* multi_pw_aff = isl_multi_pw_aff_zero(space);
  return PypetExprUpdateDomain(expr, multi_pw_aff);
}

PypetExpr* PypetExprUpdateDomain(PypetExpr* expr, isl_multi_pw_aff* update) {
  expr = PypetExprMapAccess(expr, PypetExprUpdateDomainWrapperFunc, update);
  isl_multi_pw_aff_free(update);
  return expr;
}

PypetExpr* PypetExprAccessUpdateDomain(PypetExpr* expr,
                                       isl_multi_pw_aff* update) {
  expr = PypetExprCow(expr);
  CHECK(expr);
  CHECK(expr->type == PypetExprType::PYPET_EXPR_ACCESS);

  update = isl_multi_pw_aff_copy(update);
  if (expr->arg_num > 0) {
    isl_space* space = isl_multi_pw_aff_get_space(expr->acc.index);
    space = isl_space_domain(space);
    space = isl_space_unwrap(space);
    space = isl_space_range(space);
    space = isl_space_map_from_set(space);
    isl_multi_pw_aff* identity = isl_multi_pw_aff_identity(space);
    update = isl_multi_pw_aff_product(update, identity);
  }

  for (int type = PypetExprAccessType::PYPET_EXPR_ACCESS_BEGIN;
       type < PypetExprAccessType::PYPET_EXPR_ACCESS_END; ++type) {
    if (expr->acc.access[type] == nullptr) {
      continue;
    }
    expr->acc.access[type] = isl_union_map_preimage_domain_multi_pw_aff(
        expr->acc.access[type], isl_multi_pw_aff_copy(update));
    CHECK(expr->acc.access[type]);
  }
  expr->acc.index =
      isl_multi_pw_aff_pullback_multi_pw_aff(expr->acc.index, update);
  return expr;
}

PypetExpr* PypetExpr::Dup() {
  PypetExpr* dup = PypetExprAlloc(ctx, type);
  dup->type_size = type_size;
  dup->arg_num = arg_num;
  if (arg_num > 0) {
    dup->args = isl_alloc_array(ctx, PypetExpr*, arg_num);
  }
  for (int i = 0; i < arg_num; ++i) {
    dup->args[i] = nullptr;
    dup = PypetExprSetArg(dup, i, PypetExprCopy(args[i]));
  }

  switch (type) {
    case PypetExprType::PYPET_EXPR_ACCESS:
      if (acc.ref_id) {
        dup->acc.ref_id = isl_id_copy(acc.ref_id);
      }
      dup = PypetExprAccessSetIndex(dup, isl_multi_pw_aff_copy(acc.index));
      dup->acc.depth = acc.depth;
      for (int type = PypetExprAccessType::PYPET_EXPR_ACCESS_BEGIN;
           type < PypetExprAccessType::PYPET_EXPR_ACCESS_END; ++type) {
        if (acc.access[type] == nullptr) {
          continue;
        }
        dup->acc.access[type] = isl_union_map_copy(acc.access[type]);
      }
      dup->acc.read = acc.read;
      dup->acc.write = acc.write;
      dup->acc.kill = acc.kill;
      break;
    case PypetExprType::PYPET_EXPR_CALL:
      UNIMPLEMENTED();
      break;
    case PypetExprType::PYPET_EXPR_INT:
      dup->i = isl_val_copy(i);
      break;
    case PypetExprType::PYPET_EXPR_OP:
      dup->op = op;
      break;
    default:
      UNIMPLEMENTED();
      break;
  }
  return dup;
}

PypetExpr* PypetExpr::Cow() {
  if (ref == 1) {
    hash = 0;
    return this;
  } else {
    --ref;
    return Dup();
  }
}

PypetExpr* PypetExpr::RemoveDuplicateArgs() {
  if (arg_num < 2) {
    return this;
  }
  PypetExpr* expr = this;
  for (int i = arg_num - 1; i >= 0; --i) {
    for (int j = 0; j < i; ++j) {
      if (args[i]->IsEqual(args[j])) {
        expr = expr->EquateArg(j, i);
        break;
      }
    }
  }
  return expr;
}

bool PypetExpr::HasRelevantAccessRelation() {
  // TODO(yizhu1): fake killed, may read, may write
  return false;
}

bool PypetExpr::IsEqual(PypetExpr* rhs) {
  CHECK(rhs);
  if (type != rhs->type) {
    return false;
  }
  if (arg_num != rhs->arg_num) {
    return false;
  }
  for (int i = 0; i < arg_num; ++i) {
    if (!args[i]->IsEqual(rhs->args[i])) {
      return false;
    }
  }
  switch (type) {
    case PypetExprType::PYPET_EXPR_INT:
      if (!isl_val_eq(i, rhs->i)) {
        return false;
      }
      break;
    case PypetExprType::PYPET_EXPR_ACCESS:
      if (acc.read != rhs->acc.read) {
        return false;
      }
      if (acc.write != rhs->acc.write) {
        return false;
      }
      if (acc.kill != rhs->acc.kill) {
        return false;
      }
      if (acc.ref_id != rhs->acc.ref_id) {
        return false;
      }
      if (!acc.index || !rhs->acc.index) {
        return false;
      }
      if (!MultiPwAffIsEqual(acc.index, rhs->acc.index)) {
        return false;
      }
      if (acc.depth != rhs->acc.depth) {
        return false;
      }
      if (HasRelevantAccessRelation() != rhs->HasRelevantAccessRelation()) {
        UNIMPLEMENTED();
      }
      for (int type = PypetExprAccessType::PYPET_EXPR_ACCESS_BEGIN;
           type < PypetExprAccessType::PYPET_EXPR_ACCESS_END; ++type) {
        if (!acc.access[type] != !rhs->acc.access[type]) {
          return false;
        }
        if (!acc.access[type]) {
          continue;
        }
        if (!isl_union_map_is_equal(acc.access[type], rhs->acc.access[type])) {
          return false;
        }
      }
      break;
    case PypetExprType::PYPET_EXPR_OP:
      if (op != rhs->op) {
        return false;
      }
      break;
    default:
      UNIMPLEMENTED();
      break;
  }
  return true;
}

PypetExpr* PypetExpr::EquateArg(int i, int j) {
  if (i == j) {
    return this;
  }
  if (i > j) {
    return EquateArg(j, i);
  }
  CHECK_GE(i, 0);
  CHECK_LT(j, arg_num);
  isl_space* space = isl_multi_pw_aff_get_domain_space(acc.index);
  space = isl_space_unwrap(space);
  int in_dim = isl_space_dim(space, isl_dim_in);
  isl_space_free(space);

  i += in_dim;
  j += in_dim;
  space = isl_multi_pw_aff_get_domain_space(acc.index);
  space = isl_space_map_from_set(space);
  isl_multi_aff* multi_aff = isl_multi_aff_identity(space);
  multi_aff =
      isl_multi_aff_set_aff(multi_aff, j, isl_multi_aff_get_aff(multi_aff, i));

  PypetExpr* expr = AccessPullbackMultiAff(multi_aff);
  return expr->AccessProjectOutArg(in_dim, j - in_dim);
}

PypetExpr* PypetExpr::AccessPullbackMultiAff(isl_multi_aff* multi_aff) {
  PypetExpr* expr = Cow();
  CHECK(expr);
  CHECK(multi_aff);
  CHECK(expr->type == PypetExprType::PYPET_EXPR_ACCESS);
  for (int type = PypetExprAccessType::PYPET_EXPR_ACCESS_BEGIN;
       type < PypetExprAccessType::PYPET_EXPR_ACCESS_END; ++type) {
    if (expr->acc.access[type] == nullptr) {
      continue;
    }
    expr->acc.access[type] = isl_union_map_preimage_domain_multi_aff(
        expr->acc.access[type], isl_multi_aff_copy(multi_aff));
    CHECK(expr->acc.access[type]);
  }
  expr->acc.index =
      isl_multi_pw_aff_pullback_multi_aff(expr->acc.index, multi_aff);
  CHECK(expr->acc.index);
  return expr;
}

PypetExpr* PypetExpr::AccessPullbackMultiPwAff(isl_multi_pw_aff* multi_pw_aff) {
  PypetExpr* expr = Cow();
  CHECK(expr);
  CHECK(multi_pw_aff);
  CHECK(expr->type == PypetExprType::PYPET_EXPR_ACCESS);
  for (int type = PypetExprAccessType::PYPET_EXPR_ACCESS_BEGIN;
       type < PypetExprAccessType::PYPET_EXPR_ACCESS_END; ++type) {
    if (expr->acc.access[type] == nullptr) {
      continue;
    }
    expr->acc.access[type] = isl_union_map_preimage_domain_multi_pw_aff(
        expr->acc.access[type], isl_multi_pw_aff_copy(multi_pw_aff));
    CHECK(expr->acc.access[type]);
  }
  expr->acc.index =
      isl_multi_pw_aff_pullback_multi_pw_aff(expr->acc.index, multi_pw_aff);
  CHECK(expr->acc.index);
  return expr;
}

PypetExpr* PypetExpr::AccessProjectOutArg(int dim, int pos) {
  PypetExpr* expr = Cow();
  CHECK(expr);
  CHECK(expr->type == PypetExprType::PYPET_EXPR_ACCESS);
  CHECK_GE(pos, 0);
  CHECK_LT(pos, expr->arg_num);
  isl_bool involves =
      isl_multi_pw_aff_involves_dims(expr->acc.index, isl_dim_in, dim + pos, 1);
  CHECK(involves == 0);
  isl_space* space = isl_multi_pw_aff_get_domain_space(expr->acc.index);
  isl_map* map = isl_map_identity(isl_space_map_from_set(space));
  map = isl_map_eliminate(map, isl_dim_out, dim + pos, 1);
  isl_union_map* umap = isl_union_map_from_map(map);
  for (int type = PypetExprAccessType::PYPET_EXPR_ACCESS_BEGIN;
       type < PypetExprAccessType::PYPET_EXPR_ACCESS_END; ++type) {
    if (!expr->acc.access[type]) {
      continue;
    }
    expr->acc.access[type] = isl_union_map_apply_domain(
        expr->acc.access[type], isl_union_map_copy(umap));
    CHECK(expr->acc.access[type]);
  }
  isl_union_map_free(umap);

  space = isl_multi_pw_aff_get_domain_space(expr->acc.index);
  space = isl_space_unwrap(space);
  isl_space* dom =
      isl_space_map_from_set(isl_space_domain(isl_space_copy(space)));
  isl_multi_aff* ma1 = isl_multi_aff_identity(dom);
  if (expr->arg_num == 1) {
    isl_multi_aff* ma2 = isl_multi_aff_zero(space);
    ma1 = isl_multi_aff_range_product(ma1, ma2);
  } else {
    isl_space* ran = isl_space_map_from_set(isl_space_range(space));
    isl_multi_aff* ma2 = isl_multi_aff_identity(ran);
    ma2 = isl_multi_aff_drop_dims(ma2, isl_dim_in, pos, 1);
    ma1 = isl_multi_aff_product(ma1, ma2);
  }

  expr = expr->AccessPullbackMultiAff(ma1);
  PypetExprFree(expr->args[pos]);
  for (int i = pos; i + 1 < expr->arg_num; ++i) {
    expr->args[i] = expr->args[i + 1];
  }
  --expr->arg_num;
  return expr;
}

PypetExpr* PypetExpr::PlugIn(int pos, isl_pw_aff* value) {
  isl_space* space = isl_multi_pw_aff_get_space(acc.index);
  space = isl_space_unwrap(isl_space_domain(space));
  int in_dim = isl_space_dim(space, isl_dim_in);
  isl_multi_aff* multi_aff = isl_multi_aff_domain_map(space);
  value = isl_pw_aff_pullback_multi_aff(value, multi_aff);

  space = isl_multi_pw_aff_get_space(acc.index);
  space = isl_space_map_from_set(isl_space_domain(space));
  isl_multi_pw_aff* multi_pw_aff = isl_multi_pw_aff_identity(space);
  multi_pw_aff = isl_multi_pw_aff_set_pw_aff(multi_pw_aff, in_dim + pos, value);

  PypetExpr* expr = AccessPullbackMultiPwAff(multi_pw_aff);
  return expr->AccessProjectOutArg(in_dim, pos);
}

PypetExpr* PypetExprRestrict(PypetExpr* expr, isl_set* set) {
  expr = PypetExprCow(expr);
  for (int i = 0; i < expr->arg_num; ++i) {
    expr->args[i] = PypetExprRestrict(expr->args[i], isl_set_copy(set));
  }
  if (expr->type != PypetExprType::PYPET_EXPR_ACCESS) {
    isl_set_free(set);
    return expr;
  }

  expr = IntroduceAccessRelations(expr);
  set = AddArguments(set, expr->arg_num);
  isl_union_set* uset = isl_union_set_from_set(isl_set_copy(set));

  for (int type = PypetExprAccessType::PYPET_EXPR_ACCESS_BEGIN;
       type < PypetExprAccessType::PYPET_EXPR_ACCESS_END; ++type) {
    if (!expr->acc.access[type]) {
      continue;
    }
    expr->acc.access[type] = isl_union_map_intersect_domain(
        expr->acc.access[type], isl_union_set_copy(uset));
    CHECK(expr->acc.access[type]);
  }
  isl_union_set_free(uset);
  expr->acc.index = isl_multi_pw_aff_gist(expr->acc.index, set);
  CHECK(expr->acc.index);
  return expr;
}

void ExprPrettyPrinter::Print(std::ostream& out, const PypetExpr* expr,
                              int indent) {
  CHECK(expr);
  isl_printer* p = isl_printer_to_str(expr->ctx);
  CHECK(p);
  p = isl_printer_set_indent(p, indent);
  p = isl_printer_set_yaml_style(p, ISL_YAML_STYLE_BLOCK);
  p = isl_printer_start_line(p);
  p = ExprPrettyPrinter::Print(p, expr);
  out << std::string(isl_printer_get_str(p));
  isl_printer_free(p);
}

__isl_give isl_printer* ExprPrettyPrinter::Print(__isl_take isl_printer* p,
                                                 const PypetExpr* expr) {
  CHECK(p);
  if (!expr) {
    isl_printer_free(p);
    LOG(FATAL) << "null expr." << std::endl;
    return nullptr;
  }

  switch (expr->type) {
    case PypetExprType::PYPET_EXPR_INT:
      p = isl_printer_print_val(p, expr->i);
      break;
    case PypetExprType::PYPET_EXPR_ACCESS:
      p = isl_printer_yaml_start_mapping(p);
      if (expr->acc.ref_id) {
        p = isl_printer_print_str(p, "ref_id");
        p = isl_printer_yaml_next(p);
        p = isl_printer_print_id(p, expr->acc.ref_id);
        p = isl_printer_yaml_next(p);
      }
      p = isl_printer_print_str(p, "index");
      p = isl_printer_yaml_next(p);
      if (expr->acc.index)
        p = isl_printer_print_multi_pw_aff(p, expr->acc.index);
      p = isl_printer_yaml_next(p);

      p = isl_printer_print_str(p, "depth");
      p = isl_printer_yaml_next(p);
      p = isl_printer_print_int(p, expr->acc.depth);
      p = isl_printer_yaml_next(p);
      if (expr->acc.kill) {
        p = isl_printer_print_str(p, "kill");
        p = isl_printer_yaml_next(p);
        p = isl_printer_print_int(p, 1);
        p = isl_printer_yaml_next(p);
      } else {
        p = isl_printer_print_str(p, "read");
        p = isl_printer_yaml_next(p);
        p = isl_printer_print_int(p, expr->acc.read);
        p = isl_printer_yaml_next(p);
        p = isl_printer_print_str(p, "write");
        p = isl_printer_yaml_next(p);
        p = isl_printer_print_int(p, expr->acc.write);
        p = isl_printer_yaml_next(p);
      }
      if (expr->acc.access[PYPET_EXPR_ACCESS_MAY_READ]) {
        p = isl_printer_print_str(p, "may_read");
        p = isl_printer_yaml_next(p);
        p = isl_printer_print_union_map(
            p, expr->acc.access[PYPET_EXPR_ACCESS_MAY_READ]);
        p = isl_printer_yaml_next(p);
      }
      if (expr->acc.access[PYPET_EXPR_ACCESS_MAY_WRITE]) {
        p = isl_printer_print_str(p, "may_write");
        p = isl_printer_yaml_next(p);
        p = isl_printer_print_union_map(
            p, expr->acc.access[PYPET_EXPR_ACCESS_MAY_WRITE]);
        p = isl_printer_yaml_next(p);
      }
      if (expr->acc.access[PYPET_EXPR_ACCESS_MUST_WRITE]) {
        p = isl_printer_print_str(p, "must_write");
        p = isl_printer_yaml_next(p);
        p = isl_printer_print_union_map(
            p, expr->acc.access[PYPET_EXPR_ACCESS_MUST_WRITE]);
        p = isl_printer_yaml_next(p);
      }
      p = PrintArguments(expr, p);
      p = isl_printer_yaml_end_mapping(p);
      break;
    case PypetExprType::PYPET_EXPR_OP:
      p = isl_printer_yaml_start_mapping(p);
      p = isl_printer_print_str(p, "op");
      p = isl_printer_yaml_next(p);
      p = isl_printer_print_str(p, op_type_to_string[expr->op]);
      p = isl_printer_yaml_next(p);
      p = PrintArguments(expr, p);
      p = isl_printer_yaml_end_mapping(p);
      break;
    case PypetExprType::PYPET_EXPR_CALL:
      p = isl_printer_yaml_start_mapping(p);
      p = isl_printer_print_str(p, "call");
      p = isl_printer_yaml_next(p);
      p = isl_printer_print_str(p, expr->call.name);
      p = isl_printer_print_str(p, "/");
      p = isl_printer_print_int(p, expr->arg_num);
      p = isl_printer_yaml_next(p);
      p = PrintArguments(expr, p);
      if (expr->call.summary) {
        p = isl_printer_print_str(p, "summary");
        p = isl_printer_yaml_next(p);
        p = PrintFuncSummary(expr->call.summary, p);
      }
      p = isl_printer_yaml_end_mapping(p);
      break;
    case PypetExprType::PYPET_EXPR_ERROR:
      p = isl_printer_print_str(p, "ERROR");
      break;
    default:
      UNIMPLEMENTED();
      break;
  }
  return p;
}

__isl_give isl_printer* ExprPrettyPrinter::PrintFuncSummary(
    const __isl_keep PypetFuncSummary* summary, __isl_take isl_printer* p) {
  if (!summary || !p) return isl_printer_free(p);
  p = isl_printer_yaml_start_sequence(p);
  for (size_t i = 0; i < summary->n; ++i) {
    switch (summary->arg[i].type) {
      case PYPET_ARG_INT:
        p = isl_printer_yaml_start_mapping(p);
        p = isl_printer_print_str(p, "id");
        p = isl_printer_yaml_next(p);
        p = isl_printer_print_id(p, summary->arg[i].id);
        p = isl_printer_yaml_next(p);
        p = isl_printer_yaml_end_mapping(p);
        break;
      case PYPET_ARG_TENSOR:  // TODO(Ying): not fully implemented yet.
        p = isl_printer_yaml_start_mapping(p);
        p = isl_printer_print_str(p, "tensor");
        p = isl_printer_yaml_next(p);
        p = isl_printer_print_id(p, summary->arg[i].id);
        p = isl_printer_yaml_next(p);
        p = isl_printer_yaml_end_mapping(p);
        break;
      case PYPET_ARG_OTHER:
        p = isl_printer_print_str(p, "other");
        break;
      case PYPET_ARG_ARRAY:
        p = isl_printer_yaml_start_mapping(p);
        p = isl_printer_print_str(p, "may_read");
        p = isl_printer_yaml_next(p);
        p = isl_printer_print_union_map(
            p, summary->arg[i].access[PYPET_EXPR_ACCESS_MAY_READ]);
        p = isl_printer_yaml_next(p);
        p = isl_printer_print_str(p, "may_write");
        p = isl_printer_yaml_next(p);
        p = isl_printer_print_union_map(
            p, summary->arg[i].access[PYPET_EXPR_ACCESS_MAY_WRITE]);
        p = isl_printer_yaml_next(p);
        p = isl_printer_print_str(p, "must_write");
        p = isl_printer_yaml_next(p);
        p = isl_printer_print_union_map(
            p, summary->arg[i].access[PYPET_EXPR_ACCESS_MUST_WRITE]);
        p = isl_printer_yaml_next(p);
        p = isl_printer_yaml_end_mapping(p);
        break;
    }
  }
  p = isl_printer_yaml_end_sequence(p);

  return p;
}

__isl_give isl_printer* ExprPrettyPrinter::PrintArguments(
    const __isl_keep PypetExpr* expr, __isl_take isl_printer* p) {
  if (expr->arg_num == 0) return p;

  p = isl_printer_print_str(p, "args");
  p = isl_printer_yaml_next(p);
  p = isl_printer_yaml_start_sequence(p);
  for (size_t i = 0; i < expr->arg_num; ++i) {
    p = Print(p, expr->args[i]);
    p = isl_printer_yaml_next(p);
  }
  p = isl_printer_yaml_end_sequence(p);

  return p;
}

}  // namespace pypet
}  // namespace pypoly
