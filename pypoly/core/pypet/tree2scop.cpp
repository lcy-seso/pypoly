#include "pypoly/core/pypet/tree2scop.h"

#include "pypoly/core/pypet/pypet.h"

namespace pypoly {
namespace pypet {

namespace {

int PypetContextDim(PypetContext* pc) {
  CHECK(pc);
  return isl_set_dim(pc->domain, isl_dim_set);
}

PypetContext* PypetContextDup(PypetContext* context) {
  UNIMPLEMENTED();
  return nullptr;
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
  return !strcmp(name, "_PypetExpr");
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
  context->assignments = isl_id_to_pw_aff_drop(context->assignments, id);
  CHECK(context->assignments);
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
  context->assignments = isl_id_to_pw_aff_drop(context->assignments, id);
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
  // TODO
  return false;
}

isl_pw_aff* PypetContextGetValue(PypetContext* context, isl_id* id) {
  // TODO
  return nullptr;
}

isl_pw_aff* NestedAccess(PypetExpr* expr, PypetContext* context) {
  return nullptr;
}

isl_pw_aff* ExtractAffineFromAccess(PypetExpr* expr, PypetContext* context) {
  if (PypetExprIsAffine(expr)) {
    return PypetExprGetAffine(expr);
  }
  CHECK_NE(expr->type_size, 0);
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
  return nullptr;
}

isl_pw_aff* ExtractAffineDivMod(PypetExpr* expr, PypetContext* context) {
  return nullptr;
}

isl_pw_aff* ExtractAffineMul(PypetExpr* expr, PypetContext* context) {
  return nullptr;
}

isl_pw_aff* ExtractAffineNeg(PypetExpr* expr, PypetContext* context) {
  return nullptr;
}

isl_pw_aff* ExtractAffineCond(PypetExpr* expr, PypetContext* context) {
  return nullptr;
}

isl_pw_aff* PypetExprExtractAffineCondition(PypetExpr* expr,
                                            PypetContext* context) {
  return nullptr;
}

isl_pw_aff* PypetWrapPwAff(isl_pw_aff* pw_aff, unsigned width) {
  return nullptr;
}

isl_pw_aff* SignedOverflow(isl_pw_aff* pw_aff, unsigned width) {
  return nullptr;
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

  context->extracted_affine.insert({expr, pw_aff});
  return pw_aff;
}

isl_val* PypetExtractCst(isl_pw_aff* pa) {
  // TODO
  return nullptr;
}

isl_set* PypetContextGetDomain(PypetContext* pc) {
  // TODO
  return nullptr;
}

PypetExpr* PypetContextEvaluateExpr(PypetContext* pc, PypetExpr* expr) {
  // TODO
  return nullptr;
}

PypetContext* PypetContextSetAllowNested(PypetContext* pc, int val) {
  return nullptr;
}

bool IsNestedAllowed(isl_pw_aff* pa, PypetTree* tree) { return false; }

bool CanWrap(isl_set* cond, PypetExpr* expr, isl_val* inc) { return false; }

isl_set* EnforceSubset(isl_set* init_val_map, isl_set* valid_cond) {
  return nullptr;
}

isl_pw_aff* PypetExprExtractComparison(PypetOpType type, PypetExpr* iv,
                                       PypetExpr* init, PypetContext* pc) {
  return nullptr;
}

isl_set* StridedDomain(isl_pw_aff* init_val, isl_val* inc) { return nullptr; }

isl_multi_aff* MapToLast(PypetContext* pc, int, isl_id* label) {
  return nullptr;
}

isl_set* ValidOnNext(isl_set* valid_cond, isl_set* domain, isl_val* inc) {
  return nullptr;
}

PypetContext* PypetContextIntersectDomain(PypetContext* pc, isl_set* domain) {
  return nullptr;
}

}  // namespace

/*
 * create PypetContext from a given domain.
 */
__isl_give PypetContext* CreatePypetContext(__isl_take isl_set* domain) {
  if (!domain) return nullptr;

  isl_id_to_pw_aff* assignments =
      isl_id_to_pw_aff_alloc(isl_set_get_ctx(domain), 0);

  PypetContext* pc =
      isl_calloc_type(isl_set_get_ctx(domain), struct PypetContext);
  if (!pc) {
    isl_set_free(domain);
    isl_id_to_pw_aff_free(assignments);
    return nullptr;
  }

  pc->ref = 1;
  pc->domain = domain;
  pc->assignments = assignments;
  return pc;
}

/*
 * Free a reference to "pc" and return nullptr.
 */
__isl_null PypetContext* FreePypetContext(__isl_take PypetContext* pc) {
  if (!pc) return nullptr;
  if (--pc->ref > 0) return nullptr;

  isl_set_free(pc->domain);
  isl_id_to_pw_aff_free(pc->assignments);
  free(pc);
  return nullptr;
}

/*
 * Add an assignment to "pc" for each parameter in "tree".
 */
__isl_give PypetContext* PypetContextAddParameter(__isl_keep PypetTree* tree,
                                                  __isl_keep PypetContext* pc) {
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
  pc_nested = PypetContextSetAllowNested(pc_nested, 1);

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

  // isl_multi_aff* sched = MapToLast(pc, state->n_loop++, tree->label);
  isl_multi_aff* sched = MapToLast(pc, -1, tree->label);
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

  struct PypetScop* scop = ToScop(tree, pc, nullptr /*TODO*/);
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
