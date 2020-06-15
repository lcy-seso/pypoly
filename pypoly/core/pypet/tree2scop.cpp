#include "pypoly/core/pypet/tree2scop.h"

#include "pypoly/core/pypet/aff.h"
#include "pypoly/core/pypet/array.h"
#include "pypoly/core/pypet/context.h"
#include "pypoly/core/pypet/isl_printer.h"
#include "pypoly/core/pypet/nest.h"
#include "pypoly/core/pypet/pypet.h"

namespace pypoly {
namespace pypet {

namespace {

int IsAssigned(PypetExpr* expr, PypetTree* tree) {
  isl_id* id = PypetExprAccessGetId(expr);
  int assigned = PypetTreeWrites(tree, id);
  isl_id_free(id);
  return assigned;
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

PypetContext* HandleAssignment(PypetContext* context, PypetTree* tree) {
  PypetExpr* var = nullptr;
  PypetExpr* val = nullptr;
  isl_id* id = nullptr;

  if (tree->type == PypetTreeType::PYPET_TREE_DECL_INIT) {
    var = PypetTreeDeclGetVar(tree);
    var = PypetTreeDeclGetInit(tree);
  } else {
    PypetExpr* expr = PypetTreeExprGetExpr(tree);
    var = PypetExprGetArg(expr, 0);
    val = PypetExprGetArg(expr, 1);
    PypetExprFree(expr);
  }

  if (!PypetExprIsScalarAccess(var)) {
    PypetExprFree(var);
    PypetExprFree(val);
    return context;
  }

  isl_pw_aff* pw_aff = PypetExprExtractAffine(val, context);
  CHECK(pw_aff);
  if (!isl_pw_aff_involves_nan(pw_aff)) {
    id = PypetExprAccessGetId(var);
    context = PypetContextSetValue(context, id, pw_aff);
  } else {
    isl_pw_aff_free(pw_aff);
  }
  PypetExprFree(var);
  PypetExprFree(val);
  return context;
}

PypetContext* ScopHandleWrites(PypetScop* scop, PypetContext* context) {
  for (int i = 0; i < scop->stmt_num; ++i) {
    context = PypetContextClearWritesInTree(context, scop->stmts[i]->body);
  }
  return context;
}

bool IsAssignment(PypetTree* tree) {
  if (tree->type == PypetTreeType::PYPET_TREE_DECL_INIT) {
    return true;
  }
  return PypetTreeIsAssign(tree);
}

PypetTree* PypetTreeResolveAssume(PypetTree* tree, PypetContext* pc) {
  // TODO(yizhu1): assume primitive
  return tree;
}

bool IsAffineCondition(PypetExpr* expr, PypetContext* pc) {
  isl_pw_aff* pa = PypetExprExtractAffineCondition(expr, pc);
  int is_affine = !isl_pw_aff_involves_nan(pa);
  isl_pw_aff_free(pa);
  return is_affine;
}

bool IsConditionalAssignment(PypetTree* tree, PypetContext* pc) {
  if (tree->type != PypetTreeType::PYPET_TREE_IF_ELSE) {
    return false;
  }
  if (tree->ast.IfElse.if_body->type != PypetTreeType::PYPET_TREE_EXPR) {
    return false;
  }
  if (tree->ast.IfElse.else_body->type != PypetTreeType::PYPET_TREE_EXPR) {
    return false;
  }
  PypetExpr* if_expr = tree->ast.IfElse.if_body->ast.Expr.expr;
  PypetExpr* else_expr = tree->ast.IfElse.else_body->ast.Expr.expr;
  if (if_expr->type != PypetExprType::PYPET_EXPR_OP ||
      else_expr->type != PypetExprType::PYPET_EXPR_OP) {
    return false;
  }
  if (if_expr->op != PypetOpType::PYPET_ASSIGN ||
      else_expr->op != PypetOpType::PYPET_ASSIGN) {
    return false;
  }
  PypetExpr* if_var = PypetExprGetArg(if_expr, 0);
  PypetExpr* else_var = PypetExprGetArg(else_expr, 0);
  bool is_equal = if_var->IsEqual(else_var);
  PypetExprFree(if_var);
  PypetExprFree(else_var);
  if (!is_equal) {
    return false;
  }
  if (IsAffineCondition(tree->ast.IfElse.cond, pc)) {
    return false;
  }
  return true;
}

PypetScop* SetIndependence(PypetScop* scop, PypetTree* tree, isl_set* domain,
                           int sign, PypetContext* pc, PypetState* state) {
  // TODO
  return scop;
}

}  // namespace

__isl_keep PypetScop* TreeToScop::ScopFromBlock(__isl_keep PypetTree* tree,
                                                __isl_keep PypetContext* pc,
                                                __isl_take PypetState* state) {
  isl_ctx* ctx = tree->ctx;
  isl_space* space = PypetContextGetSpace(pc);
  pc = PypetContextCopy(pc);
  PypetScop* scop = PypetScop::Create(isl_space_copy(space));
  // TODO(yizhu1): support for kill
  for (int i = 0; i < tree->ast.Block.n; ++i) {
    // TODO(yizhu1): support for continue and break
    PypetScop* cur_scop = ToScop(tree->ast.Block.children[i], pc, state);
    // TODO(yizhu1): check assume primitive
    pc = ScopHandleWrites(cur_scop, pc);
    if (IsAssignment(tree->ast.Block.children[i])) {
      pc = HandleAssignment(pc, tree->ast.Block.children[i]);
    }
    scop = PypetScopAddSeq(ctx, scop, cur_scop);
  }
  FreePypetContext(pc);
  return scop;
}

__isl_keep PypetScop* TreeToScop::ScopFromBreak(__isl_keep PypetTree* tree,
                                                __isl_keep PypetContext* pc,
                                                __isl_take PypetState* state) {
  UNIMPLEMENTED();
  return nullptr;
}

__isl_keep PypetScop* TreeToScop::ScopFromContinue(
    __isl_keep PypetTree* tree, __isl_keep PypetContext* pc,
    __isl_take PypetState* state) {
  UNIMPLEMENTED();
  return nullptr;
}

__isl_keep PypetScop* TreeToScop::ScopFromDecl(__isl_keep PypetTree* tree,
                                               __isl_keep PypetContext* pc,
                                               __isl_take PypetState* state) {
  return nullptr;
}

PypetScop* TreeToScop::ScopFromEvaluatedTree(PypetTree* tree, int stmt_num,
                                             PypetContext* pc) {
  isl_space* space = PypetContextGetSpace(pc);
  tree = PypetTreeResolveNested(tree, space);
  tree = PypetTreeResolveAssume(tree, pc);

  isl_set* domain = PypetContextGetDomain(pc);
  PypetStmt* stmt = PypetStmt::Create(domain, stmt_num, tree);
  return PypetScop::Create(space, stmt);
}

PypetScop* TreeToScop::ScopFromUnevaluatedTree(PypetTree* tree, int stmt_num,
                                               PypetContext* pc) {
  tree = PypetContextEvaluateTree(pc, tree);
  return ScopFromEvaluatedTree(tree, stmt_num, pc);
}

__isl_keep PypetScop* TreeToScop::ScopFromTreeExpr(
    __isl_keep PypetTree* tree, __isl_keep PypetContext* pc,
    __isl_take PypetState* state) {
  return ScopFromUnevaluatedTree(PypetTreeCopy(tree), state->stmt_num++, pc);
}

__isl_keep PypetScop* TreeToScop::ScopFromReturn(__isl_keep PypetTree* tree,
                                                 __isl_keep PypetContext* pc,
                                                 __isl_take PypetState* state) {
  UNIMPLEMENTED();
  return nullptr;
}

PypetScop* TreeToScop::ScopFromEvaluatedExpr(
    PypetExpr* expr, PypetContext* pc, int stmt_num,
    torch::jit::SourceRange const* range) {
  PypetTree* tree = PypetTreeNewExpr(expr);
  tree->range = range;
  return ScopFromEvaluatedTree(tree, stmt_num, pc);
}

PypetScop* TreeToScop::ScopFromConditionalAssignment(PypetTree* tree,
                                                     isl_pw_aff* cond_pw_aff,
                                                     PypetContext* pc,
                                                     PypetState* state) {
  isl_set* cond = isl_pw_aff_non_zero_set(isl_pw_aff_copy(cond_pw_aff));
  isl_set* comp = isl_pw_aff_zero_set(isl_pw_aff_copy(cond_pw_aff));
  isl_multi_pw_aff* index = isl_multi_pw_aff_from_pw_aff(cond_pw_aff);

  PypetExpr* if_expr = tree->ast.IfElse.if_body->ast.Expr.expr;
  PypetExpr* else_expr = tree->ast.IfElse.else_body->ast.Expr.expr;

  PypetExpr* expr_cond = PypetExprFromIndex(index);
  PypetExpr* expr_then = PypetExprGetArg(if_expr, 1);
  expr_then = PypetContextEvaluateExpr(pc, expr_then);
  expr_then = PypetExprRestrict(expr_then, cond);

  PypetExpr* expr_else = PypetExprGetArg(else_expr, 1);
  expr_else = PypetContextEvaluateExpr(pc, expr_else);
  expr_else = PypetExprRestrict(expr_else, comp);

  PypetExpr* expr_write = PypetExprGetArg(if_expr, 0);
  expr_write = PypetContextEvaluateExpr(pc, expr_write);

  PypetExpr* gen_expr = PypetExprNewTernary(expr_cond, expr_then, expr_else);
  int type_size = expr_write->type_size;
  gen_expr = PypetExprNewBinary(type_size, PypetOpType::PYPET_ASSIGN,
                                expr_write, gen_expr);

  PypetScop* scop =
      ScopFromEvaluatedExpr(gen_expr, pc, state->stmt_num++, tree->range);
  FreePypetContext(pc);
  return scop;
}

PypetScop* TreeToScop::ScopFromAffineIf(PypetTree* tree, isl_pw_aff* cond,
                                        PypetContext* pc, PypetState* state) {
  isl_ctx* ctx = tree->ctx;
  bool has_else = tree->type == PypetTreeType::PYPET_TREE_IF_ELSE;
  isl_set* valid = isl_pw_aff_domain(isl_pw_aff_copy(cond));
  isl_set* set = isl_pw_aff_non_zero_set(isl_pw_aff_copy(cond));

  PypetContext* pc_body = PypetContextCopy(pc);
  pc_body = PypetContextIntersectDomain(pc_body, isl_set_copy(set));
  PypetScop* scop_if = ToScop(tree->ast.IfElse.if_body, pc_body, state);
  FreePypetContext(pc_body);

  PypetScop* scop_else = nullptr;
  if (has_else) {
    PypetContext* pc_else = PypetContextCopy(pc);
    isl_set* complement = isl_set_copy(valid);
    complement = isl_set_subtract(valid, isl_set_copy(set));
    pc_else = PypetContextIntersectDomain(
        pc_else, isl_set_copy(isl_set_copy(complement)));
    scop_else = ToScop(tree->ast.IfElse.else_body, pc_else, state);
    scop_else = PypetScopRestrict(scop_else, complement);
    FreePypetContext(pc_else);
  }

  isl_pw_aff_free(cond);

  PypetScop* scop = PypetScopRestrict(scop_if, set);
  if (has_else) {
    scop = PypetScopAddPar(ctx, scop, scop_else);
  }
  scop = PypetScopResolveNested(scop);
  scop = PypetScopRestrictContext(scop, valid);

  FreePypetContext(pc);
  return scop;
}

__isl_keep PypetScop* TreeToScop::ScopFromIf(__isl_keep PypetTree* tree,
                                             __isl_keep PypetContext* pc,
                                             __isl_take PypetState* state) {
  bool has_else = tree->type == PypetTreeType::PYPET_TREE_IF_ELSE;
  pc = PypetContextCopy(pc);
  pc = PypetContextClearWritesInTree(pc, tree->ast.IfElse.if_body);
  if (has_else) {
    pc = PypetContextClearWritesInTree(pc, tree->ast.IfElse.else_body);
  }
  PypetExpr* cond_expr = PypetExprCopy(tree->ast.IfElse.cond);
  cond_expr = PypetContextEvaluateExpr(pc, cond_expr);
  PypetContext* pc_nested = PypetContextCopy(pc);
  pc_nested = PypetContextSetAllowNested(pc_nested, true);
  isl_pw_aff* cond = PypetExprExtractAffineCondition(cond_expr, pc_nested);
  FreePypetContext(pc_nested);
  PypetExprFree(cond_expr);
  CHECK(cond);
  CHECK(isl_pw_aff_involves_nan(cond) == 0);

  if (IsConditionalAssignment(tree, pc)) {
    return ScopFromConditionalAssignment(tree, cond, pc, state);
  }

  if ((!IsNestedAllowed(cond, tree->ast.IfElse.if_body)) ||
      (has_else && !IsNestedAllowed(cond, tree->ast.IfElse.else_body))) {
    isl_pw_aff_free(cond);
    UNIMPLEMENTED();
    return nullptr;
  }

  return ScopFromAffineIf(tree, cond, pc, state);
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
    isl_set* tmp_cond = isl_pw_aff_non_zero_set(pa);
    domain = isl_set_intersect(domain, tmp_cond);
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
  scop = PypetScopResolveNested(scop);
  scop = SetIndependence(scop, tree, domain, isl_val_sgn(inc), pc, state);
  scop = PypetScopEmbed(scop, domain, sched);

  valid_inc = isl_set_intersect(valid_inc, valid_cond_next);
  valid_inc = isl_set_intersect(valid_inc, valid_cond_init);
  valid_inc = isl_set_project_out(valid_inc, isl_dim_set, pos, 1);
  scop = PypetScopRestrictContext(scop, valid_inc);

  isl_val_free(inc);
  valid_init = isl_set_project_out(valid_init, isl_dim_set, pos, 1);
  scop = PypetScopRestrictContext(scop, valid_init);
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

__isl_give PypetScop* TreeToScop::ScopFromTree(__isl_take PypetTree* tree) {
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
