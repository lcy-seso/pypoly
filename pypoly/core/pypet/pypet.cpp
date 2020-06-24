#include "pypoly/core/pypet/pypet.h"

#include "pypoly/core/pypet/aff.h"
#include "pypoly/core/pypet/array.h"
#include "pypoly/core/pypet/expr.h"
#include "pypoly/core/pypet/isl_printer.h"
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
  // TODO(yizhu1): a temporary workaround to avoid adding parameters in APPLY
  // into context
  if (expr->type == PypetExprType::PYPET_EXPR_OP &&
      expr->op == PypetOpType::PYPET_APPLY) {
    return context;
  }
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

  int start = 0;
  if (expr->type == PypetExprType::PYPET_EXPR_OP &&
      expr->op == PypetOpType::PYPET_ATTRIBUTE) {
    start = 1;
  }

  for (int i = start; i < expr->arg_num; ++i) {
    context = ExprExtractContext(expr->args[i], context);
  }

  if (expr->type == PypetExprType::PYPET_EXPR_ACCESS) {
    context = AccessExtractContext(expr, context);
  }
  return context;
}

isl_set* ContextEmbed(isl_set* context, isl_set* dom) {
  int pos = isl_set_dim(context, isl_dim_set) - 1;
  context = isl_set_subtract(isl_set_copy(dom), context);
  context = isl_set_project_out(context, isl_dim_set, pos, 1);
  context = isl_set_complement(context);
  context = PypetNestedRemoveFromSet(context);
  return context;
}

struct PypetOuterProjectionData {
  int n;
  isl_union_pw_multi_aff* ret;
};

isl_stat AddOuterProjection(isl_set* set, void* user) {
  PypetOuterProjectionData* data = static_cast<PypetOuterProjectionData*>(user);
  int dim = isl_set_dim(set, isl_dim_set);
  isl_space* space = isl_set_get_space(set);
  isl_pw_multi_aff* pma = isl_pw_multi_aff_project_out_map(
      space, isl_dim_set, data->n, dim - data->n);
  data->ret = isl_union_pw_multi_aff_add_pw_multi_aff(data->ret, pma);

  isl_set_free(set);
  return isl_stat_ok;
}

isl_multi_union_pw_aff* OuterProjectionMupa(isl_union_set* domain, int n) {
  PypetOuterProjectionData data;
  isl_space* space = isl_union_set_get_space(domain);
  data.n = n;
  data.ret = isl_union_pw_multi_aff_empty(space);
  CHECK_GE(isl_union_set_foreach_set(domain, &AddOuterProjection, &data), 0);
  isl_union_set_free(domain);
  return isl_multi_union_pw_aff_from_union_pw_multi_aff(data.ret);
}

isl_schedule* ScheduleEmbed(isl_schedule* schedule, isl_multi_aff* prefix) {
  isl_union_set* domain = isl_schedule_get_domain(schedule);

  int empty = isl_union_set_is_empty(domain);
  CHECK_GE(empty, 0);
  if (empty) {
    isl_union_set_free(domain);
    return schedule;
  }

  int n = isl_multi_aff_dim(prefix, isl_dim_in);
  isl_multi_union_pw_aff* mupa = OuterProjectionMupa(domain, n);
  isl_multi_aff* ma = isl_multi_aff_copy(prefix);
  mupa = isl_multi_union_pw_aff_apply_multi_aff(mupa, ma);
  schedule = isl_schedule_insert_partial_schedule(schedule, mupa);
  return schedule;
}

}  // namespace

__isl_give isl_printer* ArrayPrettyPrinter::Print(__isl_take isl_printer* p,
                                                  const PypetArray* array) {
  p = isl_printer_yaml_start_mapping(p);

  p = isl_printer_print_str(p, "context");
  p = isl_printer_yaml_next(p);
  p = isl_printer_print_set(p, array->context);
  p = isl_printer_yaml_next(p);

  p = isl_printer_print_str(p, "extent");
  p = isl_printer_yaml_next(p);
  p = isl_printer_print_set(p, array->extent);
  p = isl_printer_yaml_next(p);

  if (array->value_bounds) {
    p = isl_printer_print_str(p, "value_bounds");
    p = isl_printer_yaml_next(p);
    CHECK(array->value_bounds);
    p = isl_printer_print_set(p, array->value_bounds);
    p = isl_printer_yaml_next(p);
  }

  p = isl_printer_print_str(p, "declared");
  p = isl_printer_yaml_next(p);
  p = isl_printer_print_int(p, int(array->declared));
  p = isl_printer_yaml_next(p);

  p = isl_printer_print_str(p, "element type");
  p = isl_printer_yaml_next(p);
  p = isl_printer_print_str(p, array->element_type);
  p = isl_printer_yaml_next(p);

  p = isl_printer_print_str(p, "element shape");
  p = isl_printer_yaml_next(p);

  std::string s = "[";
  for (size_t i = 0; i < array->element_dim - 1; ++i)
    s += std::to_string(array->element_shape[i]) + ",";
  s += (std::to_string(array->element_shape[array->element_dim - 1]) + "]");
  p = isl_printer_print_str(p, s.c_str());
  p = isl_printer_yaml_next(p);

  p = isl_printer_print_str(p, "element size");
  p = isl_printer_yaml_next(p);
  p = isl_printer_print_int(p, array->element_size);
  p = isl_printer_yaml_next(p);

  p = isl_printer_yaml_end_mapping(p);
  return p;
}

void ArrayPrettyPrinter::Print(std::ostream& out, const PypetArray* array,
                               int indent) {
  CHECK(array);
  isl_printer* p = isl_printer_to_str(isl_set_get_ctx(array->context));
  CHECK(p);

  p = isl_printer_set_indent(p, indent);
  p = isl_printer_set_yaml_style(p, ISL_YAML_STYLE_BLOCK);
  p = isl_printer_start_line(p);
  p = Print(p, array);

  out << std::string(isl_printer_get_str(p));
  isl_printer_free(p);
}

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

  CHECK_EQ(tree->type, PYPET_TREE_EXPR);
  stmt->arg_num = 0;
  stmt->args = nullptr;

  return stmt;
}

__isl_null PypetStmt* PypetStmt::Free(__isl_take PypetStmt* stmt) {
  if (!stmt) return nullptr;

  isl_set_free(stmt->domain);
  PypetTreeFree(stmt->body);

  for (size_t i = 0; i < stmt->arg_num; ++i) {
    PypetExprFree(stmt->args[i]);
  }
  free(stmt->args);
  free(stmt);
  return nullptr;
}

isl_union_map* ExprCollectAccess(PypetExpr* expr, PypetExprAccessType type,
                                 isl_union_map* accesses,
                                 isl_union_set* domain) {
  isl_union_map* access = PypetExprAccessGetAccess(expr, type);
  access = isl_union_map_intersect_domain(access, isl_union_set_copy(domain));
  return isl_union_map_union(accesses, access);
}

struct PypetExprCollectAccessesData {
  PypetExprAccessType type;
  isl_union_set* domain;
  isl_union_map* accesses;
};

int ExprCollectAccesses(PypetExpr* expr, void* user) {
  PypetExprCollectAccessesData* data =
      static_cast<PypetExprCollectAccessesData*>(user);
  if (PypetExprIsAffine(expr)) {
    return 0;
  }
  if ((data->type == PypetExprAccessType::PYPET_EXPR_ACCESS_MAY_READ &&
       expr->acc.read) ||
      ((data->type == PypetExprAccessType::PYPET_EXPR_ACCESS_MAY_WRITE ||
        data->type == PypetExprAccessType::PYPET_EXPR_ACCESS_MUST_WRITE) &&
       expr->acc.write)) {
    data->accesses =
        ExprCollectAccess(expr, data->type, data->accesses, data->domain);
  }
  CHECK(data->accesses);
  return 0;
}

isl_union_map* PypetStmt::CollectAccesses(PypetExprAccessType type,
                                          isl_space* dim) const {
  bool must = false;
  if (type == PypetExprAccessType::PYPET_EXPR_ACCESS_MUST_WRITE) {
    must = true;
  }

  if (must && arg_num > 0) {
    return isl_union_map_empty(dim);
  }
  if (must && body->type == PypetTreeType::PYPET_TREE_EXPR) {
    return isl_union_map_empty(dim);
  }

  PypetExprCollectAccessesData data;
  data.type = type;
  data.domain = isl_union_set_from_set(DropArguments(isl_set_copy(domain)));
  data.accesses = isl_union_map_empty(dim);

  PypetTreeForeachAccessExpr(body, ExprCollectAccesses, &data);
  isl_union_set_free(data.domain);
  return data.accesses;
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

__isl_null PypetScop* PypetScop::Free(__isl_take PypetScop* scop) {
  if (!scop) return nullptr;

  isl_set_free(scop->context);
  isl_set_free(scop->context_value);
  isl_schedule_free(scop->schedule);
  if (scop->arrays) {
    for (size_t i = 0; i < scop->array_num; ++i)
      PypetArrayFree(scop->arrays[i]);
  }
  free(scop->arrays);
  if (scop->stmts) {
    for (size_t i = 0; i < scop->stmt_num; ++i) PypetStmt::Free(scop->stmts[i]);
  }
  free(scop->stmts);
  free(scop);
  return nullptr;
}

isl_union_map* PypetScop::ComputeAnyToInner() const {
  return ComputeToInner(false, true);
}

isl_map* SetInnerDomain(isl_map* map, isl_set* dom) {
  isl_map* copy;
  copy = isl_map_copy(map);
  copy = isl_map_intersect_range(copy, isl_set_copy(dom));
  dom = isl_map_domain(isl_map_copy(copy));
  copy = isl_map_gist_domain(copy, dom);
  dom = isl_map_range(copy);
  map = isl_map_intersect_range(map, dom);
  return map;
}

isl_union_map* PypetScop::ComputeToInner(bool from_outermost,
                                         bool to_innermost) const {
  isl_union_map* to_inner = isl_union_map_empty(isl_set_get_space(context));

  for (int i = 0; i < array_num; ++i) {
    PypetArray* array = arrays[i];
    if (to_innermost && array->outer) {
      continue;
    }
    isl_set* set = isl_set_copy(array->extent);
    isl_space* space = isl_set_get_space(set);
    isl_map* map = isl_set_identity(isl_set_universe(space));

    while (map && isl_map_domain_is_wrapping(map)) {
      if (!from_outermost) {
        to_inner = isl_union_map_add_map(to_inner, isl_map_copy(map));
      }
      map = isl_map_domain_factor_domain(map);
      map = SetInnerDomain(map, set);
    }
    isl_set_free(set);
    to_inner = isl_union_map_add_map(to_inner, map);
  }

  return to_inner;
}

isl_union_map* PypetScop::GetMayReads() const {
  return CollectAccesses(PypetExprAccessType::PYPET_EXPR_ACCESS_MAY_READ);
}

isl_union_map* PypetScop::GetMayWrites() const {
  return CollectAccesses(PypetExprAccessType::PYPET_EXPR_ACCESS_MAY_WRITE);
}

isl_union_map* PypetScop::GetMustWrites() const {
  return CollectAccesses(PypetExprAccessType::PYPET_EXPR_ACCESS_MUST_WRITE);
}

isl_union_map* PypetScop::CollectAccesses(PypetExprAccessType type) const {
  isl_space* space = isl_set_get_space(context);
  isl_union_map* accesses = isl_union_map_empty(space);

  for (int i = 0; i < stmt_num; ++i) {
    isl_union_map* accesses_i = stmts[i]->CollectAccesses(type, space);
    accesses = isl_union_map_union(accesses, accesses_i);
  }

  isl_union_set* arrays =
      isl_union_set_empty(isl_union_map_get_space(accesses));
  for (int i = 0; i < array_num; ++i) {
    arrays =
        isl_union_set_add_set(arrays, isl_set_copy(this->arrays[i]->extent));
  }
  accesses = isl_union_map_intersect_range(accesses, arrays);

  isl_union_map* to_inner = ComputeAnyToInner();
  accesses = isl_union_map_apply_range(accesses, to_inner);

  return accesses;
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

PypetScop* PypetScopEmbed(PypetScop* scop, isl_set* dom,
                          isl_multi_aff* schedule) {
  scop->context = ContextEmbed(scop->context, dom);
  scop->schedule = ScheduleEmbed(scop->schedule, schedule);
  isl_set_free(dom);
  isl_multi_aff_free(schedule);
  return scop;
}

__isl_give isl_printer* StmtPrettyPrinter::Print(__isl_take isl_printer* p,
                                                 const PypetStmt* stmt) {
  CHECK(stmt);
  CHECK(p);

  p = isl_printer_yaml_start_mapping(p);

  p = isl_printer_print_str(p, "line");
  p = isl_printer_yaml_next(p);
  p = isl_printer_print_int(p, stmt->body->get_lineno());
  p = isl_printer_yaml_next(p);

  p = isl_printer_print_str(p, "domain");
  p = isl_printer_yaml_next(p);
  p = isl_printer_print_set(p, stmt->domain);
  p = isl_printer_yaml_next(p);

  p = isl_printer_print_str(p, "body");
  CHECK_EQ(stmt->body->type, PYPET_TREE_EXPR);
  p = isl_printer_yaml_next(p);
  p = ExprPrettyPrinter::Print(p, stmt->body->ast.Expr.expr);
  p = isl_printer_yaml_end_mapping(p);
  return p;
}

void StmtPrettyPrinter::Print(std::ostream& out, const PypetStmt* stmt,
                              int indent) {
  CHECK(stmt);
  isl_printer* p = isl_printer_to_str(isl_set_get_ctx(stmt->domain));
  CHECK(p);

  p = isl_printer_set_indent(p, indent);
  p = isl_printer_set_yaml_style(p, ISL_YAML_STYLE_BLOCK);
  p = isl_printer_start_line(p);
  p = Print(p, stmt);

  out << stmt->body << std::endl;
  isl_printer_free(p);
}

__isl_give isl_printer* ScopPrettyPrinter::Print(__isl_take isl_printer* p,
                                                 const PypetScop* scop) {
  p = isl_printer_yaml_start_mapping(p);
  p = isl_printer_set_indent(p, 0);

  p = isl_printer_print_str(p, "context");
  p = isl_printer_yaml_next(p);
  p = isl_printer_print_set(p, scop->context);
  p = isl_printer_yaml_next(p);

  p = isl_printer_print_str(p, "context value");
  p = isl_printer_yaml_next(p);
  p = isl_printer_print_set(p, scop->context_value);
  p = isl_printer_yaml_next(p);

  p = isl_printer_print_str(p, "schedule");
  p = isl_printer_yaml_next(p);
  p = isl_printer_print_schedule(p, scop->schedule);
  p = isl_printer_yaml_next(p);

  p = isl_printer_print_str(p, "arrays");
  p = isl_printer_yaml_next(p);

  if (scop->array_num > 0 && scop->arrays) {
    for (size_t i = 0; i < scop->array_num; ++i) {
      p = isl_printer_yaml_start_sequence(p);
      p = ArrayPrettyPrinter::Print(p, scop->arrays[i]);
      p = isl_printer_yaml_end_sequence(p);
    }
  }

  p = isl_printer_yaml_start_mapping(p);
  p = isl_printer_print_str(p, "statements");
  p = isl_printer_yaml_next(p);
  p = isl_printer_yaml_end_mapping(p);

  if (scop->stmt_num > 0 && scop->stmts) {
    for (size_t i = 0; i < scop->stmt_num; ++i) {
      p = isl_printer_yaml_start_sequence(p);
      p = StmtPrettyPrinter::Print(p, scop->stmts[i]);
      p = isl_printer_yaml_end_sequence(p);
    }
  }

  p = isl_printer_yaml_end_mapping(p);
  return p;
}

void ScopPrettyPrinter::Print(std::ostream& out, const PypetScop* scop,
                              int indent) {
  CHECK(scop);
  CHECK(scop->schedule);
  isl_printer* p = isl_printer_to_str(isl_schedule_get_ctx(scop->schedule));
  CHECK(p);
  p = isl_printer_set_indent(p, indent);
  p = isl_printer_set_yaml_style(p, ISL_YAML_STYLE_BLOCK);
  p = isl_printer_start_line(p);
  p = Print(p, scop);
  out << std::string(isl_printer_get_str(p));
  isl_printer_free(p);
}

}  // namespace pypet
}  // namespace pypoly
