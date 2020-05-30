#include "pypoly/core/pypet/expr.h"

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
}  // namespace

__isl_give PypetExpr* PypetExprAlloc(isl_ctx* ctx, PypetExprType expr_type) {
  PypetExpr* expr = isl_alloc_type(ctx, struct PypetExpr);
  CHECK(expr);
  expr->ctx = ctx;
  isl_ctx_ref(ctx);
  expr->type = expr_type;
  expr->ref = 1;

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
  // TODO(yizhu1): fix type_size case
  dup->arg_num = expr->arg_num;
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

PypetExpr* PypetExprFromIslVal(isl_val* val) {
  isl_ctx* ctx = isl_val_get_ctx(val);
  PypetExpr* expr = PypetExprAlloc(ctx, PypetExprType::PYPET_EXPR_INT);
  expr->i = val;
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
  PypetExprFree(expr->args[pos]);
  expr->args[pos] = arg;
  return expr;
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
  // TODO: type_size
  expr->arg_num = 2;
  expr->args = new PypetExpr*[2];
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

void ExprPrettyPrinter::Print(std::ostream& out, const PypetExpr* expr,
                              int indent) {
  CHECK(expr) << "null pointer.";

  isl_printer* p = isl_printer_to_str(expr->ctx);
  p = isl_printer_set_indent(p, indent);
  p = isl_printer_set_yaml_style(p, ISL_YAML_STYLE_BLOCK);
  p = isl_printer_start_line(p);
  p = PrintExpr(expr, p);
  out << std::string(isl_printer_get_str(p));
  isl_printer_free(p);
}

bool PypetExprIsAffine(PypetExpr* expr) {
  CHECK(expr);
  CHECK(expr->type == PypetExprType::PYPET_EXPR_ACCESS);
  int has_id = isl_multi_pw_aff_has_tuple_id(expr->acc.index, isl_dim_out);
  CHECK_GE(has_id, 0);
  return !has_id;
}

isl_pw_aff* PypetExprGetAffine(PypetExpr* expr) {
  CHECK(PypetExprIsAffine(expr));
  isl_multi_pw_aff* multi_pw_aff = expr->acc.index;
  isl_pw_aff* pw_aff = isl_multi_pw_aff_get_pw_aff(multi_pw_aff, 0);
  isl_multi_pw_aff_free(multi_pw_aff);
  return pw_aff;
}

__isl_give isl_printer* ExprPrettyPrinter::PrintExpr(
    const PypetExpr* expr, __isl_take isl_printer* p) {
  CHECK(p);
  if (!expr) return isl_printer_free(p);

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
    p = PrintExpr(expr->args[i], p);
    p = isl_printer_yaml_next(p);
  }
  p = isl_printer_yaml_end_sequence(p);

  return p;
}

}  // namespace pypet
}  // namespace pypoly
