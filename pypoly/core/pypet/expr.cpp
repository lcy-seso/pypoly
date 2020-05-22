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
      expr->acc.depth = 1;
      expr->acc.write = 1;
      expr->acc.kill = 1;
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
  UNIMPLEMENTED();
  return nullptr;
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

void ExprPrettyPrinter::Print(std::ostream& out, const PypetExpr* expr,
                              int indent) {
  CHECK(expr);

  isl_printer* p = isl_printer_to_str(expr->ctx);
  p = isl_printer_set_indent(p, indent);
  p = isl_printer_set_yaml_style(p, ISL_YAML_STYLE_BLOCK);
  p = isl_printer_start_line(p);
  p = PrintExpr(expr, p);
  out << std::string(isl_printer_get_str(p));
  isl_printer_free(p);
}

__isl_give isl_printer* ExprPrettyPrinter::PrintExpr(
    const PypetExpr* expr, __isl_take isl_printer* p) {
  CHECK(expr);
  CHECK(p);

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
