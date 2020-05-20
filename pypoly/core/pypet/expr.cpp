#include "pypoly/core/pypet/expr.h"

namespace pypoly {
namespace pypet {

static const char* op_str[] = {
    [PypetOpType::PYPET_ASSIGN] = "=", [PypetOpType::PYPET_ADD] = "+",
    [PypetOpType::PYPET_SUB] = "-",    [PypetOpType::PYPET_MUL] = "*",
    [PypetOpType::PYPET_DIV] = "/",    [PypetOpType::PYPET_MOD] = "%",
    [PypetOpType::PYPET_EQ] = "==",    [PypetOpType::PYPET_NE] = "!=",
    [PypetOpType::PYPET_LE] = "<=",    [PypetOpType::PYPET_GE] = ">=",
    [PypetOpType::PYPET_LT] = "<",     [PypetOpType::PYPET_GT] = ">",
    [PypetOpType::PYPET_AND] = "&",    [PypetOpType::PYPET_XOR] = "^",
    [PypetOpType::PYPET_OR] = "|",     [PypetOpType::PYPET_NOT] = "~",
};

PypetExpr* PypetExprAlloc(isl_ctx* ctx, PypetExprType expr_type) {
  PypetExpr* expr = isl_alloc_type(ctx, struct PypetExpr);
  CHECK(expr);
  expr->ctx = ctx;
  isl_ctx_ref(ctx);
  expr->type = expr_type;
  expr->ref = 1;
  switch (expr_type) {
    case PypetExprType::PYPET_EXPR_ACCESS:
      expr->arg_num = 0;
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

  for (unsigned int i = 0; i < expr->arg_num; ++i) PypetExprFree(expr->args[i]);
  free(expr->args);

  switch (expr->type) {
    case PYPET_EXPR_ACCESS:
      isl_id_free(expr->acc.ref_id);
      for (int type = PYPET_EXPR_ACCESS_BEGIN; type < PYPET_EXPR_ACCESS_END;
           ++type)
        isl_union_map_free(expr->acc.access[type]);
      isl_multi_pw_aff_free(expr->acc.index);
      break;
    case PYPET_EXPR_CALL:
      UNIMPLEMENTED();
      break;
    case PYPET_EXPR_OP:
      UNIMPLEMENTED();
      break;
    case PYPET_EXPR_ERROR:
      UNIMPLEMENTED();
      break;
    default:
      UNIMPLEMENTED();
      break;
  }

  return nullptr;
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

isl_printer* dump_arguments(PypetExpr* expr, isl_printer* p) {
  if (expr->arg_num == 0) {
    return p;
  }

  p = isl_printer_print_str(p, "args");
  p = isl_printer_yaml_next(p);
  p = isl_printer_yaml_start_sequence(p);
  for (unsigned int i = 0; i < expr->arg_num; ++i) {
    p = PypetExprPrint(expr->args[i], p);
    p = isl_printer_yaml_next(p);
  }
  p = isl_printer_yaml_end_sequence(p);
  return p;
}

isl_printer* PypetExprPrint(PypetExpr* expr, isl_printer* p) {
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
      p = dump_arguments(expr, p);
      p = isl_printer_yaml_end_mapping(p);
      break;
    case PypetExprType::PYPET_EXPR_OP:
      p = isl_printer_yaml_start_mapping(p);
      p = isl_printer_print_str(p, "op");
      p = isl_printer_yaml_next(p);
      p = isl_printer_print_str(p, op_str[expr->op]);
      p = isl_printer_yaml_next(p);
      p = dump_arguments(expr, p);
      p = isl_printer_yaml_end_mapping(p);
      break;
    case PypetExprType::PYPET_EXPR_CALL:
      UNIMPLEMENTED();
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

void PypetExprPrint2Stdout(PypetExpr* expr) {
  isl_printer* p = isl_printer_to_str(expr->ctx);
  p = PypetExprPrint(expr, p);
  char* ret_str = isl_printer_get_str(p);
  isl_printer_free(p);
  puts(ret_str);
  free(ret_str);
}

}  // namespace pypet
}  // namespace pypoly
