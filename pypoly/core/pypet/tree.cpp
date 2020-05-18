#include "pypoly/core/pypet/tree.h"

namespace pypoly {
namespace pypet {

__isl_give PypetTree* CreatePypetTree(isl_ctx* ctx,
                                      const torch::jit::SourceRange& range,
                                      enum PypetTreeType tree_type) {
  PypetTree* tree;

  tree = isl_calloc_type(ctx, struct PypetTree);
  if (!tree) return nullptr;

  tree->ctx = ctx;
  isl_ctx_ref(ctx);
  tree->ref = 1;
  tree->type = tree_type;
  tree->range = range;

  return tree;
}

__isl_give PypetTree* CreatePypetTreeBlock(isl_ctx* ctx,
                                           const torch::jit::SourceRange& range,
                                           int block, int n) {
  PypetTree* tree;

  tree = CreatePypetTree(ctx, range, PYPET_TREE_BLOCK);
  if (!tree) return nullptr;

  tree->ast.Block.block = block;
  tree->ast.Block.n = 0;
  tree->ast.Block.max = n;  // what is the difference between `n` and `max`?
  tree->ast.Block.children = isl_calloc_array(ctx, PypetTree*, n);
  if (n && !tree->ast.Block.children) return PypetTreeFree(tree);

  return tree;
}

__isl_give PypetTree* CreatePypetTreFor(isl_ctx* ctx,
                                        const torch::jit::SourceRange& range,
                                        int block, int n) {
  PypetTree* tree;

  tree = CreatePypetTree(ctx, range, PYPET_TREE_FOR);
  if (!tree) return nullptr;

  tree->ast.Block.block = block;
  tree->ast.Block.n = 0;
  tree->ast.Block.max = n;  // what is the difference between `n` and `max`?
  tree->ast.Block.children = isl_calloc_array(ctx, PypetTree*, n);
  if (n && !tree->ast.Block.children) return PypetTreeFree(tree);

  return tree;
}

__isl_null PypetTree* PypetTreeFree(__isl_take PypetTree* tree) {
  if (!tree) return nullptr;
  if (--tree->ref > 0) return nullptr;

  isl_id_free(tree->label);

  switch (tree->type) {
    case PYPET_TREE_ERROR:
      break;
    case PYPET_TREE_BLOCK:
      for (int i = 0; i < tree->ast.Block.n; ++i)
        PypetTreeFree(tree->ast.Block.children[i]);
      free(tree->ast.Block.children);
      break;
    case PYPET_TREE_BREAK:
    case PYPET_TREE_CONTINUE:
      break;
    case PYPET_TREE_DECL:
      PypetExprFree(tree->ast.Decl.var);
      break;
    case PYPET_TREE_EXPR:
    case PYPET_TREE_RETURN:
      PypetExprFree(tree->ast.Expr.expr);
      break;
    case PYPET_TREE_FOR:
      PypetExprFree(tree->ast.Loop.iv);
      PypetExprFree(tree->ast.Loop.init);
      PypetExprFree(tree->ast.Loop.inc);
    case PYPET_TREE_IF_ELSE:
      PypetTreeFree(tree->ast.IfElse.else_body);
    case PYPET_TREE_IF:
      PypetExprFree(tree->ast.IfElse.cond);
      PypetTreeFree(tree->ast.IfElse.if_body);
      break;
  }

  isl_ctx_deref(tree->ctx);
  free(tree);
  return nullptr;
}

/* DFS traverse the given tree. Call "fn" on each node of "tree", including
 * "tree" itself.
 * Return 0 on success and -1 on error, where "fn" returning a negative value is
 * treated as an error.
 */
int PypetTreeForeachSubTree(
    __isl_keep PypetTree* tree,
    const std::function<int(PypetTree* tree, void* user)>& fn,
    void* user /*points to any type.*/) {
  if (!tree) return -1;

  if (fn(tree, user) < 0) return -1;

  switch (tree->type) {
    case PYPET_TREE_ERROR:
      return -1;
    case PYPET_TREE_BLOCK:
      for (int i = 0; i < tree->ast.Block.n; ++i)
        if (PypetTreeForeachSubTree(tree->ast.Block.children[i], fn, user) < 0)
          return -1;
      break;
    case PYPET_TREE_BREAK:
    case PYPET_TREE_CONTINUE:
    case PYPET_TREE_DECL:
    case PYPET_TREE_EXPR:
    case PYPET_TREE_RETURN:
      break;
    case PYPET_TREE_IF:
      if (PypetTreeForeachSubTree(tree->ast.IfElse.if_body, fn, user) < 0)
        return -1;
      break;
    case PYPET_TREE_IF_ELSE:
      if (PypetTreeForeachSubTree(tree->ast.IfElse.if_body, fn, user) < 0)
        return -1;
      if (PypetTreeForeachSubTree(tree->ast.IfElse.else_body, fn, user) < 0)
        return -1;
      break;
    case PYPET_TREE_FOR:
      if (PypetTreeForeachSubTree(tree->ast.Loop.body, fn, user) < 0) return -1;
      break;
  }

  return 0;
}

void TreePrettyPrinter::Print(std::ostream& out,
                              const __isl_keep PypetExpr* expr, int indent) {
  if (!expr) return;
  isl_printer* p = isl_printer_to_str(expr->ctx);
  p = isl_printer_set_indent(p, indent);
  p = isl_printer_set_yaml_style(p, ISL_YAML_STYLE_BLOCK);
  p = isl_printer_start_line(p);
  p = PrintExpr(expr, p);
  out << std::string(isl_printer_get_str(p));
  isl_printer_free(p);
}

void TreePrettyPrinter::Print(std::ostream& out,
                              const __isl_keep PypetTree* tree, int indent) {
  if (!tree) return;

  if (tree->label) {
    out << std::string(indent, ' ');
    out << std::string(isl_id_to_str(tree->label));
  }

  switch (tree->type) {
    case PYPET_TREE_ERROR:
      out << "ERROR!";
      return;
    case PYPET_TREE_BLOCK:
      for (int i = 0; i < tree->ast.Block.n; ++i)
        Print(out, tree->ast.Block.children[i], indent + 2);
      break;
    case PYPET_TREE_BREAK:
    case PYPET_TREE_CONTINUE:
    case PYPET_TREE_EXPR:
    case PYPET_TREE_RETURN:
      Print(out, tree->ast.Expr.expr, indent + 2);
      break;
    case PYPET_TREE_DECL:
      Print(out, tree->ast.Decl.var, indent + 2);
      out << std::string(indent, ' ');
      Print(out, tree->ast.Decl.init, indent + 2);
      break;
    case PYPET_TREE_IF:
    case PYPET_TREE_IF_ELSE:
      Print(out, tree->ast.IfElse.cond, indent + 2);
      out << std::string(indent, ' ');
      Print(out, tree->ast.IfElse.if_body, indent + 2);
      if (tree->type != PYPET_TREE_IF_ELSE) break;
      out << std::string(indent, ' ');
      Print(out, tree->ast.IfElse.else_body, indent + 2);
      break;
    case PYPET_TREE_FOR:
      out << std::string(indent, ' ');
      Print(out, tree->ast.Loop.iv, indent + 2);
      out << std::string(indent, ' ');
      Print(out, tree->ast.Loop.init, indent + 2);
      out << std::string(indent, ' ');
      Print(out, tree->ast.Loop.cond, indent + 2);
      out << std::string(indent, ' ');
      Print(out, tree->ast.Loop.inc, indent + 2);
      out << std::string(indent, ' ');
      Print(out, tree->ast.Loop.body, indent + 2);
      break;
  }
}

__isl_give isl_printer* TreePrettyPrinter::PrintFuncSummary(
    const __isl_keep PypetFuncSummary* summary, __isl_take isl_printer* p) {
  if (!summary || !p) return isl_printer_free(p);
  p = isl_printer_yaml_start_sequence(p);
  for (int i = 0; i < summary->n; ++i) {
    switch (summary->arg[i].type) {
      case PYPET_ARG_INT:
        p = isl_printer_yaml_start_mapping(p);
        p = isl_printer_print_str(p, "id");
        p = isl_printer_yaml_next(p);
        p = isl_printer_print_id(p, summary->arg[i].id);
        p = isl_printer_yaml_next(p);
        p = isl_printer_yaml_end_mapping(p);
        break;
      case PYPET_ARG_TENSOR:  // TODO(Ying): not implemented yet.
        p = isl_printer_yaml_start_mapping(p);
        p = isl_printer_print_str(p, "tensor");
        p = isl_printer_yaml_next(p);
        p = isl_printer_print_id(
            p, summary->arg[i].id);  // Is tensor stored same as int?
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

__isl_give isl_printer* TreePrettyPrinter::PrintArguments(
    const __isl_keep PypetExpr* expr, __isl_take isl_printer* p) {
  if (expr->arg_num == 0) return p;

  p = isl_printer_print_str(p, "args");
  p = isl_printer_yaml_next(p);
  p = isl_printer_yaml_start_sequence(p);
  for (int i = 0; i < expr->arg_num; ++i) {
    p = PrintExpr(expr->args[i], p);
    p = isl_printer_yaml_next(p);
  }
  p = isl_printer_yaml_end_sequence(p);

  return p;
}

__isl_give isl_printer* TreePrettyPrinter::PrintExpr(
    const __isl_keep PypetExpr* expr, __isl_take isl_printer* p) {
  if (!expr || !p) return isl_printer_free(p);

  switch (expr->type) {
    case PYPET_EXPR_ACCESS:
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
            p, expr->acc.access[PYPET_EXPR_ACCESS_MAY_WRITE]);
        p = isl_printer_yaml_next(p);
      }
      p = PrintArguments(expr, p);
      p = isl_printer_yaml_end_mapping(p);
      break;
    case PYPET_EXPR_OP:
      p = isl_printer_yaml_start_mapping(p);
      p = isl_printer_print_str(p, "op");
      p = isl_printer_yaml_next(p);
      p = isl_printer_print_str(p, OpToString[expr->op]);
      p = isl_printer_yaml_next(p);
      p = PrintArguments(expr, p);
      p = isl_printer_yaml_end_mapping(p);
      break;
    case PYPET_EXPR_CALL:
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
        // p = pet_function_summary_print(expr->c.summary, p);
      }
      p = isl_printer_yaml_end_mapping(p);
      break;
    case PYPET_TREE_ERROR:
      p = isl_printer_print_str(p, "ERROR");
      break;
  }
}
}  // namespace pypet
}  // namespace pypoly
