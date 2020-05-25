#include "pypoly/core/pypet/tree.h"

namespace pypoly {
namespace pypet {

__isl_give PypetTree* CreatePypetTree(isl_ctx* ctx,
                                      torch::jit::SourceRange const* range,
                                      enum PypetTreeType tree_type) {
  PypetTree* tree;

  tree = isl_calloc_type(ctx, struct PypetTree);
  if (!tree) return nullptr;

  tree->ctx = ctx;
  isl_ctx_ref(ctx);
  tree->ref = 1;
  tree->type = tree_type;
  if (range != nullptr) {
    tree->range = range;
  } else {
    tree->range = nullptr;
  }

  return tree;
}

__isl_give PypetTree* CreatePypetTreeBlock(isl_ctx* ctx, int block, int n) {
  PypetTree* tree;

  tree = CreatePypetTree(ctx, nullptr, PypetTreeType::PYPET_TREE_BLOCK);
  if (!tree) return nullptr;

  tree->ast.Block.block = block;
  tree->ast.Block.n = n;
  tree->ast.Block.max = n;
  tree->ast.Block.children = isl_calloc_array(ctx, PypetTree*, n);
  if (n && !tree->ast.Block.children) return PypetTreeFree(tree);

  return tree;
}

__isl_give PypetTree* CreatePypetTreeFor(isl_ctx* ctx,
                                         const torch::jit::SourceRange& range,
                                         int block, int n) {
  PypetTree* tree;

  tree = CreatePypetTree(ctx, range, PYPET_TREE_FOR);
  if (!tree) return nullptr;

  tree->ast.Block.block = block;
  tree->ast.Block.n = 0;
  tree->ast.Block.max = n;
  tree->ast.Block.children = isl_calloc_array(ctx, PypetTree*, n);
  if (n && !tree->ast.Block.children) {
    return PypetTreeFree(tree);
  }

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
      for (int i = 0; i < tree->ast.Block.n; ++i) {
        PypetTreeFree(tree->ast.Block.children[i]);
      }
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
                              const __isl_keep PypetTree* tree, int indent) {
  CHECK(tree);

  out << std::string(indent, ' ');
  out << std::string(tree_type_str[tree->type]) << std::endl;

  if (tree->label) {
    out << std::string(indent, ' ');
    out << std::string(isl_id_to_str(tree->label)) << std::endl;
    ;
  }

  switch (tree->type) {
    case PYPET_TREE_ERROR:
      UNIMPLEMENTED();
      break;
    case PYPET_TREE_BLOCK:
      for (int i = 0; i < tree->ast.Block.n; ++i) {
        Print(out, tree->ast.Block.children[i], indent + 2);
      }
      break;
    case PYPET_TREE_EXPR:
      DumpPypetExprWithIndent(out, tree->ast.Expr.expr, indent + 2);
      break;
    case PYPET_TREE_BREAK:
    case PYPET_TREE_CONTINUE:
    case PYPET_TREE_RETURN:
    case PYPET_TREE_DECL:
      UNIMPLEMENTED();
      break;
    case PYPET_TREE_IF:
    case PYPET_TREE_IF_ELSE:
      out << std::string(indent, ' ') << "condition:" << std::endl;
      DumpPypetExprWithIndent(out, tree->ast.IfElse.cond, indent + 2);
      out << std::string(indent, ' ') << "then:" << std::endl;
      Print(out, tree->ast.IfElse.if_body, indent + 2);
      if (tree->type != PYPET_TREE_IF_ELSE) break;
      out << std::string(indent, ' ') << "else:" << std::endl;
      Print(out, tree->ast.IfElse.else_body, indent + 2);
      break;
    case PYPET_TREE_FOR: {
      out << std::string(indent, ' ') << "var:" << std::endl;
      DumpPypetExprWithIndent(out, tree->ast.Loop.iv, indent + 2);
      out << std::string(indent, ' ') << "init:" << std::endl;
      DumpPypetExprWithIndent(out, tree->ast.Loop.init, indent + 2);
      out << std::string(indent, ' ') << "cond:" << std::endl;
      DumpPypetExprWithIndent(out, tree->ast.Loop.cond, indent + 2);
      out << std::string(indent, ' ') << "inc:" << std::endl;
      DumpPypetExprWithIndent(out, tree->ast.Loop.inc, indent + 2);
      Print(out, tree->ast.Loop.body, indent + 2);
      break;
    }
    default:
      UNIMPLEMENTED();
      break;
  }
}

}  // namespace pypet
}  // namespace pypoly
