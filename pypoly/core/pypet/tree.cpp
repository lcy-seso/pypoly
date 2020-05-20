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
  tree->ast.Block.max = n;
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

void TreePrettyPrinter::Print(std::ostream& out,
                              const __isl_keep PypetTree* tree, int indent) {
  if (!tree) return;

  if (tree->label) {
    out << std::string(indent, ' ');
    out << std::string(isl_id_to_str(tree->label));
  }

  // TODO: The indention is not tested for compliated PypetTree structurem which
  // requires further implementations.
  switch (tree->type) {
    case PYPET_TREE_ERROR:
      out << "ERROR!";
      return;
    case PYPET_TREE_BLOCK:
      for (int i = 0; i < tree->ast.Block.n; ++i) {
        Print(out, tree->ast.Block.children[i], indent + 2);
      }
      break;
    case PYPET_TREE_BREAK:
    case PYPET_TREE_CONTINUE:
    case PYPET_TREE_EXPR:
    case PYPET_TREE_RETURN:
      out << tree->ast.Expr.expr;
      break;
    case PYPET_TREE_DECL:
      out << tree->ast.Decl.var;
      out << std::string(indent, ' ');
      out << tree->ast.Decl.init;
      break;
    case PYPET_TREE_IF:
    case PYPET_TREE_IF_ELSE:
      out << tree->ast.IfElse.cond;
      out << std::string(indent, ' ');
      out << tree->ast.IfElse.if_body;
      if (tree->type != PYPET_TREE_IF_ELSE) break;
      out << tree->ast.IfElse.else_body;
      break;
    case PYPET_TREE_FOR: {
      out << std::string(indent, ' ');
      out << tree->ast.Loop.iv;
      out << tree->ast.Loop.init;
      out << tree->ast.Loop.cond;
      out << tree->ast.Loop.inc;
      out << tree->ast.Loop.body;
    } break;
  }
}

}  // namespace pypet
}  // namespace pypoly
