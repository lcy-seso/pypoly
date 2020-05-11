#include "pypet/core/tree.h"

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

  tree = CreatePypetTree(ctx, range, Pypet_Tree_Block);
  if (!tree) return nullptr;

  tree->ast.Block.block = block;
  tree->ast.Block.n = 0;
  tree->ast.Block.max = n;
  tree->ast.Block.children = isl_calloc_array(ctx, PypetTree*, n);
  if (n && !tree->ast.Block.children) return PypetTreeFree(tree);

  return tree;
}

__isl_null PypetTree* PypetTreeFree(__isl_take PypetTree* tree) {
  int i;

  if (!tree) return nullptr;
  if (--tree->ref > 0) return nullptr;

  isl_id_free(tree->label);

  switch (tree->type) {
    case Pypet_Tree_Error:
      break;
    case Pypet_Tree_Block:
      for (i = 0; i < tree->ast.Block.n; ++i)
        PypetTreeFree(tree->ast.Block.children[i]);
      free(tree->ast.Block.children);
      break;
    case Pypet_Tree_Break:
    case Pypet_Tree_Continue:
      break;
    case Pypet_Tree_Decl:
      PypetExprFree(tree->ast.Decl.var);
      break;
    case Pypet_Tree_Expr:
    case Pypet_Tree_Return:
      PypetExprFree(tree->ast.Expr.expr);
      break;
    case Pypet_Tree_For:
      PypetExprFree(tree->ast.Loop.iv);
      PypetExprFree(tree->ast.Loop.init);
      PypetExprFree(tree->ast.Loop.inc);
    case Pypet_Tree_If_Else:
      PypetTreeFree(tree->ast.IfElse.else_body);
    case Pypet_Tree_If:
      PypetExprFree(tree->ast.IfElse.cond);
      PypetTreeFree(tree->ast.IfElse.if_body);
      break;
  }

  isl_ctx_deref(tree->ctx);
  free(tree);
  return nullptr;
}

}  // namespace pypet
