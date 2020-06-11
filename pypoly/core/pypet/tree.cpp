#include "pypoly/core/pypet/tree.h"

namespace pypoly {
namespace pypet {

namespace {

PypetExpr* UpdateDomainWrapperFunc(PypetExpr* expr, void* user) {
  isl_multi_pw_aff* update = static_cast<isl_multi_pw_aff*>(user);
  return PypetExprUpdateDomain(expr, isl_multi_pw_aff_copy(update));
}

struct PypetTreeWritesData {
  isl_id* id;
  int writes;
};

int CheckWrites(PypetExpr* expr, void* user) {
  PypetTreeWritesData* data = static_cast<PypetTreeWritesData*>(user);
  data->writes = PypetExprWrites(expr, data->id);
  if (data->writes < 0 || data->writes) {
    return -1;
  }
  return 0;
}

struct PypetTreeMapAccessExprData {
  std::function<PypetExpr*(
      PypetExpr*, const std::function<PypetExpr*(PypetExpr*, void*)>&, void*)>
      map;
  std::function<PypetExpr*(PypetExpr*, void*)> fn;
  void* user;
};

PypetExpr* PypetTreeMapAccessExprFuncWrapper(PypetExpr* expr, void* user) {
  PypetTreeMapAccessExprData* data =
      static_cast<PypetTreeMapAccessExprData*>(user);
  return data->map(expr, data->fn, data->user);
}

}  // namespace

__isl_give PypetTree* CreatePypetTree(isl_ctx* ctx,
                                      torch::jit::SourceRange const* range,
                                      enum PypetTreeType tree_type) {
  PypetTree* tree = isl_calloc_type(ctx, struct PypetTree);
  CHECK(tree);

  tree->ctx = ctx;
  isl_ctx_ref(ctx);
  tree->ref = 1;
  tree->type = tree_type;
  tree->range = range;
  tree->label = nullptr;

  return tree;
}

__isl_give PypetTree* CreatePypetTreeBlock(isl_ctx* ctx, int block, int n) {
  PypetTree* tree =
      CreatePypetTree(ctx, nullptr, PypetTreeType::PYPET_TREE_BLOCK);

  tree->ast.Block.block = block;
  tree->ast.Block.n = n;
  tree->ast.Block.max = n;
  tree->ast.Block.children = isl_calloc_array(ctx, PypetTree*, n);
  tree->range = nullptr;
  if (n && !tree->ast.Block.children) {
    UNIMPLEMENTED();
  }
  return tree;
}

__isl_null PypetTree* PypetTreeFree(__isl_take PypetTree* tree) {
  CHECK(tree);
  if (--tree->ref > 0) return nullptr;

  if (tree->label != nullptr) {
    isl_id_free(tree->label);
  }

  switch (tree->type) {
    case PYPET_TREE_BLOCK:
      for (int i = 0; i < tree->ast.Block.n; ++i) {
        PypetTreeFree(tree->ast.Block.children[i]);
      }
      free(tree->ast.Block.children);
      break;
    case PYPET_TREE_DECL:
      PypetExprFree(tree->ast.Decl.var);
      break;
    case PYPET_TREE_DECL_INIT:
      PypetExprFree(tree->ast.Decl.var);
      PypetExprFree(tree->ast.Decl.init);
      break;
    case PYPET_TREE_EXPR:
      PypetExprFree(tree->ast.Expr.expr);
      break;
    case PYPET_TREE_FOR:
      PypetExprFree(tree->ast.Loop.iv);
      PypetExprFree(tree->ast.Loop.init);
      PypetExprFree(tree->ast.Loop.cond);
      PypetExprFree(tree->ast.Loop.inc);
      PypetTreeFree(tree->ast.Loop.body);
      break;
    case PYPET_TREE_IF:
      PypetExprFree(tree->ast.IfElse.cond);
      PypetTreeFree(tree->ast.IfElse.if_body);
      break;
    case PYPET_TREE_IF_ELSE:
      PypetExprFree(tree->ast.IfElse.cond);
      PypetTreeFree(tree->ast.IfElse.if_body);
      PypetTreeFree(tree->ast.IfElse.else_body);
      break;
    case PYPET_TREE_ERROR:
    case PYPET_TREE_BREAK:
    case PYPET_TREE_CONTINUE:
    case PYPET_TREE_RETURN:
    default:
      UNIMPLEMENTED();
      break;
  }

  isl_ctx_deref(tree->ctx);
  free(tree);
  return nullptr;
}

PypetTree* PypetTreeDup(PypetTree* tree) {
  PypetTree* ret = CreatePypetTree(tree->ctx, tree->range, tree->type);
  switch (tree->type) {
    case PypetTreeType::PYPET_TREE_BLOCK:
      ret->ast.Block.block = tree->ast.Block.block;
      ret->ast.Block.n = tree->ast.Block.n;
      ret->ast.Block.max = tree->ast.Block.max;
      ret->ast.Block.children =
          isl_calloc_array(tree->ctx, PypetTree*, ret->ast.Block.n);
      for (int i = 0; i < ret->ast.Block.n; ++i) {
        ret->ast.Block.children[i] = PypetTreeCopy(tree->ast.Block.children[i]);
      }
      break;
    case PypetTreeType::PYPET_TREE_DECL:
      ret->ast.Decl.var = PypetExprCopy(tree->ast.Decl.var);
      break;
    case PypetTreeType::PYPET_TREE_DECL_INIT:
      ret->ast.Decl.var = PypetExprCopy(tree->ast.Decl.var);
      ret->ast.Decl.init = PypetExprCopy(tree->ast.Decl.init);
      break;
    case PypetTreeType::PYPET_TREE_IF:
      ret->ast.IfElse.cond = PypetExprCopy(tree->ast.IfElse.cond);
      ret->ast.IfElse.if_body = PypetTreeCopy(tree->ast.IfElse.if_body);
      break;
    case PypetTreeType::PYPET_TREE_IF_ELSE:
      ret->ast.IfElse.cond = PypetExprCopy(tree->ast.IfElse.cond);
      ret->ast.IfElse.if_body = PypetTreeCopy(tree->ast.IfElse.if_body);
      ret->ast.IfElse.else_body = PypetTreeCopy(tree->ast.IfElse.else_body);
      break;
    case PypetTreeType::PYPET_TREE_FOR:
      ret->ast.Loop.independent = tree->ast.Loop.independent;
      ret->ast.Loop.declared = tree->ast.Loop.declared;
      ret->ast.Loop.iv = PypetExprCopy(tree->ast.Loop.iv);
      ret->ast.Loop.init = PypetExprCopy(tree->ast.Loop.init);
      ret->ast.Loop.cond = PypetExprCopy(tree->ast.Loop.cond);
      ret->ast.Loop.inc = PypetExprCopy(tree->ast.Loop.inc);
      ret->ast.Loop.body = PypetTreeCopy(tree->ast.Loop.body);
      break;
    case PypetTreeType::PYPET_TREE_EXPR:
      ret->ast.Expr.expr = PypetExprCopy(tree->ast.Expr.expr);
      break;
    case PypetTreeType::PYPET_TREE_ERROR:
    case PypetTreeType::PYPET_TREE_BREAK:
    case PypetTreeType::PYPET_TREE_CONTINUE:
    case PypetTreeType::PYPET_TREE_RETURN:
    default:
      UNIMPLEMENTED();
      break;
  }
  return ret;
}

PypetTree* PypetTreeCopy(PypetTree* tree) {
  ++tree->ref;
  return tree;
}

PypetTree* PypetTreeCow(PypetTree* tree) {
  if (tree->ref == 1) {
    return tree;
  } else {
    --tree->ref;
    return PypetTreeDup(tree);
  }
}

PypetTree* PypetTreeNewExpr(PypetExpr* expr) {
  PypetTree* tree =
      CreatePypetTree(expr->ctx, nullptr, PypetTreeType::PYPET_TREE_EXPR);
  tree->ast.Expr.expr = PypetExprCopy(expr);
  return tree;
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
    case PYPET_TREE_DECL_INIT:
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

struct ForeachExprData {
  std::function<int(PypetExpr*, void*)> func;
  void* user;
};

int ForeachExpr(PypetTree* tree, void* user) {
  CHECK(tree);
  ForeachExprData* data = static_cast<ForeachExprData*>(user);
  CHECK(data);
  const std::function<int(PypetExpr*, void*)>& func_user = data->func;
  void* data_user = data->user;
  switch (tree->type) {
    case PypetTreeType::PYPET_TREE_EXPR:
    case PypetTreeType::PYPET_TREE_RETURN:
      if (func_user(tree->ast.Expr.expr, data_user) < 0) {
        return -1;
      }
      break;
    case PypetTreeType::PYPET_TREE_DECL:
    case PypetTreeType::PYPET_TREE_DECL_INIT:
      UNIMPLEMENTED();
      break;
    case PypetTreeType::PYPET_TREE_IF:
    case PypetTreeType::PYPET_TREE_IF_ELSE:
      if (func_user(tree->ast.IfElse.cond, data_user) < 0) {
        return -1;
      }
      break;
    case PypetTreeType::PYPET_TREE_FOR:
      if (func_user(tree->ast.Loop.iv, data_user) < 0) {
        return -1;
      }
      if (func_user(tree->ast.Loop.init, data_user) < 0) {
        return -1;
      }
      if (func_user(tree->ast.Loop.cond, data_user) < 0) {
        return -1;
      }
      if (func_user(tree->ast.Loop.inc, data_user) < 0) {
        return -1;
      }
      break;
    case PypetTreeType::PYPET_TREE_BLOCK:
    case PypetTreeType::PYPET_TREE_BREAK:
    case PypetTreeType::PYPET_TREE_CONTINUE:
      break;
    case PypetTreeType::PYPET_TREE_ERROR:
    default:
      return -1;
  }
  return 0;
}

int PypetTreeForeachExpr(
    __isl_keep PypetTree* tree,
    const std::function<int(PypetExpr* expr, void* user)>& fn, void* user) {
  struct ForeachExprData data = {fn, user};
  return PypetTreeForeachSubTree(tree, &ForeachExpr, &data);
}

int ForeachAccessExpr(PypetExpr* expr, void* user) {
  struct ForeachExprData* data = static_cast<ForeachExprData*>(user);
  return PypetExprForeachAccessExpr(expr, data->func, data->user);
}

int PypetTreeForeachAccessExpr(
    PypetTree* tree, const std::function<int(PypetExpr* expr, void* user)>& fn,
    void* user) {
  struct ForeachExprData data = {fn, user};
  return PypetTreeForeachExpr(tree, &ForeachAccessExpr, &data);
}

PypetTree* PypetTreeMapExpr(
    PypetTree* tree, const std::function<PypetExpr*(PypetExpr*, void*)>& fn,
    void* user) {
  tree = PypetTreeCow(tree);

  switch (tree->type) {
    case PypetTreeType::PYPET_TREE_BLOCK:
      for (int i = 0; i < tree->ast.Block.n; ++i) {
        tree->ast.Block.children[i] =
            PypetTreeMapExpr(tree->ast.Block.children[i], fn, user);
      }
      break;
    case PypetTreeType::PYPET_TREE_DECL:
      tree->ast.Decl.var = fn(tree->ast.Decl.var, user);
      break;
    case PypetTreeType::PYPET_TREE_DECL_INIT:
      tree->ast.Decl.var = fn(tree->ast.Decl.var, user);
      tree->ast.Decl.init = fn(tree->ast.Decl.init, user);
      break;
    case PypetTreeType::PYPET_TREE_EXPR:
      tree->ast.Expr.expr = fn(tree->ast.Expr.expr, user);
      break;
    case PypetTreeType::PYPET_TREE_IF:
      tree->ast.IfElse.cond = fn(tree->ast.IfElse.cond, user);
      tree->ast.IfElse.if_body =
          PypetTreeMapExpr(tree->ast.IfElse.if_body, fn, user);
      break;
    case PypetTreeType::PYPET_TREE_IF_ELSE:
      tree->ast.IfElse.cond = fn(tree->ast.IfElse.cond, user);
      tree->ast.IfElse.if_body =
          PypetTreeMapExpr(tree->ast.IfElse.if_body, fn, user);
      tree->ast.IfElse.else_body =
          PypetTreeMapExpr(tree->ast.IfElse.else_body, fn, user);
      break;
    case PypetTreeType::PYPET_TREE_FOR:
      tree->ast.Loop.iv = fn(tree->ast.Loop.iv, user);
      tree->ast.Loop.init = fn(tree->ast.Loop.init, user);
      tree->ast.Loop.cond = fn(tree->ast.Loop.cond, user);
      tree->ast.Loop.inc = fn(tree->ast.Loop.inc, user);
      tree->ast.Loop.body = PypetTreeMapExpr(tree->ast.Loop.body, fn, user);
      break;
    case PypetTreeType::PYPET_TREE_ERROR:
    case PypetTreeType::PYPET_TREE_BREAK:
    case PypetTreeType::PYPET_TREE_CONTINUE:
    case PypetTreeType::PYPET_TREE_RETURN:
    default:
      UNIMPLEMENTED();
      break;
  }
  return tree;
}

PypetTree* PypetTreeMapAccessExpr(
    PypetTree* tree, const std::function<PypetExpr*(PypetExpr*, void*)>& fn,
    void* user) {
  PypetTreeMapAccessExprData data = {PypetExprMapAccess, fn, user};
  return PypetTreeMapExpr(tree, PypetTreeMapAccessExprFuncWrapper, &data);
}

int PypetTreeWrites(PypetTree* tree, isl_id* id) {
  struct PypetTreeWritesData data = {id, 0};
  if (PypetTreeForeachExpr(tree, CheckWrites, &data) < 0 && !data.writes) {
    return -1;
  }
  return data.writes;
}

bool PypetTreeHasContinueOrBreak(PypetTree* tree) {
  // TODO(yizhu1): add support for continue and break
  return false;
}

PypetExpr* PypetTreeDeclGetVar(PypetTree* tree) {
  CHECK(tree->type == PypetTreeType::PYPET_TREE_DECL ||
        tree->type == PypetTreeType::PYPET_TREE_DECL_INIT);
  return PypetExprCopy(tree->ast.Decl.var);
}

PypetExpr* PypetTreeDeclGetInit(PypetTree* tree) {
  CHECK(tree->type == PypetTreeType::PYPET_TREE_DECL_INIT);
  return PypetExprCopy(tree->ast.Decl.init);
}

PypetExpr* PypetTreeExprGetExpr(PypetTree* tree) {
  CHECK(tree->type == PypetTreeType::PYPET_TREE_EXPR);
  return PypetExprCopy(tree->ast.Expr.expr);
}

bool PypetTreeIsAssign(PypetTree* tree) {
  if (tree->type != PypetTreeType::PYPET_TREE_EXPR) {
    return false;
  }
  if (tree->ast.Expr.expr->type != PypetExprType::PYPET_EXPR_OP) {
    return false;
  }
  return tree->ast.Expr.expr->op == PypetOpType::PYPET_ASSIGN;
}

PypetTree* PypetTreeUpdateDomain(PypetTree* tree,
                                 isl_multi_pw_aff* multi_pw_aff) {
  tree = PypetTreeMapExpr(tree, UpdateDomainWrapperFunc, multi_pw_aff);
  isl_multi_pw_aff_free(multi_pw_aff);
  return tree;
}

void TreePrettyPrinter::Print(std::ostream& out,
                              const __isl_keep PypetTree* tree, int indent) {
  CHECK(tree);

  out << std::string(indent, ' ');
  out << std::string(tree_type_str[tree->type]) << std::endl;

  out << std::string(indent, ' ');
  out << "line : " << tree->get_lineno() << std::endl;

  if (tree->label) {
    out << std::string(indent, ' ');
    out << std::string(isl_id_to_str(tree->label)) << std::endl;
  }

  switch (tree->type) {
    case PYPET_TREE_ERROR:
      UNIMPLEMENTED();
      break;
    case PYPET_TREE_BLOCK:
      for (int i = 0; i < tree->ast.Block.n; ++i) {
        out << std::string(indent, ' ') << "block child " << i << ":"
            << std::endl;
        TreePrettyPrinter::Print(out, tree->ast.Block.children[i], indent + 2);
      }
      break;
    case PYPET_TREE_EXPR:
      ExprPrettyPrinter::Print(out, tree->ast.Expr.expr, indent + 2);
      break;
    case PYPET_TREE_BREAK:
    case PYPET_TREE_CONTINUE:
    case PYPET_TREE_RETURN:
    case PYPET_TREE_DECL_INIT:
      UNIMPLEMENTED();
      break;
    case PYPET_TREE_IF:
    case PYPET_TREE_IF_ELSE:
      out << std::string(indent, ' ') << "condition:" << std::endl;
      ExprPrettyPrinter::Print(out, tree->ast.IfElse.cond, indent + 2);
      out << std::string(indent, ' ') << "if:" << std::endl;
      TreePrettyPrinter::Print(out, tree->ast.IfElse.if_body, indent + 2);
      if (tree->type != PYPET_TREE_IF_ELSE) break;
      out << std::string(indent, ' ') << "else:" << std::endl;
      TreePrettyPrinter::Print(out, tree->ast.IfElse.else_body, indent + 2);
      break;
    case PYPET_TREE_FOR: {
      out << std::string(indent, ' ') << "var:" << std::endl;
      ExprPrettyPrinter::Print(out, tree->ast.Loop.iv, indent + 2);
      out << std::string(indent, ' ') << "init:" << std::endl;
      ExprPrettyPrinter::Print(out, tree->ast.Loop.init, indent + 2);
      out << std::endl;
      out << std::string(indent, ' ') << "cond:" << std::endl;
      ExprPrettyPrinter::Print(out, tree->ast.Loop.cond, indent + 2);
      out << std::string(indent, ' ') << "inc:" << std::endl;
      ExprPrettyPrinter::Print(out, tree->ast.Loop.inc, indent + 2);
      out << std::endl;
      out << std::string(indent, ' ') << "for body:" << std::endl;
      TreePrettyPrinter::Print(out, tree->ast.Loop.body, indent + 2);
      break;
    }
    default:
      UNIMPLEMENTED();
      break;
  }
}

}  // namespace pypet
}  // namespace pypoly
