#ifndef _PYPET_TREE_H
#define _PYPET_TREE_H

#include "pypoly/core/pypet/expr.h"

namespace pypoly {
namespace pypet {

struct PypetExpr;

enum PypetTreeType {
  PYPET_TREE_ERROR = 0,
  PYPET_TREE_EXPR,
  PYPET_TREE_BLOCK,
  PYPET_TREE_BREAK,
  PYPET_TREE_CONTINUE,
  PYPET_TREE_DECL,
  PYPET_TREE_IF,      /* An if without an else branch */
  PYPET_TREE_IF_ELSE, /* An if with an else branch */
  PYPET_TREE_FOR,
  PYPET_TREE_RETURN,
};

static constexpr const char* tree_type_str[] = {
    [PYPET_TREE_ERROR] = "error",
    [PYPET_TREE_EXPR] = "expression",
    [PYPET_TREE_BLOCK] = "block",
    [PYPET_TREE_BREAK] = "break",
    [PYPET_TREE_CONTINUE] = "continue",
    [PYPET_TREE_DECL] = "declaration",
    [PYPET_TREE_IF] = "if",
    [PYPET_TREE_IF_ELSE] = "if-else",
    [PYPET_TREE_FOR] = "for",
    [PYPET_TREE_RETURN] = "return",
};

struct PypetTree {
  PypetTree() = default;
  ~PypetTree() = default;

  int ref;  // reference identifier.
  isl_ctx* ctx;
  torch::jit::SourceRange const* range;

  isl_id* label;  // unique label of this polyhedral statement.

  enum PypetTreeType type;

  union {
    struct {
      int block;  // whether the sequence has its own scope. When this field is
                  // set false?
      int n;      // how many statements in this block.
      int max;
      PypetTree** children;  // each statement in the block is represented in
                             // the form of a tree.
    } Block;                 // block, such as the body of a for construct.
    struct {
      PypetExpr* var;
      PypetExpr* init;
    } Decl;  // declaration.
    struct {
      PypetExpr* expr;
    } Expr;  // expression
    struct {
      int independent;
      int declared;
      PypetExpr* iv;
      PypetExpr* init;
      PypetExpr* cond;
      PypetExpr* inc;
      PypetTree* body;
    } Loop;  // for construct
    struct {
      PypetExpr* cond;
      PypetTree* if_body;
      PypetTree* else_body;
    } IfElse;  // if-else construct
  } ast;
};

__isl_give PypetTree* CreatePypetTree(isl_ctx* ctx,
                                      torch::jit::SourceRange const* range,
                                      enum PypetTreeType tree_type);
__isl_give PypetTree* CreatePypetTreeBlock(isl_ctx* ctx, int block, int n);
__isl_null PypetTree* PypetTreeFree(__isl_take PypetTree* tree);

struct TreePrettyPrinter {
  TreePrettyPrinter(const __isl_keep PypetTree* tree) : tree(tree) {}
  const PypetTree* tree;

  void Print(std::ostream& out, const __isl_keep PypetTree* tree,
             int indent = 2);
};

static inline std::ostream& operator<<(std::ostream& out, TreePrettyPrinter t) {
  t.Print(out, t.tree, 0);
  return out << std::endl;
}

static inline std::ostream& operator<<(std::ostream& out, const PypetTree* t) {
  return out << TreePrettyPrinter(t);
}
}  // namespace pypet
}  // namespace pypoly
#endif
