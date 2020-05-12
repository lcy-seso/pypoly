#ifndef _PYPET_TREE_H
#define _PYPET_TREE_H

#include "pypet/core/expr.h"

#include <isl/aff.h>
#include <isl/ctx.h>
#include <isl/id.h>
#include <isl/map.h>
#include <isl/set.h>
#include <isl/space.h>
#include <isl/union_map.h>
#include <isl/union_set.h>
#include <isl/val.h>
#include <torch/csrc/jit/frontend/source_range.h>

namespace pypet {

struct PypetExpr;

enum PypetTreeType {
  Pypet_Tree_Error = -1,
  Pypet_Tree_Expr,
  Pypet_Tree_Block,
  Pypet_Tree_Break,
  Pypet_Tree_Continue,
  Pypet_Tree_Decl,
  Pypet_Tree_If,      /* An if without an else branch */
  Pypet_Tree_If_Else, /* An if with an else branch */
  Pypet_Tree_For,
  Pypet_Tree_Return,
};

struct PypetTree {
  PypetTree() = default;
  ~PypetTree() = default;

  int ref;  // reference identifier.
  isl_ctx* ctx;
  torch::jit::SourceRange range;

  // TODO(Ying): a C style structure for source range is required.
  isl_id* label;

  enum PypetTreeType type;

  union {
    struct {
      int block;
      int n;
      int max;
      PypetTree** children;
    } Block;  // block, such as the body of a for construct.
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
                                      const torch::jit::SourceRange& range,
                                      enum PypetTreeType tree_type);
__isl_give PypetTree* CreatePypetTreeBlock(isl_ctx* ctx,
                                           const torch::jit::SourceRange& range,
                                           int block, int n);
__isl_null PypetTree* PypetTreeFree(__isl_take PypetTree* tree);

}  // namespace pypet
#endif
