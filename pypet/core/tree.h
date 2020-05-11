#ifndef _PYPET_TREE_H
#define _PYPET_TREE_H

#include "pypet/core/pypet.h"

#include <isl/aff.h>
#include <isl/ctx.h>
#include <isl/id.h>
#include <isl/map.h>
#include <isl/set.h>
#include <isl/space.h>
#include <isl/union_map.h>
#include <isl/union_set.h>
#include <isl/val.h>

namespace Pypet {

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
  PypetTree(){};
  ~PypetTree() = default;

 private:
  int ref;  // reference identifier.
  isl_ctx* ctx;

  torch::jit::SourceRange range;
  isl_id* label;

  enum PypetTreeType type;

  // AST for this statement.
  union {
    struct {
      int block;
      int n;
      int max;
      std::vector<std::shared_ptr<PypetTree>> child;
    } block;  // block.
    struct {
      std::shared_ptr<PypetExpr> var;
      std::shared_ptr<PypetExpr> init;
    } decl;  // declaration.
    struct {
      std::shared_ptr<PypetExpr> expr;
    } expr;  // expression
    struct {
      int independent;
      int declared;
      std::shared_ptr<PypetExpr> iv;
      std::shared_ptr<PypetExpr> init;
      std::shared_ptr<PypetExpr> cond;
      std::shared_ptr<PypetExpr> inc;
      std::shared_ptr<PypetTree> body;
    } loop;  // for
    struct {
      std::shared_ptr<PypetExpr> cond;
      std::shared_ptr<PypetExpr> then_body;
      std::shared_ptr<PypetTree> else_body;
    } if_else;  // if-else
  } ast;
}  // namespace Pypet
