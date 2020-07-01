#ifndef PYPOLY_CORE_PYPET_TREE_H_
#define PYPOLY_CORE_PYPET_TREE_H_

#include "pypoly/core/pypet/expr.h"

namespace pypoly {
namespace pypet {

struct PypetExpr;

static constexpr const char* tree_type_str[] = {
    [PYPET_TREE_ERROR] = "error",
    [PYPET_TREE_EXPR] = "expression",
    [PYPET_TREE_BLOCK] = "block",
    [PYPET_TREE_BREAK] = "break",
    [PYPET_TREE_CONTINUE] = "continue",
    [PYPET_TREE_DECL] = "declaration",
    [PYPET_TREE_DECL_INIT] = "declaration-init",
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

  int get_lineno() const {
    // Block statement does not correspond to a line in source codes.
    // return -1 instead.
    if (!range) {
      return -1;
    }

    const std::shared_ptr<torch::jit::Source> src = range->source();
    return src->lineno_to_source_lineno(src->lineno_for_offset(range->start()));
  }
};

__isl_give PypetTree* CreatePypetTree(isl_ctx* ctx,
                                      torch::jit::SourceRange const* range,
                                      enum PypetTreeType tree_type);

__isl_give PypetTree* CreatePypetTreeBlock(isl_ctx* ctx, int block, int n);

__isl_null PypetTree* PypetTreeFree(__isl_take PypetTree* tree);

PypetTree* PypetTreeDup(PypetTree* tree);

PypetTree* PypetTreeCopy(PypetTree* tree);

PypetTree* PypetTreeCow(PypetTree* tree);

PypetTree* PypetTreeNewExpr(PypetExpr* expr);

int ForeachExpr(PypetTree* tree, void* user);

int PypetTreeForeachSubTree(
    __isl_keep PypetTree* tree,
    const std::function<int(PypetTree* tree, void* user)>& fn,
    void* user /* points to user data that can be any type.*/);

int PypetTreeForeachExpr(
    __isl_keep PypetTree* tree,
    const std::function<int(PypetExpr* expr, void* user)>& fn, void* user);

int PypetTreeForeachAccessExpr(
    PypetTree* tree, const std::function<int(PypetExpr* expr, void* user)>& fn,
    void* user);

PypetTree* PypetTreeMapExpr(
    PypetTree* tree, const std::function<PypetExpr*(PypetExpr*, void*)>& fn,
    void* user);

PypetTree* PypetTreeMapAccessExpr(
    PypetTree* tree, const std::function<PypetExpr*(PypetExpr*, void*)>& fn,
    void* user);

int PypetTreeWrites(PypetTree* tree, isl_id* id);

bool PypetTreeHasContinueOrBreak(PypetTree* tree);

PypetExpr* PypetTreeDeclGetVar(PypetTree* tree);

PypetExpr* PypetTreeDeclGetInit(PypetTree* tree);

PypetExpr* PypetTreeExprGetExpr(PypetTree* tree);

bool PypetTreeIsAssign(PypetTree* tree);

PypetTree* PypetTreeUpdateDomain(PypetTree* tree,
                                 isl_multi_pw_aff* multi_pw_aff);

struct TreePrettyPrinter {
  static void Print(std::ostream& out, const __isl_keep PypetTree* tree,
                    int indent = 0);
};

static inline std::ostream& operator<<(std::ostream& out, const PypetTree* t) {
  TreePrettyPrinter::Print(out, t);
  return out;
}
}  // namespace pypet
}  // namespace pypoly
#endif  // PYPOLY_CORE_PYPET_TREE_H_
