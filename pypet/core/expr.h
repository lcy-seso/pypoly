
#ifndef PYPET_EXPR_H
#define PYPET_EXPR_H

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

namespace pypet {

enum PypetExprType {
  PYPET_EXPR_ERROR = -1,
  PYPET_EXPR_ACCESS,
  PYPET_EXPR_CALL,
  PYPET_EXPR_OP,
};

enum PypetOpType {
  PYPET_ASSIGN = 0,  // Tensor operation leads to an assignment.
  Pypet_Add,
  Pypet_Sub,
  Pypet_Mul,
  Pypet_Div,
  Pypet_Mod,
  Pypet_Eq,
  Pypet_Ne,
  Pypet_Le,
  Pypet_Ge,
  Pypet_Lt,
  Pypet_Gt,
  Pypet_And,
  Pypet_Xor,
  Pypet_Or,
  Pypet_Not,
};

enum PypetExprAccessType {
  // TODO(Ying) check whether we needs so many access types or not?
  Pypet_Expr_Access_May_Read = 0,
  Pypet_Expr_Access_Begin = Pypet_Expr_Access_May_Read,
  Pypet_Expr_Access_Fake_killed = Pypet_Expr_Access_May_Read,
  Pypet_Expr_Access_May_Write,
  Pypet_Expr_Access_Must_Write,
  Pypet_Expr_Access_End,
  Pypet_Expr_Access_Killed,
};

struct PypetExprAccess {
  PypetExprAccess(){};
  ~PypetExprAccess() = default;
:w

 private:
  isl_id* ref_id;
isl_multi_pw_aff* index;  // index expression.
int depth;
size_t read;
size_t write;
size_t kill;
std::vector<isl_union_map*> access;  // access relation.
};

struct PypetExpr {
  friend PypetExprAccess;

  PypetExpr(){};
  ~PypetExpr() = default;

 private:
  int ref;
  isl_ctx* ctx;

  enum PypetExprType type;

  int type_size;

  std::vector<std::shared_ptr<PyetExpr>> args;

  union {
    struct PypetExprAccess acc;
    enum PypetOpType op;

    // TODO(Ying) Add external function call in future.
    // struct PypetExprCall call_;

    std::string type_name;
    isl_val* i;
  };
};
}  // namespace pypet

#endif
