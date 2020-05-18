#ifndef _PYPET_EXPR_H
#define _PYPET_EXPR_H

#include "pypoly/core/pypet/pypet.h"

#include <isl/aff.h>
#include <isl/ctx.h>
#include <isl/id.h>
#include <isl/map.h>
#include <isl/set.h>
#include <isl/space.h>
#include <isl/union_map.h>
#include <isl/union_set.h>
#include <isl/val.h>

namespace pypoly {
namespace pypet {

enum PypetExprType {
  PYPET_EXPR_ERROR = -1,
  PYPET_EXPR_ACCESS,
  PYPET_EXPR_CALL,
  PYPET_EXPR_OP,
};

enum PypetOpType {
  PYPET_ASSIGN = 0,  // Tensor operation leads to an assignment.
  PYPET_ADD,
  PYPET_SUB,
  PYPET_MUL,
  PYPET_DIV,
  PYPET_MOD,
  PYPET_EQ,
  PYPET_NE,
  PYPET_LE,
  PYPET_GE,
  PYPET_LT,
  PYPET_GT,
  PYPET_AND,
  PYPET_XOR,
  PYPET_OR,
  PYPET_NOT,
};

constexpr const char* OpToString[] = {
    [PYPET_ASSIGN] = "=", [PYPET_ADD] = "+",  [PYPET_SUB] = "-",
    [PYPET_MUL] = "*",    [PYPET_DIV] = "/",  [PYPET_MOD] = "%",
    [PYPET_EQ] = "==",    [PYPET_NE] = "not", [PYPET_LE] = "<=",
    [PYPET_GE] = ">=",    [PYPET_LT] = "<",   [PYPET_GT] = ">",
    [PYPET_AND] = "and",  [PYPET_XOR] = "^",  [PYPET_OR] = "or",
    [PYPET_NOT] = "not",
};

enum PypetExprAccessType {
  // TODO(Ying) check whether we needs so many access types or not, but only
  // MUST_READ/WRITE relations?
  PYPET_EXPR_ACCESS_MAY_READ = 0,
  PYPET_EXPR_ACCESS_BEGIN = PYPET_EXPR_ACCESS_MAY_READ,
  PYPET_EXPR_ACCESS_FAKE_KILL = PYPET_EXPR_ACCESS_MAY_READ,
  PYPET_EXPR_ACCESS_MAY_WRITE,
  PYPET_EXPR_ACCESS_MUST_WRITE,
  PYPET_EXPR_ACCESS_END,
  PYPET_EXPR_ACCESS_KILL,
};

struct PypetExprAccess {
  PypetExprAccess() = default;
  ~PypetExprAccess() = default;

  isl_id* ref_id;
  isl_multi_pw_aff* index;  // index expression.
  int depth;
  size_t read;
  size_t write;
  size_t kill;
  isl_union_map* access[PYPET_EXPR_ACCESS_END];  // access relation.
};

enum PypetArgType {
  PYPET_ARG_INT,
  PYPET_ARG_TENSOR,
  PYPET_ARG_ARRAY,
  PYPET_ARG_OTHER,  // int, float, etc. other numeric types.
  // TODO(Ying): Do we need more argument types?
};

struct PypetFuncSummaryArg {
  enum PypetArgType type;

  union {
    isl_id* id;
    isl_union_map* access[PYPET_EXPR_ACCESS_END];
  };
};

struct PypetFuncSummary {
  int ref;
  isl_ctx* ctx;

  size_t n;  // the number of arguments.

  struct PypetFuncSummaryArg arg[];
};

struct PypetExprCall {
  char* name;
  PypetFuncSummary* summary;

  PypetExprCall() = default;
  ~PypetExprCall() = default;
};

struct PypetExpr {
  friend PypetExprAccess;

  PypetExpr() = default;
  ~PypetExpr() = default;

  int ref;
  isl_ctx* ctx;

  enum PypetExprType type;

  int type_size;

  size_t arg_num;
  PypetExpr** args;

  union {
    struct PypetExprAccess acc;
    enum PypetOpType op;

    struct PypetExprCall call;

    char* type_name;
    isl_val* i;
  };
};

__isl_null PypetExpr* PypetExprFree(__isl_take PypetExpr* expr);

}  // namespace pypet
}  // namespace pypoly

#endif
