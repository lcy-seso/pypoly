#ifndef PYPOLY_CORE_PYPET_EXPR_H_
#define PYPOLY_CORE_PYPET_EXPR_H_

#include "pypoly/core/pypet/pypet.h"

namespace pypoly {
namespace pypet {

enum PypetExprType {
  PYPET_EXPR_ERROR = -1,
  PYPET_EXPR_ACCESS,
  PYPET_EXPR_CALL,
  PYPET_EXPR_OP,
  PYPET_EXPR_INT,
};

enum PypetOpType {
  PYPET_ASSIGN = 0,  // Tensor operation leads to an assignment.
  PYPET_ADD,
  PYPET_SUB,
  PYPET_MUL,
  PYPET_DIV,
  PYPET_MOD,
  PYPET_MINUS,
  PYPET_EQ,
  PYPET_NE,
  PYPET_LE,
  PYPET_GE,
  PYPET_LT,
  PYPET_GT,
  PYPET_COND,
  PYPET_AND,
  PYPET_XOR,
  PYPET_OR,
  PYPET_NOT,
  PYPET_APPLY,
  PYPET_LIST_LITERAL,
  PYPET_ATTRIBUTE,
  PYPET_UNKNOWN,
};

static constexpr const char* op_type_to_string[] = {
    [PYPET_ASSIGN] = "=",
    [PYPET_ADD] = "+",
    [PYPET_SUB] = "-",
    [PYPET_MUL] = "*",
    [PYPET_DIV] = "/",
    [PYPET_MOD] = "%",
    [PYPET_MINUS] = "-",
    [PYPET_EQ] = "==",
    [PYPET_NE] = "!=",
    [PYPET_LE] = "<=",
    [PYPET_GE] = ">=",
    [PYPET_LT] = "<",
    [PYPET_GT] = ">",
    [PYPET_COND] = "?:",
    [PYPET_AND] = "&",
    [PYPET_XOR] = "^",
    [PYPET_OR] = "or",
    [PYPET_NOT] = "not",
    [PYPET_APPLY] = "apply",
    [PYPET_LIST_LITERAL] = "[]",
    [PYPET_ATTRIBUTE] = "attribute",
};

enum PypetExprAccessType {
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
  PypetFuncSummaryArg() = default;
  ~PypetFuncSummaryArg() = default;

  enum PypetArgType type;

  union {
    isl_id* id;
    isl_union_map* access[PYPET_EXPR_ACCESS_END];
  };
};

struct PypetFuncSummary {
  PypetFuncSummary() = default;
  ~PypetFuncSummary() = default;

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
  PypetExpr() = default;
  ~PypetExpr() = default;

  bool IsComparison();
  bool IsBoolean();
  bool IsMin();
  bool IsMax();
  bool HasRelevantAccessRelation();

  PypetExpr* Dup();
  PypetExpr* Cow();

  PypetExpr* RemoveDuplicateArgs();
  bool IsEqual(PypetExpr* rhs);
  PypetExpr* EquateArg(int i, int j);

  PypetExpr* AccessPullbackMultiAff(isl_multi_aff* multi_aff);
  PypetExpr* AccessPullbackMultiPwAff(isl_multi_pw_aff* multi_pw_aff);
  PypetExpr* AccessProjectOutArg(int dim, int pos);

  PypetExpr* PlugIn(int pos, isl_pw_aff* value);

  int ref;
  isl_ctx* ctx;

  uint32_t hash;

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

__isl_give PypetExpr* PypetExprAlloc(isl_ctx* ctx, PypetExprType expr_type);

__isl_null PypetExpr* PypetExprFree(__isl_take PypetExpr* expr);

__isl_keep PypetExpr* PypetExprDup(__isl_keep PypetExpr* expr);

__isl_keep PypetExpr* PypetExprCow(__isl_keep PypetExpr* expr);

PypetExpr* PypetExprNewBinary(int type_size, PypetOpType type, PypetExpr* lhs,
                              PypetExpr* rhs);

PypetExpr* PypetExprNewTernary(PypetExpr* p, PypetExpr* q, PypetExpr* r);

PypetExpr* PypetExprSetTypeSize(PypetExpr* expr, int type_size);

__isl_keep PypetExpr* PypetExprFromIslVal(__isl_keep isl_val* val);

__isl_keep PypetExpr* PypetExprFromIntVal(__isl_keep isl_ctx* ctx, long val);

__isl_give PypetExpr* PypetExprCreateCall(isl_ctx* ctx, const char* name,
                                          size_t arg_num);

PypetExpr* PypetExprAccessSetIndex(PypetExpr* expr, isl_multi_pw_aff* index);

PypetExpr* PypetExprFromIndex(isl_multi_pw_aff* index);

PypetExpr* PypetExprSetNArgs(PypetExpr* expr, int n);

PypetExpr* PypetExprCopy(PypetExpr* expr);

PypetExpr* PypetExprGetArg(PypetExpr* expr, int pos);

PypetExpr* PypetExprSetArg(PypetExpr* expr, int pos, PypetExpr* arg);

isl_space* PypetExprAccessGetAugmentedDomainSpace(PypetExpr* expr);

isl_space* PypetExprAccessGetDomainSpace(PypetExpr* expr);

PypetExpr* PypetExprAccessPullbackMultiAff(PypetExpr* expr,
                                           isl_multi_aff* multi_aff);

PypetExpr* PypetExprAccessMoveDims(PypetExpr* expr, enum isl_dim_type dst_type,
                                   unsigned dst_pos, enum isl_dim_type src_type,
                                   unsigned src_pos, unsigned n);

PypetExpr* PypetExprAccessAlignParams(PypetExpr* expr);

bool PypetExprAccessHasAnyAccessRelation(PypetExpr* expr);

bool PypetExprIsSubAccess(PypetExpr* lhs, PypetExpr* rhs, int arg_num);

isl_union_map* ConstructAccessRelation(PypetExpr* expr);

isl_map* ExtendRange(isl_map* access, int n);

PypetExpr* IntroduceAccessRelations(PypetExpr* expr);

PypetExpr* PypetExprInsertArg(PypetExpr* expr, int pos, PypetExpr* arg);

isl_multi_pw_aff* PypetArraySubscript(isl_multi_pw_aff* base,
                                      isl_pw_aff* index);

PypetExpr* PypetExprAccessSubscript(PypetExpr* expr, PypetExpr* index);

PypetExpr* BuildPypetBinaryOpExpr(isl_ctx* ctx, PypetOpType op_type,
                                  PypetExpr* lhs, PypetExpr* rhs);

char* PypetArrayMemberAccessName(isl_ctx* ctx, const char* base,
                                 const char* field);

isl_multi_pw_aff* PypetArrayMember(isl_multi_pw_aff* base,
                                   isl_multi_pw_aff* field);

PypetExpr* PypetExprAccessMember(PypetExpr* expr, isl_id* id);

int PypetExprForeachExprOfType(PypetExpr* expr, PypetExprType type,
                               const std::function<int(PypetExpr*, void*)>& fn,
                               void* user);

int PypetExprForeachAccessExpr(PypetExpr* expr,
                               const std::function<int(PypetExpr*, void*)>& fn,
                               void* user);

int PypetExprIsScalarAccess(PypetExpr* expr);

bool PypetExprIsAffine(PypetExpr* expr);

isl_pw_aff* PypetExprExtractComparison(PypetOpType type, PypetExpr* lhs,
                                       PypetExpr* rhs, PypetContext* context);

isl_pw_aff* PypetExprExtractAffineCondition(PypetExpr* expr,
                                            PypetContext* context);

isl_pw_aff* PypetExprExtractAffine(PypetExpr* expr, PypetContext* context);

isl_pw_aff* PypetExprGetAffine(PypetExpr* expr);

PypetExpr* PypetExprMapExprOfType(
    PypetExpr* expr, PypetExprType type,
    const std::function<PypetExpr*(PypetExpr*, void*)>& fn, void* user);

PypetExpr* PypetExprMapAccess(
    PypetExpr* expr, const std::function<PypetExpr*(PypetExpr*, void*)>& fn,
    void* user);

PypetExpr* PypetExprMapTopDown(
    PypetExpr* expr, const std::function<PypetExpr*(PypetExpr*, void*)>& fn,
    void* user);

int PypetExprWrites(PypetExpr* expr, isl_id* id);

isl_id* PypetExprAccessGetId(PypetExpr* expr);

isl_pw_aff* NonAffine(isl_space* space);

isl_space* PypetExprAccessGetParameterSpace(PypetExpr* expr);

isl_ctx* PypetExprGetCtx(PypetExpr* expr);

PypetExpr* PypetExprInsertDomain(PypetExpr* expr, isl_space* space);

PypetExpr* PypetExprUpdateDomain(PypetExpr* expr, isl_multi_pw_aff* update);

PypetExpr* PypetExprAccessUpdateDomain(PypetExpr* expr,
                                       isl_multi_pw_aff* update);

PypetExpr* PypetExprRestrict(PypetExpr* expr, isl_set* set);

/* Does "expr" represent the "integer" infinity?*/
inline bool IsInftyVal(__isl_keep PypetExpr* expr) {
  CHECK(expr);

  if (expr->type != PYPET_EXPR_INT) return false;
  isl_val* val = isl_val_copy(expr->i);
  isl_bool res = isl_val_is_infty(val);
  isl_val_free(val);

  return res;
};

struct ExprPrettyPrinter {
  static void Print(std::ostream& out, const PypetExpr* expr, int indent = 0);
  static __isl_give isl_printer* Print(__isl_take isl_printer* p,
                                       const PypetExpr* expr);

  static __isl_give isl_printer* PrintArguments(
      const __isl_keep PypetExpr* expr, __isl_take isl_printer* p);

  static __isl_give isl_printer* PrintFuncSummary(
      const __isl_keep PypetFuncSummary* summary, __isl_take isl_printer* p);
};

static inline std::ostream& operator<<(std::ostream& out,
                                       const PypetExpr* expr) {
  ExprPrettyPrinter::Print(out, expr);
  return out;
}
}  // namespace pypet
}  // namespace pypoly

#endif  // PYPOLY_CORE_PYPET_EXPR_H_
