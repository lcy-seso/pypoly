#ifndef PYPOLY_CORE_PYPET_PYPET_H_
#define PYPOLY_CORE_PYPET_PYPET_H_

#include "util.h"

#include <torch/csrc/jit/frontend/source_range.h>

namespace pypoly {
namespace pypet {

struct PypetScop;
struct PypetExpr;
struct PypetTree;
struct PypetContext;
struct PypetState;

struct PypetArray {
  PypetArray(){};
  ~PypetArray() = default;

  isl_set* context;
  isl_set* extent;
  isl_set* value_bounds;
  char* element_type;

  /* TODO(Ying): copy from pet, to check.
  int element_is_record;
  int element_size;
  int live_out;
  int uniquely_defined;
  int declared;
  int exposed;
  int outer;
  */
};

// A polyhedral statement.
struct PypetStmt {
  friend PypetTree;

  PypetStmt(const torch::jit::SourceRange& range)
      : range(range), domain(nullptr), args(nullptr), body(nullptr){};
  ~PypetStmt() = default;

  static PypetStmt* Create(isl_set* domain, int id, PypetTree* tree);

  torch::jit::SourceRange range;
  isl_set* domain;

  // A polyhedral statement is either an expression statement or a larger
  // statement that contain control part.
  // the subset of the instance set containing instances of this polyhedral
  // statement;
  int arg_num;
  PypetExpr** args;
  // Information to print the body of the statement in source program.
  PypetTree* body;
};

isl_set* StmtExtractContext(PypetStmt* stmt, isl_set* context);

struct PypetScop {
  friend PypetArray;
  friend PypetStmt;

  PypetScop() = delete;
  ~PypetScop() = default;

  static PypetScop* Create(isl_space* space);
  static PypetScop* Create(isl_space* space, PypetStmt* stmt);
  static PypetScop* Create(isl_space* space, int n, isl_schedule* schedule);

  torch::jit::SourceRange range;

  // program parameters. A unit set.
  isl_set* context;
  isl_set* context_value;

  // the schedule tree.
  isl_schedule* schedule;

  // array declaration
  int array_num;
  PypetArray** arrays;

  // the statement list.
  // a polyhedral statement may correspond to an expression statement in the
  // source program's AST, a collection of program statements, or, a program
  // statement may be broken up into several polyhedral statements.
  int stmt_num;
  PypetStmt** stmts;
};

PypetScop* PypetScopAdd(isl_ctx* ctx, isl_schedule* schedule, PypetScop* lhs,
                        PypetScop* rhs);

PypetScop* PypetScopAddPar(isl_ctx* ctx, PypetScop* lhs, PypetScop* rhs);

PypetScop* PypetScopAddSeq(isl_ctx* ctx, PypetScop* lhs, PypetScop* rhs);

PypetScop* PypetScopEmbed(PypetScop* scop, isl_set* dom,
                          isl_multi_aff* schedule);

inline PypetScop* ScopCollectImplications(isl_ctx* ctx, PypetScop* scop,
                                          PypetScop* lhs, PypetScop* rhs) {
  // TODO
  return scop;
}

PypetScop* PypetScopRestrict(PypetScop* scop, isl_set* cond);

PypetScop* PypetScopRestrictContext(PypetScop* scop, isl_set* context);

inline PypetScop* PypetScopCombineSkips(PypetScop* scop, PypetScop* lhs,
                                        PypetScop* rhs) {
  // TODO
  return scop;
}

inline PypetScop* PypetScopCombineStartEnd(PypetScop* scop, PypetScop* lhs,
                                           PypetScop* rhs) {
  // TODO
  return scop;
}

inline PypetScop* PypetScopCollectIndependence(isl_ctx* ctx, PypetScop* scop,
                                               PypetScop* lhs, PypetScop* rhs) {
  // TODO
  return scop;
}

}  // namespace pypet
}  // namespace pypoly
#endif  // PYPOLY_CORE_PYPET_PYPET_H_
