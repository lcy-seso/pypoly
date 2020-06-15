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
  // In our analysis, we may have two kinds of array: array whose elements have
  // primary types, like int, float; and tensor arrayi whose elements are
  // tensors with the same shape.
  PypetArray() = delete;
  ~PypetArray() = default;

  isl_set* context;  // set parameters.
  isl_set* extent;   // data element set.
  isl_set* value_bounds;

  // int, float, or a tensor.
  char* element_type;
  // For tensor arry, element_size` is the size of tensor element.
  int element_size;
  // `element_shape` is a list of integer that records shape of the element
  // stored in the array. If array elements are scalars with primary types,
  // element_shape is always equal to : [1],
  size_t element_dim;
  int* element_shape;

  // TODO(Ying): In current implementations, we are not able to distinguish
  // a varaible declaration or a name reuse.
  int declared;

  // TODO(Ying): below information is recorded by pet, but not considered by
  // us in current implementations.
  /*
  int element_is_record;
  int live_out;
  int uniquely_defined;
  int exposed;
  int outer;
  */
};
struct ArrayPrettyPrinter {
  static __isl_give isl_printer* Print(__isl_take isl_printer* p,
                                       const PypetArray* array);
  static void Print(std::ostream& out, const PypetArray* array, int indent = 0);
};
static inline std::ostream& operator<<(std::ostream& out,
                                       const PypetArray* array) {
  ArrayPrettyPrinter::Print(out, array);
  return out;
};

// A polyhedral statement.
struct PypetStmt {
  PypetStmt() = delete;
  ~PypetStmt() = default;

  static PypetStmt* Create(isl_set* domain, int id, PypetTree* tree);
  static __isl_null PypetStmt* Free(__isl_take PypetStmt* stmt);

  torch::jit::SourceRange range;
  isl_set* domain;

  // A polyhedral statement is either an expression statement or a larger
  // statement that contain control part.
  // the subset of the instance set containing instances of this polyhedral
  // statement;
  size_t arg_num;
  PypetExpr** args;
  // Information to print the body of the statement in source program.
  PypetTree* body;
};
isl_set* StmtExtractContext(PypetStmt* stmt, isl_set* context);

struct StmtPrettyPrinter {
  static __isl_give isl_printer* Print(__isl_take isl_printer* p,
                                       const PypetStmt* stmt);
  static void Print(std::ostream& out, const PypetStmt* stmt, int indent = 0);
};
static inline std::ostream& operator<<(std::ostream& out,
                                       const PypetStmt* stmt) {
  StmtPrettyPrinter::Print(out, stmt);
  return out;
};

struct PypetScop {
  PypetScop() = delete;
  ~PypetScop() = default;

  static PypetScop* Create(isl_space* space);
  static PypetScop* Create(isl_space* space, PypetStmt* stmt);
  static PypetScop* Create(isl_space* space, int n, isl_schedule* schedule);
  static __isl_null PypetScop* Free(__isl_take PypetScop* scop);

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

struct ScopPrettyPrinter {
  static __isl_give isl_printer* Print(__isl_take isl_printer* p,
                                       const PypetScop* scop);
  static void Print(std::ostream& out, const PypetScop* scop, int indent = 0);
};
static inline std::ostream& operator<<(std::ostream& out,
                                       const PypetScop* scop) {
  ScopPrettyPrinter::Print(out, scop);
  return out;
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
