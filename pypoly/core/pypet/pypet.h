#ifndef _PYPET_H
#define _PYPET_H

#include <isl/aff.h>
#include <isl/arg.h>
#include <isl/map.h>
#include <isl/schedule.h>
#include <isl/set.h>
#include <isl/union_map.h>
#include <torch/csrc/jit/frontend/source_range.h>

#include <memory>
#include <vector>

namespace pypoly {
namespace pypet {

struct PypetScop;
using PypetScopPtr = std::shared_ptr<PypetScop>;

struct PypetExpr;
struct PypetTree;

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

  torch::jit::SourceRange range;
  isl_set* domain;

  // A polyhedral statement is either an expression statement or a larger
  // statement that contain control part.
  // the subset of the instance set containing instances of this polyhedral
  // statement;
  PypetExpr** args;
  // Information to print the body of the statement in source program.
  PypetTree* body;
};

struct PypetScop {
  friend PypetArray;
  friend PypetStmt;

  PypetScop() = default;
  ~PypetScop() = default;

  // program parameters. A unit set.
  isl_set* context;
  isl_set* context_value;

  // the schedule tree.
  isl_schedule* schedule;

  // array declaration
  PypetArray** arrays;

  // the statement list.
  // a polyhedral statement may correspond to an expression statement in the
  // source program's AST, a collection of program statements, or, a program
  // statement may be broken up into several polyhedral statements.
  PypetStmt** stmts;
};

}  // namespace pypet
}  // namespace pypoly
#endif
