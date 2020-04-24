#ifndef PYPET_H
#define PYPET_H

#include <isl/aff.h>
#include <isl/arg.h>
#include <isl/map.h>
#include <isl/schedule.h>
#include <isl/set.h>
#include <isl/union_map.h>

#include <memory>
#include <string>
#include <vector>

namespace pypet {

struct PypetExpr;
struct PypetTree;

struct SourceLoc {
  SourceLoc() : start_(0), end_(0){};
  ~SourceLoc() = default;

 private:
  size_t start_;
  size_t end_;
};

struct PypetArray {
  PypetArray(){};
  ~PypetArray() = default;

 private:
  isl_set* context;
  isl_set* extent;
  isl_set* value_bounds;
  std::string element_type;

  /* TODO(Ying): copy from pet, to check.
  int element_is_record_;
  int element_size_;
  int live_out_;
  int uniquely_defined_;
  int declared_;
  int exposed_;
  int outer_;
  */
};

// A polyhedral statement.
struct PypetStmt {
  friend PypetTree;

  PypetStmt(){};
  ~PypetStmt() = default;

 private:
  SourceLoc loc;
  isl_set* domain;

  // A polyhedral statement is either an expression statement or a larger
  // statement that contain control part.
  // the subset of the instance set containing instances of this polyhedral
  // statement;
  std::vector<std::shared_ptr<PypetExpr>> args;
  // Information to print the body of the statement in source program.
  std::shared_ptr<PypetTree> body;
};

struct PypetScop {
  friend PypetArray;
  friend PypetStmt;

  PypetScop(){};
  ~PypetScop() = default;

 private:
  SourceLoc loc;

  // program parameters. A unit set.
  isl_set* context;
  isl_set* context_value;

  // the schedule tree.
  isl_schedule* schedule;

  // array declaration
  std::vector<PypetArray> arrays;

  // the statement list.
  // a polyhedral statement may correspond to an expression statement in the
  // source program's AST, a collection of program statements, or, a program
  // statement may be broken up into several polyhedral statements.
  std::vector<PypetStmt> stmts;
};

}  // namespace pypet
#endif
