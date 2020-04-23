#ifndef PYPET_H
#define PYPET_H

#include "pypet/core/expr.h"

#include <isl/aff.h>
#include <isl/arg.h>
#include <isl/map.h>
#include <isl/schedule.h>
#include <isl/set.h>
#include <isl/union_map.h>

#include <string>
#include <vector>

namespace pypet {

struct SourceLoc {
  SourceLoc() = default;
  ~SourceLoc() = default;

 private:
  size_t start_;
  size_t end_;
};

struct PypetArray {
  PypetArray(){};
  ~PypetArray(){};

 private:
  isl_set* context_;
  isl_set* extent_;
  isl_set* value_bounds_;
  std::string element_type_;

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

struct PypetStmt {
  PypetStmt(){};
  ~PypetStmt(){};

 private:
  SourceLoc loc_;
  isl_set* domain_;
  PypetTree* body_;

  std::vector<PypetExpr> args_;
};

struct PypetScop {
  friend PypetArray;

  PypetScop(){};
  ~PypetScop();

 private:
  SourceLoc loc_;

  // program parameters. A unit set.
  isl_set* context_;
  isl_set* context_value_;

  // the schedule tree.
  isl_schedule* schedule_;

  // array declaration
  std::vector<PypetArray> arrays_;

  // the statement list.
  std::vector<PypetStmt> stmts_;
};

}  // namespace pypet
#endif
