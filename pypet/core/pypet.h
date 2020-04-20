#ifndef PYPET_H
#define PYPET_H

#include <isl/aff.h>
#include <isl/arg.h>
#include <isl/map.h>
#include <isl/schedule.h>
#include <isl/set.h>
#include <isl/union_map.h>

namespace pypet {

struct PypetScop {
  PypetScop(){};
  ~PypetScop() = default;

 private:
  isl_set* context;
  isl_set* context_value;
  isl_schedule* schedule;

  // array and statements.
};
}  // namespace pypet
#endif
