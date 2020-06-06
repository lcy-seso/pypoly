#ifndef PYPOLY_PYPET_NEST_H_
#define PYPOLY_PYPET_NEST_H_

#include "pypoly/core/pypet/expr.h"

namespace pypoly {
namespace pypet {

bool PypetNestedInId(isl_id* id);

bool PypetNestedInSpace(isl_space* space, int pos);

bool PypetNestedInSet(isl_set* set, int pos);

isl_space* PypetNestedRemoveFromSpace(isl_space* space);

isl_set* PypetNestedRemoveFromSet(isl_set* set);

isl_id* PypetNestedPypetExpr(PypetExpr* expr);

bool PypetNestedAnyInSpace(isl_space* space);

PypetExpr* PypetNestedExtractExpr(isl_id* id);

bool PypetNestedAnyInPwAff(isl_pw_aff* pa);

PypetStmt* PypetStmtResolveNested(PypetStmt* stmt);

PypetScop* PypetScopResolveNested(PypetScop* scop);

}  // namespace pypet
}  // namespace pypoly

#endif  // PYPOLY_PYPET_NEST_H_
