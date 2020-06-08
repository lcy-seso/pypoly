#ifndef PYPOLY_CORE_PYPET_NEST_H_
#define PYPOLY_CORE_PYPET_NEST_H_

#include "pypoly/core/pypet/expr.h"

namespace pypoly {
namespace pypet {

bool PypetNestedInId(isl_id* id);

bool PypetNestedInSpace(isl_space* space, int pos);

bool PypetNestedInSet(isl_set* set, int pos);

bool PypetNestedInMap(isl_map* map, int pos);

bool PypetNestedInUnionMap(isl_union_map* umap, int pos);

isl_space* PypetNestedRemoveFromSpace(isl_space* space);

isl_set* PypetNestedRemoveFromSet(isl_set* set);

isl_id* PypetNestedPypetExpr(PypetExpr* expr);

bool PypetNestedAnyInSpace(isl_space* space);

int PypetNestedNInSet(isl_set* set);

int PypetNestedNInSpace(isl_space* space);

PypetExpr* PypetNestedExtractExpr(isl_id* id);

bool PypetNestedAnyInPwAff(isl_pw_aff* pa);

int PypetExprDomainDim(PypetExpr* expr);

PypetExpr* Embed(PypetExpr* expr, isl_space* space);

int PypetExtractNestedFromSpace(isl_space* space, int arg_num, PypetExpr** args,
                                int* param2pos);

PypetExpr* PypetExprExtractNested(PypetExpr* expr, int n, int* param2pos);

PypetExpr* PypetExprResolveNested(PypetExpr* expr, isl_space* domain);

PypetTree* PypetTreeResolveNested(PypetTree* tree, isl_space* space);

PypetStmt* PypetStmtExtractNested(PypetStmt* stmt, int n, int* param2pos);

PypetStmt* PypetStmtResolveNested(PypetStmt* stmt);

PypetScop* PypetScopResolveNested(PypetScop* scop);

PypetExpr* MarkSelfDependences(PypetExpr* expr, int first);

PypetExpr* RemoveMarkedSelfDependences(PypetExpr* expr, int dim, int first);

}  // namespace pypet
}  // namespace pypoly

#endif  // PYPOLY_PYPET_NEST_H_
