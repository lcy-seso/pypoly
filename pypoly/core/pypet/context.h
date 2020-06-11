#ifndef PYPOLY_CORE_PYPET_CONTEXT_H_
#define PYPOLY_CORE_PYPET_CONTEXT_H_

#include "pypoly/core/pypet/pypet.h"

namespace pypoly {
namespace pypet {

struct PypetContext {
  PypetContext();
  ~PypetContext() = default;

  int ref;
  isl_set* domain;
  // for each parameter in "tree". Parameters any integer variable that is read
  // anywhere in "tree" or in any of the size expressions for any of the arrays
  // accessed in "tree".
  bool allow_nested;
  std::map<isl_id*, isl_pw_aff*> assignments;
  std::map<PypetExpr*, isl_pw_aff*> extracted_affine;
};

__isl_give PypetContext* CreatePypetContext(__isl_take isl_set* domain);

__isl_null PypetContext* FreePypetContext(__isl_take PypetContext* pc);

__isl_give PypetContext* PypetContextAddParameter(__isl_keep PypetTree* tree,
                                                  __isl_keep PypetContext* pc);

PypetContext* ContextAlloc(isl_set* domain, bool allow_nested);

PypetContext* PypetContextDup(PypetContext* context);

PypetContext* PypetContextCopy(PypetContext* context);

PypetContext* PypetContextCow(PypetContext* context);

int PypetContextDim(PypetContext* pc);

isl_space* PypetContextGetSpace(PypetContext* context);

int PypetContextGetDim(PypetContext* context);

isl_set* PypetContextGetDomain(PypetContext* context);

PypetContext* ExtendDomain(PypetContext* context, isl_id* id);

PypetContext* PypetContextIntersectDomain(PypetContext* pc, isl_set* domain);

PypetContext* PypetContextSetValue(PypetContext* context, isl_id* id,
                                   isl_pw_aff* pw_aff);

PypetContext* PypetContextClearValue(PypetContext* context, isl_id* id);

isl_pw_aff* PypetContextGetValue(PypetContext* context, isl_id* id);

PypetContext* PypetContextAddInnerIterator(PypetContext* context, isl_id* id);

PypetContext* PypetContextClearWritesInTree(PypetContext* context,
                                            PypetTree* tree);

bool PypetContextIsAssigned(PypetContext* context, isl_id* id);

PypetContext* PypetContextSetAllowNested(PypetContext* context, bool val);

PypetExpr* PypetContextEvaluateExpr(PypetContext* context, PypetExpr* expr);

PypetTree* PypetContextEvaluateTree(PypetContext* pc, PypetTree* tree);

}  // namespace pypet
}  // namespace pypoly

#endif  // PYPOLY_CORE_PYPET_CONTEXT_H_
