#ifndef _PYPET_TREE2SCOP_H
#define _PYPET_TREE2SCOP_H

#include "pypoly/core/pypet/pypet.h"
#include "pypoly/core/pypet/tree.h"

#include <isl/id_to_pw_aff.h>
#include <isl/set.h>
#include <isl/space.h>

namespace pypoly {
namespace pypet {

struct PypetState;

struct PypetContext {
  PypetContext();
  ~PypetContext() = default;

  int ref;
  isl_set* domain;
  // for each parameter in "tree". Parameters any integer variable that is read
  // anywhere in "tree" or in any of the size expressions for any of the arrays
  // accessed in "tree".
  isl_id_to_pw_aff* assignments;
  bool allow_nested;
  std::map<PypetExpr*, isl_pw_aff*> extracted_affine;
};

__isl_give PypetContext* CreatePypetContext(__isl_take isl_set* domain);
__isl_null PypetContext* FreePypetContext(__isl_take PypetContext* pc);
__isl_give PypetContext* PypetContextAddParameter(__isl_keep PypetTree* tree,
                                                  __isl_keep PypetContext* pc);

// construct SCoP from PypetTree
struct TreeToScop {
  TreeToScop(isl_ctx* ctx) : ctx(ctx){};
  ~TreeToScop() = default;

  __isl_give PypetScop* ScopFromTree(__isl_keep PypetTree* tree);

 private:
  isl_ctx* ctx;

  __isl_keep PypetScop* ToScop(__isl_take PypetTree* tree,
                               __isl_take PypetContext* pc,
                               __isl_take PypetState* state);
  __isl_keep PypetScop* ScopFromBlock(__isl_keep PypetTree* tree,
                                      __isl_keep PypetContext* pc,
                                      __isl_take PypetState* state);
  __isl_keep PypetScop* ScopFromBreak(__isl_keep PypetTree* tree,
                                      __isl_keep PypetContext* pc,
                                      __isl_take PypetState* state);
  __isl_keep PypetScop* ScopFromContinue(__isl_keep PypetTree* tree,
                                         __isl_keep PypetContext* pc,
                                         __isl_take PypetState* state);
  __isl_keep PypetScop* ScopFromDecl(__isl_keep PypetTree* tree,
                                     __isl_keep PypetContext* pc,
                                     __isl_take PypetState* state);
  __isl_keep PypetScop* ScopFromTreeExpr(__isl_keep PypetTree* tree,
                                         __isl_keep PypetContext* pc,
                                         __isl_take PypetState* state);
  __isl_keep PypetScop* ScopFromReturn(__isl_keep PypetTree* tree,
                                       __isl_keep PypetContext* pc,
                                       __isl_take PypetState* state);
  __isl_keep PypetScop* ScopFromIf(__isl_keep PypetTree* tree,
                                   __isl_keep PypetContext* pc,
                                   __isl_take PypetState* state);
  __isl_keep PypetScop* ScopFromFor(__isl_keep PypetTree* tree,
                                    __isl_keep PypetContext* pc,
                                    __isl_take PypetState* state);

  __isl_keep PypetScop* ScopFromAffineFor(__isl_keep PypetTree* tree,
                                          __isl_take isl_pw_aff* init_val,
                                          __isl_take isl_pw_aff* pa_inc,
                                          __isl_take isl_val* inc,
                                          __isl_take PypetContext* pc,
                                          __isl_take PypetState* state);

  __isl_keep PypetScop* ScopFromAffineForInit(__isl_keep PypetTree* tree,
                                              __isl_take isl_pw_aff* init_val,
                                              __isl_take isl_pw_aff* pa_inc,
                                              __isl_take isl_val* inc,
                                              __isl_keep PypetContext* pc_init,
                                              __isl_take PypetContext* pc,
                                              __isl_take PypetState* state);
};

}  // namespace pypet
}  // namespace pypoly
#endif
