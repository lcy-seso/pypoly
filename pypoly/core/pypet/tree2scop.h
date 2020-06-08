#ifndef _PYPET_TREE2SCOP_H
#define _PYPET_TREE2SCOP_H

#include "pypoly/core/pypet/pypet.h"
#include "pypoly/core/pypet/tree.h"

#include <isl/id_to_pw_aff.h>
#include <isl/set.h>
#include <isl/space.h>

namespace pypoly {
namespace pypet {

struct PypetState {
  isl_ctx* ctx;
  void* user;
  int int_size;

  int loop_num;
  int stmt_num;
  int test_num;
};

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

  PypetScop* ScopFromAffineIf(PypetTree* tree, isl_pw_aff* cond,
                              PypetContext* pc, PypetState* state);

  PypetScop* ScopFromEvaluatedTree(PypetTree* tree, int stmt_num,
                                   PypetContext* pc);

  PypetScop* ScopFromUnevaluatedTree(PypetTree* tree, int stmt_num,
                                     PypetContext* pc);

  PypetScop* ScopFromConditionalAssignment(PypetTree* tree,
                                           isl_pw_aff* cond_pw_aff,
                                           PypetContext* pc, PypetState* state);
};

}  // namespace pypet
}  // namespace pypoly
#endif
