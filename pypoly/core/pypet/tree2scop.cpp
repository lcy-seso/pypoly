#include "pypoly/core/pypet/tree2scop.h"

#include "pypoly/core/pypet/pypet.h"

namespace pypoly {
namespace pypet {

/*
 * create PypetContext from a given domain.
 */
__isl_give PypetContext* CreatePypetContext(__isl_take isl_set* domain) {
  if (!domain) return nullptr;

  isl_id_to_pw_aff* assignments =
      isl_id_to_pw_aff_alloc(isl_set_get_ctx(domain), 0);

  PypetContext* pc =
      isl_calloc_type(isl_set_get_ctx(domain), struct PypetContext);
  if (!pc) {
    isl_set_free(domain);
    isl_id_to_pw_aff_free(assignments);
    return nullptr;
  }

  pc->ref = 1;
  pc->domain = domain;
  pc->assignments = assignments;
  return pc;
}

/*
 * Free a reference to "pc" and return nullptr.
 */
__isl_null PypetContext* FreePypetContext(__isl_take PypetContext* pc) {
  if (!pc) return nullptr;
  if (--pc->ref > 0) return nullptr;

  isl_set_free(pc->domain);
  isl_id_to_pw_aff_free(pc->assignments);
  free(pc);
  return nullptr;
}

/*
 * Add an assignment to "pc" for each parameter in "tree".
 */
__isl_give PypetContext* PypetContextAddParameter(__isl_keep PypetTree* tree,
                                                  __isl_keep PypetContext* pc) {
  return pc;
}

__isl_keep PypetScop* TreeToScop::ScopFromBlock(__isl_keep PypetTree* tree,
                                                __isl_keep PypetContext* pc) {
  return nullptr;
}

__isl_keep PypetScop* TreeToScop::ScopFromBreak(__isl_keep PypetTree* tree,
                                                __isl_keep PypetContext* pc) {
  return nullptr;
}

__isl_keep PypetScop* TreeToScop::ScopFromContinue(
    __isl_keep PypetTree* tree, __isl_keep PypetContext* pc) {
  return nullptr;
}

__isl_keep PypetScop* TreeToScop::ScopFromDecl(__isl_keep PypetTree* tree,
                                               __isl_keep PypetContext* pc) {
  return nullptr;
}

__isl_keep PypetScop* TreeToScop::ScopFromTreeExpr(
    __isl_keep PypetTree* tree, __isl_keep PypetContext* pc) {
  return nullptr;
}

__isl_keep PypetScop* TreeToScop::ScopFromReturn(__isl_keep PypetTree* tree,
                                                 __isl_keep PypetContext* pc) {
  return nullptr;
}

__isl_keep PypetScop* TreeToScop::ScopFromIf(__isl_keep PypetTree* tree,
                                             __isl_keep PypetContext* pc) {
  return nullptr;
}

__isl_keep PypetScop* TreeToScop::ScopFromFor(__isl_keep PypetTree* tree,
                                              __isl_keep PypetContext* pc) {
  return nullptr;
}

__isl_keep PypetScop* TreeToScop::ToScop(__isl_take PypetTree* tree,
                                         __isl_take PypetContext* pc) {
  struct PypetScop* scop = nullptr;

  if (!tree) return nullptr;

  switch (tree->type) {
    case PYPET_TREE_ERROR:
      return nullptr;
    case PYPET_TREE_BLOCK:
      return ScopFromBlock(tree, pc);
    case PYPET_TREE_BREAK:
      return ScopFromBreak(tree, pc);
    case PYPET_TREE_CONTINUE:
      return ScopFromContinue(tree, pc);
    case PYPET_TREE_DECL:
    case PYPET_TREE_DECL_INIT:
      return ScopFromDecl(tree, pc);
    case PYPET_TREE_EXPR:
      return ScopFromTreeExpr(tree, pc);
    case PYPET_TREE_RETURN:
      return ScopFromReturn(tree, pc);
    case PYPET_TREE_IF:
    case PYPET_TREE_IF_ELSE:
      scop = ScopFromIf(tree, pc);
      break;
    case PYPET_TREE_FOR:
      scop = ScopFromFor(tree, pc);
      break;
  }
  if (!scop) return nullptr;
  return scop;
}

__isl_give PypetScop* TreeToScop::ScopFromTree(__isl_keep PypetTree* tree) {
  // create a universe set as the initial domain.
  isl_set* domain = isl_set_universe(isl_space_set_alloc(ctx, 0, 0));
  // create context with the given domain.
  PypetContext* pc = CreatePypetContext(domain);
  pc = PypetContextAddParameter(tree, pc);

  struct PypetScop* scop = ToScop(tree, pc);
  PypetTreeFree(tree);

  if (scop) {
    // Compute the parameter domain of the given set.
    scop->context = isl_set_params(scop->context);
  }

  FreePypetContext(pc);
  return scop;
}

}  // namespace pypet
}  // namespace pypoly
