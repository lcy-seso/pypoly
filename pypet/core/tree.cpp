#include "pypet/core/pypet.h"

namespace pypet {

__isl_give PypetTree* PypetTree::PypetTreeAlloc(isl_ctx* ctx,
                                                torch::jit::SourceRange range,
                                                enum PypetTreeType) {
  PypetTree* tree;

  tree = isl_calloc_type(ctx, struct PypetTree);
  if (!tree) return nullptr;

  tree->ctx = ctx;
  isl_ctx_ref(ctx);
  tree->ref = 1;
  tree->type = type;
  // tree->loc = &pet_loc_dummy;

  return tree;
}

}  // namespace pypet
