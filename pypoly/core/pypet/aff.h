#ifndef PYPOLY_CORE_PYPET_AFF_H_
#define PYPOLY_CORE_PYPET_AFF_H_

#include "pypoly/core/pypet/util.h"

namespace pypoly {
namespace pypet {

isl_multi_aff* PypetPrefixProjection(isl_space* space, int n);

isl_union_map* PypetUnionMapMoveDims(isl_union_map* umap,
                                     enum isl_dim_type dst_type,
                                     unsigned dst_pos,
                                     enum isl_dim_type src_type,
                                     unsigned src_pos, unsigned n);

isl_val* PypetExtractCst(isl_pw_aff* pw_aff);

}  // namespace pypet
}  // namespace pypoly

#endif  // PYPOLY_PYPET_AFF_H_
