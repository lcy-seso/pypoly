#include "pypoly/core/pypet/aff.h"

namespace pypoly {
namespace pypet {

namespace {

struct PypetUnionMapMoveDimsData {
  enum isl_dim_type dst_type;
  unsigned dst_pos;
  enum isl_dim_type src_type;
  unsigned src_pos;
  unsigned n;

  isl_union_map* ret;
};

isl_stat MapMoveDims(isl_map* map, void* user) {
  PypetUnionMapMoveDimsData* data =
      static_cast<PypetUnionMapMoveDimsData*>(user);
  map = isl_map_move_dims(map, data->dst_type, data->dst_pos, data->src_type,
                          data->src_pos, data->n);
  data->ret = isl_union_map_add_map(data->ret, map);
  return isl_stat_ok;
}

}  // namespace

isl_multi_aff* PypetPrefixProjection(isl_space* space, int n) {
  int dim = isl_space_dim(space, isl_dim_set);
  return isl_multi_aff_project_out_map(space, isl_dim_set, n, dim - n);
}

isl_union_map* PypetUnionMapMoveDims(isl_union_map* umap,
                                     enum isl_dim_type dst_type,
                                     unsigned dst_pos,
                                     enum isl_dim_type src_type,
                                     unsigned src_pos, unsigned n) {
  PypetUnionMapMoveDimsData data = {dst_type, dst_pos, src_type,
                                    src_pos,  n,       nullptr};
  isl_space* space = isl_union_map_get_space(umap);
  if (src_type == isl_dim_param) {
    space = isl_space_drop_dims(space, src_type, src_pos, n);
  }
  data.ret = isl_union_map_empty(space);
  CHECK_GE(isl_union_map_foreach_map(umap, &MapMoveDims, &data), 0);
  isl_union_map_free(umap);
  return data.ret;
}

}  // namespace pypet
}  // namespace pypoly
