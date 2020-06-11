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

isl_stat ExtractCst(isl_set* set, isl_aff* aff, void* user) {
  isl_val** inc = static_cast<isl_val**>(user);
  if (isl_aff_is_cst(aff)) {
    isl_val_free(*inc);
    *inc = isl_aff_get_constant_val(aff);
  }
  isl_set_free(set);
  isl_aff_free(aff);
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

isl_val* PypetExtractCst(isl_pw_aff* pw_aff) {
  CHECK(pw_aff);
  isl_val* val = isl_val_nan(isl_pw_aff_get_ctx(pw_aff));
  if (isl_pw_aff_n_piece(pw_aff) != 1) {
    return val;
  }
  CHECK_GE(isl_pw_aff_foreach_piece(pw_aff, &ExtractCst, &val), 0);
  return val;
}

}  // namespace pypet
}  // namespace pypoly
