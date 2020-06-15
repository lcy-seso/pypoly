#include "pypoly/core/pypet/array.h"

#include "pypoly/core/pypet/expr.h"
#include "pypoly/core/pypet/isl_printer.h"
#include "pypoly/core/pypet/tree.h"

namespace pypoly {
namespace pypet {

namespace {  // For scaning array from PypetScop.

__isl_give isl_id_list* ExtractListFromTupleId(
    __isl_keep isl_space* space, const ContextVarTablePtr var_table) {
  isl_id* id = isl_space_get_tuple_id(space, isl_dim_set);

  const char* name = isl_id_get_name(id);
  auto iter = var_table->find(std::string(name));
  if (iter != var_table->end()) {
    return isl_id_list_from_id(id);
  } else {
    // TODO(Ying): integers declared in the scop should also be returned.
  }

  isl_id_free(id);
  isl_ctx* ctx = isl_space_get_ctx(space);
  return isl_id_list_alloc(ctx, 0);
}

__isl_give isl_id_list* ExtractList(__isl_keep isl_space* space,
                                    const ContextVarTablePtr var_table) {
  bool is_warpping = isl_space_is_wrapping(space);
  if (is_warpping < 0) {
    return nullptr;
  }
  if (!is_warpping) {
    return ExtractListFromTupleId(space, var_table);
  }

  space = isl_space_unwrap(isl_space_copy(space));
  isl_space* range = isl_space_range(isl_space_copy(space));
  isl_id_list* list = ExtractList(range, var_table);
  isl_space_free(range);
  space = isl_space_domain(space);
  list = isl_id_list_concat(ExtractList(space, var_table), list);
  isl_space_free(space);
  return list;
}

isl_stat SpaceCollectArrays(__isl_take isl_space* space, void* user) {
  ArrayDescSet* arrays = (ArrayDescSet*)user;
  isl_id_list* list = ExtractList(space, arrays->var_table);

  if ((isl_id_list_n_id(list) > 0 && arrays->find(list) == arrays->end())) {
    arrays->insert(isl_id_list_copy(list));
  }

  isl_id_list_free(list);
  isl_space_free(space);
  return isl_stat_ok;
}

struct PypetForeachDataSpaceData {
  isl_stat (*fn)(__isl_take isl_space* space, void* user);  // callback
  void* user;  // memory to store results of calling the callback
};

/* Given a piece of an access relation, call data->fn on the data
 * (i.e., range) space. */
isl_stat ForeachDataSpace(__isl_take isl_map* map, void* user) {
  PypetForeachDataSpaceData* data =
      static_cast<PypetForeachDataSpaceData*>(user);
  isl_space* space = isl_map_get_space(map);
  space = isl_space_range(space);
  isl_map_free(map);

  return data->fn(space, data->user);
}

isl_stat PypetExprAccessForeachDataSpace(
    __isl_keep PypetExpr* expr,
    isl_stat (*fn)(__isl_take isl_space* space, void* user) /*callback*/,
    void* user /*memory that stores the results*/) {
  struct PypetForeachDataSpaceData data = {fn, user};

  for (int type = PYPET_EXPR_ACCESS_BEGIN; type < PYPET_EXPR_ACCESS_END;
       ++type) {
    if (!expr->acc.access[type]) {
      // skip MAY_READ and FAKE_KILL, only considers MAY_WRITE and MUST_WRITE.
      continue;
    }
    if (isl_union_map_foreach_map(expr->acc.access[type] /*union map*/,
                                  &ForeachDataSpace /*callback*/,
                                  &data /*user data*/) < 0)
      return isl_stat_error;
  }

  isl_space* space = isl_multi_pw_aff_get_space(expr->acc.index);
  space = isl_space_range(space);
  return fn(space, user);
}

int AccessCollectAarrayWrap(__isl_keep PypetExpr* expr, void* user) {
  ArrayDescSet* arrays = static_cast<ArrayDescSet*>(user);
  ArrayDescSet::AccessCollectArrays(expr, *arrays);
  return 0;
}
}  // namespace

struct PypetArray* PypetArrayFree(struct PypetArray* array) {
  if (!array) return nullptr;

  if (array->context) isl_set_free(array->context);
  if (array->extent) isl_set_free(array->extent);
  if (array->value_bounds) isl_set_free(array->value_bounds);
  if (array->element_type) free(array->element_type);
  if (array->element_shape) free(array->element_shape);

  free(array);
  return nullptr;
}

int PypetArrayIsEqual(const struct PypetArray* array1,
                      const struct PypetArray* array2) {
  if (!array1 || !array2) return 0;

  if (!isl_set_is_equal(array1->context, array2->context)) return 0;
  if (!isl_set_is_equal(array1->extent, array2->extent)) return 0;
  if (!!array1->value_bounds != !!array2->value_bounds) return 0;
  if (array1->value_bounds &&
      !isl_set_is_equal(array1->value_bounds, array2->value_bounds))
    return 0;

  if (strcmp(array1->element_type, array2->element_type)) return 0;

  return 1;
}

void ArrayDescSet::AccessCollectArrays(__isl_keep PypetExpr* expr,
                                       ArrayDescSet& arrays) {
  if (PypetExprIsAffine(expr)) return;
  PypetExprAccessForeachDataSpace(expr, &SpaceCollectArrays, &arrays);
}

void ArrayDescSet::ExprCollectArrays(__isl_keep PypetExpr* expr,
                                     ArrayDescSet& arrays) {
  CHECK(expr);
  for (size_t i = 0; i < expr->arg_num; ++i) {
    // a copy of the i-th argument of the `expr` is returned, so
    // it needs to free the returned value.
    PypetExpr* arg = PypetExprGetArg(expr, i);
    ExprCollectArrays(arg, arrays);
    PypetExprFree(arg);
  }

  if (expr->type == PYPET_EXPR_ACCESS) {
    ArrayDescSet::AccessCollectArrays(expr, arrays);
  }
}

void ArrayDescSet::StmtCollectArrays(PypetStmt* stmt, ArrayDescSet& arrays) {
  CHECK(stmt);

  for (size_t i = 0; i < stmt->arg_num; ++i) {
    ArrayDescSet::ExprCollectArrays(stmt->args[i], arrays);
  }
  PypetTreeForeachAccessExpr(stmt->body, &AccessCollectAarrayWrap /*callback*/,
                             &arrays /*user data*/);
}

void ArrayDescSet::PypetScopCollectArrays(PypetScop* scop,
                                          ArrayDescSet& arrays) {
  CHECK(scop);
  for (size_t i = 0; i < scop->stmt_num; ++i) {
    ArrayDescSet::StmtCollectArrays(scop->stmts[i], arrays);
  }
}

__isl_keep PypetArray* ArrayScanner::UpdateArraySize(
    __isl_keep PypetArray* array, int pos, __isl_take isl_pw_aff* size) {
  if (!array) {
    isl_pw_aff_free(size);
    return nullptr;
  }

  isl_set* valid = isl_set_params(isl_pw_aff_nonneg_set(isl_pw_aff_copy(size)));
  array->context = isl_set_intersect(array->context, valid);

  isl_space* dim = isl_set_get_space(array->extent);
  isl_aff* aff = isl_aff_zero_on_domain(isl_local_space_from_space(dim));
  aff = isl_aff_add_coefficient_si(aff, isl_dim_in, pos, 1);
  isl_set* univ = isl_set_universe(isl_aff_get_domain_space(aff));
  isl_pw_aff* index = isl_pw_aff_alloc(univ, aff);

  size = isl_pw_aff_add_dims(size, isl_dim_in,
                             isl_set_dim(array->extent, isl_dim_set));
  isl_id* id = isl_set_get_tuple_id(array->extent);
  size = isl_pw_aff_set_tuple_id(size, isl_dim_in, id);
  isl_set* bound = isl_pw_aff_lt_set(index, size);

  array->extent = isl_set_intersect(array->extent, bound);

  if (!array->context || !array->extent) return PypetArrayFree(array);

  return array;
}

__isl_keep PypetArray* ArrayScanner::SetArrayUpperBounds(
    __isl_keep PypetArray* array) {
  CHECK(array);

  isl_ctx* ctx = isl_set_get_ctx(array->extent);
  isl_id* id = isl_set_get_tuple_id(array->extent);
  CHECK(id);

  const char* name = isl_id_get_name(id);
  int pos = GetVarDescPos(std::string(name));
  CHECK_GT(pos, 0);

  ContextVar var = ctx_desc_->vars()[pos];
  int dim = GetContextVarDim(var);

  CHECK(dim);
  isl_id_free(id);

  for (int i = 0; i < dim; ++i) {
    isl_local_space* local_space =
        isl_local_space_from_space(isl_set_get_space(array->extent));
    isl_val* upper = isl_val_int_from_si(ctx, var.upper_bound()[i]);
    isl_aff* aff = isl_aff_val_on_domain(local_space, upper);
    isl_pw_aff* size = isl_pw_aff_from_aff(aff);
    int dim = isl_pw_aff_dim(size, isl_dim_in);
    size = isl_pw_aff_drop_dims(size, isl_dim_in, 0, dim);
    array = UpdateArraySize(array, i, size);
  }
  return array;
}

int ArrayScanner::GetVarDescPos(const std::string& name) {
  auto iter = var_table_.find(std::string(name));
  if (iter != var_table_.end()) {
    return iter->second;
  } else {
    LOG(WARNING) << name << " is not the context." << std::endl;
    return -1;
  }
}

int ArrayScanner::GetContextVarDim(const ContextVar& var) {
  switch (var.type()) {
    case ContextVarType::INT32:
      return 0U;
    case ContextVarType::INT32_ARRAY:
      return 1U;
    case ContextVarType::TENSOR_ARRAY:
      return var.lower_bound_size();
    default:
      LOG(WARNING) << var.name() << " is not the context." << std::endl;
      return -1;
  }
}

__isl_give PypetArray* ArrayScanner::ExtractArray(isl_ctx* ctx,
                                                  __isl_keep isl_id* id) {
  PypetArray* array = isl_calloc_type(ctx, struct PypetArray);
  CHECK(array);

  const char* name = isl_id_get_name(id);

  int pos = GetVarDescPos(std::string(name));
  CHECK_GT(pos, 0);

  // TODO(Ying): In current implementation, we cannot distingush variable
  // declaration or a name reuse.
  array->declared = 1;

  ContextVar var = ctx_desc_->vars()[pos];
  isl_space* space = isl_space_set_alloc(ctx, 0, GetContextVarDim(var));
  space = isl_space_set_tuple_id(space, isl_dim_set, id);

  array->extent = isl_set_nat_universe(space);

  space = isl_space_params_alloc(ctx, 0);
  // array in `ContextVar` are statically declared.
  array->context = isl_set_universe(space);

  switch (var.type()) {
    case ContextVarType::INT32:
      // scalars are treated as a zero-dimentional array.
      array->element_type = strdup("int");
      array->element_dim = 1;
      break;
    case ContextVarType::INT32_ARRAY:
      array->element_type = strdup("int");
      array->element_dim = var.elem_desc().shape_size();
      CHECK_EQ(array->element_dim, 1U);
      break;
    case ContextVarType::TENSOR_ARRAY:
      array->element_type = strdup("tensor");
      array->element_dim = var.elem_desc().shape_size();
      break;
    default:
      UNIMPLEMENTED();
  }

  array->element_size = 8;  // FIXME(Ying): get right element size.
  array->element_shape = new int[array->element_dim];
  for (size_t i = 0; i < array->element_dim; ++i)
    array->element_shape[i] = var.elem_desc().shape()[i];

  array = SetArrayUpperBounds(array);
  return array;
}

__isl_give PypetArray* ArrayScanner::ExtractArray(
    isl_ctx* ctx, __isl_keep isl_id_list* decls) {
  isl_id* id = isl_id_list_get_id(decls, 0);
  PypetArray* array = ExtractArray(ctx, id);
  isl_id_free(id);

  if (isl_id_list_n_id(decls) > 1) {
    // We do not support nested array declaration in for scope.
    UNIMPLEMENTED();
  }

  return array;
}

__isl_keep PypetScop* ArrayScanner::ScanArrays(isl_ctx* ctx,
                                               __isl_keep PypetScop* scop) {
  CHECK(scop);
  ArrayDescSet arrays(&var_table_);
  ArrayDescSet::PypetScopCollectArrays(scop, arrays);

  if (arrays.size() == 0U) {
    // FIXME(Ying): If users do no use add_array in Python to declare
    // an arrary. `PypetScopCollectArrays` may not find array in scop.
    // Then the execution will hit this branch. Type and shape information
    // need to be handled more systematically in parsing scop.
    return scop;
  }
  scop->arrays = isl_realloc_array(ctx, scop->arrays, struct PypetArray*,
                                   scop->array_num + arrays.size());
  CHECK(scop->arrays);

  size_t i = scop->array_num;
  for (auto it = arrays.begin(); it != arrays.end(); ++it) {
    scop->arrays[i] = ExtractArray(ctx, *it);
    if (!scop->arrays[i]) {
      PypetScop::Free(scop);
      return nullptr;
    }
    scop->array_num++;
    scop->context = isl_set_intersect(scop->context,
                                      isl_set_copy(scop->arrays[i]->context));
    if (!scop->context) {
      PypetScop::Free(scop);
      return nullptr;
    }
    i++;
  }
  return scop;
}

}  // namespace pypet
}  // namespace pypoly
