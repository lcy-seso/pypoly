#include "pypoly/core/pypet/pypet.h"

#include "pypoly/core/pypet/nest.h"

namespace pypoly {
namespace pypet {

PypetScop* PypetScop::Create(isl_space* space) {
  isl_schedule* schedule = isl_schedule_empty(isl_space_copy(space));
  return Create(space, 0, schedule);
}

PypetScop* PypetScop::Create(isl_space* space, int n, isl_schedule* schedule) {
  CHECK(space);
  CHECK_GE(n, 0);
  CHECK(schedule);

  isl_ctx* ctx = isl_space_get_ctx(space);
  PypetScop* scop = isl_calloc_type(ctx, PypetScop);

  scop->context = isl_set_universe(isl_space_copy(space));
  scop->context_value = isl_set_universe(isl_space_params(space));
  scop->stmts = isl_calloc_array(ctx, PypetStmt*, n);
  scop->schedule = schedule;
  scop->stmt_num = n;
  scop->array_num = 0;
  scop->arrays = nullptr;

  return scop;
}

PypetScop* PypetScopRestrict(PypetScop* scop, isl_set* cond) {
  // TODO(yizhu1): pet_scop_restrict_skip

  scop->context = isl_set_intersect(scop->context, isl_set_copy(cond));
  scop->context =
      isl_set_union(scop->context, isl_set_complement(isl_set_copy(cond)));
  scop->context = isl_set_coalesce(scop->context);
  scop->context = PypetNestedRemoveFromSet(scop->context);

  isl_set_free(cond);
  return scop;
}

PypetScop* PypetScopRestrictContext(PypetScop* scop, isl_set* context) {
  context = PypetNestedRemoveFromSet(context);
  scop->context = isl_set_intersect(scop->context, context);
  return scop;
}

PypetScop* PypetScopAdd(isl_ctx* ctx, isl_schedule* schedule, PypetScop* lhs,
                        PypetScop* rhs) {
  if (lhs->stmt_num == 0) {
    return rhs;
  }

  if (rhs->stmt_num == 0) {
    return lhs;
  }

  isl_space* space = isl_set_get_space(lhs->context);
  PypetScop* ret = PypetScop::Create(space, lhs->stmt_num + rhs->stmt_num,
                                     isl_schedule_copy(schedule));
  ret->arrays =
      isl_calloc_array(ctx, PypetArray*, lhs->array_num + rhs->array_num);
  ret->array_num = lhs->array_num + rhs->array_num;

  for (int i = 0; i < lhs->stmt_num; ++i) {
    ret->stmts[i] = lhs->stmts[i];
    lhs->stmts[i] = nullptr;
  }

  for (int i = 0; i < rhs->stmt_num; ++i) {
    ret->stmts[i + lhs->stmt_num] = rhs->stmts[i];
    rhs->stmts[i] = nullptr;
  }

  for (int i = 0; i < lhs->array_num; ++i) {
    ret->arrays[i] = lhs->arrays[i];
    lhs->arrays[i] = nullptr;
  }

  for (int i = 0; i < rhs->array_num; ++i) {
    ret->arrays[i + lhs->array_num] = rhs->arrays[i];
    rhs->arrays[i] = nullptr;
  }

  ret = ScopCollectImplications(ctx, ret, lhs, rhs);
  ret = PypetScopRestrictContext(ret, isl_set_copy(lhs->context));
  ret = PypetScopRestrictContext(ret, isl_set_copy(rhs->context));
  ret = PypetScopCombineSkips(ret, lhs, rhs);
  ret = PypetScopCombineStartEnd(ret, lhs, rhs);
  ret = PypetScopCollectIndependence(ctx, ret, lhs, rhs);

  return ret;
}

PypetScop* PypetScopAddSeq(isl_ctx* ctx, PypetScop* lhs, PypetScop* rhs) {
  // TODO(yizhu1): break and continue
  isl_schedule* schedule = isl_schedule_sequence(
      isl_schedule_copy(lhs->schedule), isl_schedule_copy(rhs->schedule));
  return PypetScopAdd(ctx, schedule, lhs, rhs);
}

PypetScop* PypetScopAddPar(isl_ctx* ctx, PypetScop* lhs, PypetScop* rhs) {
  isl_schedule* schedule = isl_schedule_set(isl_schedule_copy(lhs->schedule),
                                            isl_schedule_copy(rhs->schedule));
  return PypetScopAdd(ctx, schedule, lhs, rhs);
}

}  // namespace pypet
}  // namespace pypoly
